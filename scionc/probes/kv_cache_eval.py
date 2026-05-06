import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from scionc.probes.kv_cache_spectrum import (
    allocate_budget,
    causal_attention,
    collect_grams,
    eigensystem,
    gqa_project,
    parse_budgets,
    project_keys,
    project_values,
)
from scionc.models.gpt import (
    BatchSource,
    CharDataset,
    GPT,
    GPTConfig,
    maybe_download_tiny_shakespeare,
)
from scionc.train_shakespeare import load_checkpoint


def amp_ctx(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def parse_layers(text: str, n_layer: int) -> list[int]:
    if text == "all":
        return list(range(n_layer))
    layers = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        layer = int(part)
        if layer < 0 or layer >= n_layer:
            raise ValueError(f"invalid layer {layer}; model has {n_layer} layers")
        layers.append(layer)
    return sorted(set(layers))


def load_or_init_model(args, dataset: CharDataset, device: torch.device) -> GPT:
    if args.checkpoint:
        model, _, _ = load_checkpoint(Path(args.checkpoint), device)
        return model
    cfg = GPTConfig(
        vocab_size=len(dataset.chars),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=0.0,
    )
    return GPT(cfg).to(device)


def qkv_from_attention(attn, x: torch.Tensor):
    q = attn.q(x)
    k = attn.k(x)
    v = attn.v(x)
    bsz, seqlen, _ = q.shape
    q = q.view(bsz, seqlen, attn.n_head, attn.head_dim)
    k = k.view(bsz, seqlen, attn.n_head, attn.head_dim)
    v = v.view(bsz, seqlen, attn.n_head, attn.head_dim)
    return q, k, v


def spectral_attention(attn, x: torch.Tensor, spec, key_ranks: list[int], value_rank: int):
    q, k, v = qkv_from_attention(attn, x)
    k_hat = project_keys(k, spec, key_ranks)
    v_hat = project_values(v, spec, value_rank)
    _, out, _ = causal_attention(attn, q, k_hat, v_hat)
    bsz, seqlen, n_head, head_dim = v.shape
    y = out.transpose(1, 2).contiguous().view(bsz, seqlen, n_head * head_dim)
    return attn.dropout(attn.proj(y))


def gqa_attention(attn, x: torch.Tensor, groups: int):
    q, k, v = qkv_from_attention(attn, x)
    k_hat = gqa_project(k, groups)
    v_hat = gqa_project(v, groups)
    _, out, _ = causal_attention(attn, q, k_hat, v_hat)
    bsz, seqlen, n_head, head_dim = v.shape
    y = out.transpose(1, 2).contiguous().view(bsz, seqlen, n_head * head_dim)
    return attn.dropout(attn.proj(y))


def model_logits(
    model: GPT,
    idx: torch.Tensor,
    spectral_layers: dict[int, tuple[dict, list[int], int]] | None = None,
    gqa_layers: dict[int, int] | None = None,
) -> torch.Tensor:
    spectral_layers = spectral_layers or {}
    gqa_layers = gqa_layers or {}
    x = model.tok_emb(idx)
    for layer, block in enumerate(model.blocks):
        h = block.norm1(x)
        if layer in spectral_layers:
            spec, key_ranks, value_rank = spectral_layers[layer]
            attn_out = spectral_attention(block.attn, h, spec, key_ranks, value_rank)
        elif layer in gqa_layers:
            attn_out = gqa_attention(block.attn, h, gqa_layers[layer])
        else:
            attn_out = block.attn(h)
        x = x + attn_out
        x = x + block.mlp(block.norm2(x))
    return model.lm_head(model.norm_f(x))


@torch.no_grad()
def eval_loss(
    model: GPT,
    batches,
    device: torch.device,
    spectral_layers: dict[int, tuple[dict, list[int], int]] | None = None,
    gqa_layers: dict[int, int] | None = None,
) -> float:
    total = 0.0
    with amp_ctx(device):
        for xb, yb in batches:
            logits = model_logits(model, xb, spectral_layers, gqa_layers)
            loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten())
            total += float(loss)
    return total / len(batches)


def layer_specs(model: GPT, source: BatchSource, args, layers: list[int], budget: int):
    specs = {}
    for layer in layers:
        args.layer = layer
        grams = collect_grams(model, source, args)
        spec = eigensystem(grams)
        key_ranks, value_rank, _ = allocate_budget(spec, budget)
        specs[layer] = (spec, key_ranks, value_rank)
    return specs


def parse_ints(text: str) -> list[int]:
    result = []
    for part in text.split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    return result


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="")
    p.add_argument("--data-path", default="data/tiny_shakespeare.txt")
    p.add_argument("--device", default="")
    p.add_argument("--layers", default="3")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--calib-iters", type=int, default=8)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--calib-seed", type=int, default=20260506)
    p.add_argument("--eval-seed", type=int, default=20260507)
    p.add_argument("--budgets", default="0.5,0.75")
    p.add_argument("--gqa-groups", default="1,2,3")
    p.add_argument("--json-out", default="")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--d-model", type=int, default=384)
    return p


@torch.inference_mode()
def main():
    args = make_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(1337)
    data_path = Path(args.data_path)
    maybe_download_tiny_shakespeare(data_path)
    dataset = CharDataset(data_path)
    model = load_or_init_model(args, dataset, device)
    model.eval()
    layers = parse_layers(args.layers, len(model.blocks))
    source = BatchSource(
        dataset.train,
        dataset.val,
        model.cfg.block_size,
        args.batch_size,
        device,
        train_seed=args.calib_seed,
        val_seed=args.eval_seed,
    )
    eval_batches = source.fixed_batches("val", args.eval_iters, args.eval_seed)
    n_head = model.blocks[0].attn.n_head
    head_dim = model.blocks[0].attn.head_dim
    original_dim = 2 * n_head * head_dim
    budgets = parse_budgets(args.budgets, original_dim)
    baseline = eval_loss(model, eval_batches, device)
    spectral = []
    for budget in budgets:
        specs = layer_specs(model, source, args, layers, budget)
        loss = eval_loss(model, eval_batches, device, spectral_layers=specs)
        spectral.append(
            {
                "budget": budget,
                "loss": loss,
                "delta_loss": loss - baseline,
                "layers": [
                    {
                        "layer": layer,
                        "key_ranks": specs[layer][1],
                        "value_rank": specs[layer][2],
                        "used_budget": 2 * sum(specs[layer][1]) + specs[layer][2],
                    }
                    for layer in layers
                ],
            }
        )
    gqa = []
    for groups in parse_ints(args.gqa_groups):
        if n_head % groups:
            continue
        loss = eval_loss(
            model,
            eval_batches,
            device,
            gqa_layers={layer: groups for layer in layers},
        )
        gqa.append(
            {
                "groups": groups,
                "budget": 2 * groups * head_dim,
                "loss": loss,
                "delta_loss": loss - baseline,
            }
        )
    result = {
        "checkpoint": args.checkpoint,
        "layers": layers,
        "eval_iters": args.eval_iters,
        "calib_iters": args.calib_iters,
        "calib_seed": args.calib_seed,
        "eval_seed": args.eval_seed,
        "original_cache_real_dim": original_dim,
        "baseline_loss": baseline,
        "spectral": spectral,
        "gqa": gqa,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
