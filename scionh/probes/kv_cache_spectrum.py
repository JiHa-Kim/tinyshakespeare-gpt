import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from scionh.models.gpt import (
    BatchSource,
    CharDataset,
    GPT,
    GPTConfig,
    apply_rope,
    maybe_download_tiny_shakespeare,
)
from scionh.training.checkpoints import load_checkpoint


def complex_pairs(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[..., 0::2].float(), x[..., 1::2].float())


def real_pairs(z: torch.Tensor) -> torch.Tensor:
    return torch.stack((z.real, z.imag), dim=-1).flatten(-2)


def causal_attention(
    attn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seqlen, n_head, head_dim = q.shape
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    cos = attn.rope_cos[:seqlen].view(1, 1, seqlen, head_dim // 2)
    sin = attn.rope_sin[:seqlen].view(1, 1, seqlen, head_dim // 2)
    qh = apply_rope(qh, cos, sin)
    kh = apply_rope(kh, cos, sin)
    scores = (qh.float() @ kh.float().transpose(-2, -1)) / math.sqrt(head_dim)
    mask = torch.ones(seqlen, seqlen, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask.view(1, 1, seqlen, seqlen), float("-inf"))
    weights = scores.softmax(dim=-1)
    out = weights.to(dtype=vh.dtype) @ vh
    return weights, out, scores


@torch.no_grad()
def layer_input(model: GPT, idx: torch.Tensor, layer: int) -> torch.Tensor:
    x = model.tok_emb(idx)
    for block in model.blocks[:layer]:
        x = block(x)
    return x


@torch.no_grad()
def qkv_for_layer(model: GPT, idx: torch.Tensor, layer: int):
    block = model.blocks[layer]
    attn = block.attn
    x = layer_input(model, idx, layer)
    x = block.norm1(x)
    if attn.qkv is not None:
        q, k, v = attn.qkv(x).split(x.size(-1), dim=-1)
    elif attn.q is not None and attn.k is not None and attn.v is not None:
        q = attn.q(x)
        k = attn.k(x)
        v = attn.v(x)
    else:
        raise RuntimeError("KV spectrum probe requires full QKV projections")
    bsz, seqlen, d_model = q.shape
    q = q.view(bsz, seqlen, attn.n_head, attn.head_dim)
    k = k.view(bsz, seqlen, attn.n_head, attn.head_dim)
    v = v.view(bsz, seqlen, attn.n_head, attn.head_dim)
    return attn, q, k, v


def value_gram(v: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    bsz, seqlen, n_head, head_dim = v.shape
    value_weight = weights.square().sum(dim=2).transpose(1, 2)
    value_weight = value_weight.repeat_interleave(head_dim, dim=-1)
    x = v.reshape(bsz, seqlen, n_head * head_dim) * value_weight.sqrt()
    return x.flatten(0, 1).T @ x.flatten(0, 1)


def key_grams(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, weights: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    bsz, seqlen, n_head, head_dim = v.shape
    freq_count = head_dim // 2
    q_complex = complex_pairs(q).transpose(1, 2)
    k_complex = complex_pairs(k)
    q_abs2 = q_complex.abs().square()
    v_heads = v.transpose(1, 2).float()
    diff2 = (
        v_heads[:, :, None, :, :] - out.float()[:, :, :, None, :]
    ).square().sum(dim=-1)
    score_weight = weights.square() * diff2 / head_dim
    key_weight = torch.einsum("bhts,bhtj->bhsj", score_weight, q_abs2)
    grams = []
    for freq in range(freq_count):
        x = k_complex[:, :, :, freq] * key_weight[:, :, :, freq].transpose(1, 2).sqrt()
        x = x.flatten(0, 1)
        grams.append(x.conj().T @ x)
    return torch.stack(grams)


@torch.no_grad()
def collect_grams(model: GPT, source: BatchSource, args) -> dict[str, torch.Tensor]:
    block = model.blocks[args.layer]
    if block.attn.attn_type != "softmax":
        raise ValueError("kv_cache_spectrum currently expects softmax attention")
    n_head = block.attn.n_head
    head_dim = block.attn.head_dim
    freq_count = head_dim // 2
    device = next(model.parameters()).device
    g_value = torch.zeros(n_head * head_dim, n_head * head_dim, device=device)
    g_key = torch.zeros(freq_count, n_head, n_head, dtype=torch.complex64, device=device)
    token_count = 0
    batches = source.fixed_batches("val", args.calib_iters, args.calib_seed)
    for xb, _ in batches:
        attn, q, k, v = qkv_for_layer(model, xb, args.layer)
        weights, out, _ = causal_attention(attn, q, k, v)
        g_value += value_gram(v.float(), weights)
        g_key += key_grams(q, k, v, weights, out)
        token_count += xb.numel()
    normalizer = max(token_count, 1)
    return {
        "key": (g_key / normalizer).cpu(),
        "value": (g_value / normalizer).cpu(),
    }


def eigensystem(grams: dict[str, torch.Tensor]):
    key_evals = []
    key_evecs = []
    for gram in grams["key"]:
        vals, vecs = torch.linalg.eigh(gram.to(torch.complex128))
        order = vals.real.argsort(descending=True)
        key_evals.append(vals.real[order].clamp_min(0.0))
        key_evecs.append(vecs[:, order].to(torch.complex64))
    value_vals, value_vecs = torch.linalg.eigh(grams["value"].double())
    value_order = value_vals.argsort(descending=True)
    return {
        "key_evals": torch.stack(key_evals),
        "key_evecs": torch.stack(key_evecs),
        "value_evals": value_vals[value_order].clamp_min(0.0),
        "value_evecs": value_vecs[:, value_order].float(),
    }


def allocate_budget(spec, budget: int) -> tuple[list[int], int, float]:
    key_evals = spec["key_evals"]
    value_evals = spec["value_evals"]
    key_ranks = [0 for _ in range(key_evals.size(0))]
    value_rank = 0
    candidates = []
    for freq in range(key_evals.size(0)):
        for mode, value in enumerate(key_evals[freq]):
            candidates.append(("key", freq, mode, float(value), 2))
    for mode, value in enumerate(value_evals):
        candidates.append(("value", -1, mode, float(value), 1))
    candidates.sort(key=lambda item: item[3] / item[4], reverse=True)
    used = 0
    captured = 0.0
    for kind, freq, mode, value, cost in candidates:
        if used + cost > budget:
            continue
        if kind == "key":
            if mode != key_ranks[freq]:
                continue
            key_ranks[freq] += 1
        else:
            if mode != value_rank:
                continue
            value_rank += 1
        used += cost
        captured += value
        if used >= budget:
            break
    return key_ranks, value_rank, captured


def project_keys(k: torch.Tensor, spec, ranks: list[int]) -> torch.Tensor:
    k_complex = complex_pairs(k)
    pieces = []
    for freq, rank in enumerate(ranks):
        value = k_complex[:, :, :, freq]
        if rank <= 0:
            pieces.append(torch.zeros_like(value))
            continue
        u = spec["key_evecs"][freq, :, :rank].to(device=k.device)
        coeff = value @ u.conj()
        pieces.append(coeff @ u.T)
    return real_pairs(torch.stack(pieces, dim=-1))


def project_values(v: torch.Tensor, spec, rank: int) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros_like(v)
    bsz, seqlen, n_head, head_dim = v.shape
    flat = v.reshape(bsz, seqlen, n_head * head_dim).float()
    u = spec["value_evecs"][:, :rank].to(device=v.device)
    projected = (flat @ u) @ u.T
    return projected.reshape_as(v).to(dtype=v.dtype)


@torch.no_grad()
def reconstruction_error(model: GPT, source: BatchSource, spec, args, budget: int):
    key_ranks, value_rank, captured = allocate_budget(spec, budget)
    total_sq = 0.0
    err_sq = 0.0
    score_err_sq = 0.0
    score_total_sq = 0.0
    batches = source.fixed_batches("val", args.eval_iters, args.eval_seed)
    for xb, _ in batches:
        attn, q, k, v = qkv_for_layer(model, xb, args.layer)
        _, out, scores = causal_attention(attn, q, k, v)
        k_hat = project_keys(k, spec, key_ranks)
        v_hat = project_values(v, spec, value_rank)
        _, out_hat, scores_hat = causal_attention(attn, q, k_hat, v_hat)
        mask = torch.isfinite(scores)
        score_err_sq += float((scores_hat[mask] - scores[mask]).square().sum())
        score_total_sq += float(scores[mask].square().sum())
        err_sq += float((out_hat - out).float().square().sum())
        total_sq += float(out.float().square().sum())
    return {
        "budget": budget,
        "key_ranks": key_ranks,
        "value_rank": value_rank,
        "used_budget": 2 * sum(key_ranks) + value_rank,
        "captured_taylor_energy": captured,
        "relative_output_mse": err_sq / max(total_sq, 1e-30),
        "relative_score_mse": score_err_sq / max(score_total_sq, 1e-30),
    }


def gqa_project(x: torch.Tensor, groups: int) -> torch.Tensor:
    bsz, seqlen, n_head, head_dim = x.shape
    if n_head % groups:
        raise ValueError(f"n_head={n_head} is not divisible by groups={groups}")
    group_size = n_head // groups
    grouped = x.view(bsz, seqlen, groups, group_size, head_dim).mean(dim=3, keepdim=True)
    return grouped.expand(bsz, seqlen, groups, group_size, head_dim).reshape_as(x)


@torch.no_grad()
def gqa_error(model: GPT, source: BatchSource, args, groups: int):
    total_sq = 0.0
    err_sq = 0.0
    score_err_sq = 0.0
    score_total_sq = 0.0
    batches = source.fixed_batches("val", args.eval_iters, args.eval_seed)
    for xb, _ in batches:
        attn, q, k, v = qkv_for_layer(model, xb, args.layer)
        _, out, scores = causal_attention(attn, q, k, v)
        k_hat = gqa_project(k, groups)
        v_hat = gqa_project(v, groups)
        _, out_hat, scores_hat = causal_attention(attn, q, k_hat, v_hat)
        mask = torch.isfinite(scores)
        score_err_sq += float((scores_hat[mask] - scores[mask]).square().sum())
        score_total_sq += float(scores[mask].square().sum())
        err_sq += float((out_hat - out).float().square().sum())
        total_sq += float(out.float().square().sum())
    head_dim = model.blocks[args.layer].attn.head_dim
    return {
        "groups": groups,
        "budget": 2 * groups * head_dim,
        "relative_output_mse": err_sq / max(total_sq, 1e-30),
        "relative_score_mse": score_err_sq / max(score_total_sq, 1e-30),
    }


def parse_budgets(text: str, original_dim: int) -> list[int]:
    budgets = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        budget = int(round(value * original_dim)) if value <= 1.0 else int(value)
        budgets.append(max(1, min(original_dim, budget)))
    return sorted(set(budgets))


def parse_ints(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def cumulative(values: torch.Tensor, ranks: list[int]) -> list[float]:
    result = []
    total = float(values.sum())
    for rank in ranks:
        result.append(float(values[:rank].sum() / max(total, 1e-30)))
    return result


def summarize_spectrum(spec, original_dim: int) -> dict:
    value_evals = spec["value_evals"]
    key_evals = spec["key_evals"]
    return {
        "original_cache_real_dim": original_dim,
        "value_top10": [float(x) for x in value_evals[:10]],
        "value_cumulative": cumulative(value_evals, [1, 2, 4, 8, 16, 32]),
        "key_top5_by_freq": [
            [float(x) for x in key_evals[freq, :5]] for freq in range(key_evals.size(0))
        ],
        "key_cumulative_by_freq": [
            cumulative(key_evals[freq], [1, 2, 4, 8])
            for freq in range(key_evals.size(0))
        ],
    }


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


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="")
    p.add_argument("--data-path", default="data/tiny_shakespeare.txt")
    p.add_argument("--device", default="")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--calib-iters", type=int, default=8)
    p.add_argument("--eval-iters", type=int, default=4)
    p.add_argument("--calib-seed", type=int, default=20260506)
    p.add_argument("--eval-seed", type=int, default=20260507)
    p.add_argument("--budgets", default="0.125,0.25,0.5,0.75")
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
    if args.layer < 0 or args.layer >= len(model.blocks):
        raise ValueError(f"invalid layer {args.layer}; model has {len(model.blocks)} blocks")
    source = BatchSource(
        dataset.train,
        dataset.val,
        model.cfg.block_size,
        args.batch_size,
        device,
        train_seed=args.calib_seed,
        val_seed=args.eval_seed,
    )
    grams = collect_grams(model, source, args)
    spec = eigensystem(grams)
    n_head = model.blocks[args.layer].attn.n_head
    head_dim = model.blocks[args.layer].attn.head_dim
    original_dim = 2 * n_head * head_dim
    budgets = parse_budgets(args.budgets, original_dim)
    budget_results = [
        reconstruction_error(model, source, spec, args, budget) for budget in budgets
    ]
    gqa_results = [
        gqa_error(model, source, args, groups) for groups in parse_ints(args.gqa_groups)
    ]
    result = {
        "checkpoint": args.checkpoint,
        "layer": args.layer,
        "n_head": n_head,
        "head_dim": head_dim,
        "calib_iters": args.calib_iters,
        "eval_iters": args.eval_iters,
        "calib_seed": args.calib_seed,
        "eval_seed": args.eval_seed,
        "spectrum": summarize_spectrum(spec, original_dim),
        "budgets": budget_results,
        "gqa": gqa_results,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
