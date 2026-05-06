import argparse
import json
import math
from pathlib import Path

import torch

from scionc.models.gpt import BatchSource, GPT, apply_rope
from scionc.optim.setup import build_optimizer
from scionc.train_shakespeare import (
    build_model,
    calibrate_deepnorm_branches,
    configure_derf_training,
    configure_runtime,
    load_dataset,
    make_parser,
    resolve_data_seed,
    resolve_hyperball_update,
)


def _rms(x: torch.Tensor) -> float:
    return float(x.detach().float().square().mean().sqrt())


def _effective_rank(x: torch.Tensor) -> dict[str, float]:
    flat = x.detach().float().flatten(0, -2)
    flat = flat - flat.mean(dim=0, keepdim=True)
    if flat.size(0) < 2:
        return {"entropy_rank": math.nan, "participation_rank": math.nan, "top_frac": math.nan}

    cov = flat.T @ flat / (flat.size(0) - 1)
    vals = torch.linalg.eigvalsh(cov).clamp_min(0)
    total = vals.sum()
    if float(total) <= 0.0:
        return {"entropy_rank": 0.0, "participation_rank": 0.0, "top_frac": math.nan}

    probs = vals / total
    nz = probs > 0
    entropy = -(probs[nz] * probs[nz].log()).sum()
    participation = total.square() / vals.square().sum().clamp_min(1e-30)
    return {
        "entropy_rank": float(entropy.exp()),
        "participation_rank": float(participation),
        "top_frac": float(vals.max() / total),
    }


@torch.no_grad()
def _attention_simplex_stats(module, x: torch.Tensor) -> dict[str, float]:
    if module.attn_type != "softmax" or module.kv_cache != "full":
        return {}
    if module.qkv is None:
        return {}

    bsz, seqlen, d_model = x.shape
    q, k, _v = module.qkv(x).split(d_model, dim=-1)
    q = q.view(bsz, seqlen, module.n_head, module.head_dim).transpose(1, 2)
    k = k.view(bsz, seqlen, module.n_head, module.head_dim).transpose(1, 2)
    cos = module.rope_cos[:seqlen].view(1, 1, seqlen, module.head_dim // 2)
    sin = module.rope_sin[:seqlen].view(1, 1, seqlen, module.head_dim // 2)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(module.head_dim)
    mask = module.causal_mask[:seqlen, :seqlen].view(1, 1, seqlen, seqlen)
    weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    log_weights = weights.clamp_min(1e-30).log()
    entropy = -(weights * log_weights).sum(dim=-1)
    row_sizes = torch.arange(1, seqlen + 1, device=x.device, dtype=entropy.dtype)
    normalized_entropy = entropy / row_sizes.log().clamp_min(1.0)
    effective_support = entropy.exp() / row_sizes
    return {
        "attn_entropy": float(entropy.mean()),
        "attn_entropy_norm": float(normalized_entropy.mean()),
        "attn_eff_support_frac": float(effective_support.mean()),
        "attn_max_prob": float(weights.amax(dim=-1).mean()),
        "attn_score_rms": float(scores.masked_select(mask).square().mean().sqrt()),
    }


def collect_depth_scaling_diagnostics(
    model: GPT, idx: torch.Tensor, targets: torch.Tensor
) -> dict:
    records: dict[int, dict] = {}
    handles = []

    def pre_hook(layer: int):
        def hook(_module, inputs):
            x = inputs[0]
            x.retain_grad()
            records[layer] = {"input": x}

        return hook

    def fwd_hook(layer: int):
        def hook(module, _inputs, output):
            output.retain_grad()
            records[layer].update(
                {
                    "output": output,
                    "block_type": module.block_type,
                    "deepnorm_alpha": module.deepnorm_alpha,
                    "lns_scale": module.lns_scale,
                }
            )

        return hook

    def attn_pre_hook(layer: int):
        def hook(module, inputs):
            records[layer].update(_attention_simplex_stats(module, inputs[0]))

        return hook

    def attn_out_hook(layer: int):
        def hook(_module, _inputs, output):
            records[layer]["attn_out"] = output

        return hook

    def mlp_pre_hook(layer: int):
        def hook(_module, inputs):
            records[layer]["mlp_input"] = inputs[0]

        return hook

    def mlp_out_hook(layer: int):
        def hook(_module, _inputs, output):
            records[layer]["mlp_out"] = output

        return hook

    for layer, block in enumerate(model.blocks):
        handles.append(block.register_forward_pre_hook(pre_hook(layer)))
        handles.append(block.register_forward_hook(fwd_hook(layer)))
        handles.append(block.attn.register_forward_pre_hook(attn_pre_hook(layer)))
        handles.append(block.attn.register_forward_hook(attn_out_hook(layer)))
        handles.append(block.mlp.register_forward_pre_hook(mlp_pre_hook(layer)))
        handles.append(block.mlp.register_forward_hook(mlp_out_hook(layer)))

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        _logits, loss = model(idx, targets)
        if loss is None:
            raise RuntimeError("depth diagnostics require targets")
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    layers = []
    for layer in sorted(records):
        rec = records[layer]
        x = rec["input"]
        y = rec["output"]
        gx = x.grad
        gy = y.grad
        x_rms = _rms(x)
        y_rms = _rms(y)
        delta_rms = _rms(y - x)
        attn_branch_ratio = math.nan
        mlp_branch_ratio = math.nan
        attn_scale = math.nan
        mlp_scale = math.nan
        if rec["block_type"] == "deepnorm":
            block = model.blocks[layer]
            attn_scale = float(block.deepnorm_attn_scale)
            mlp_scale = float(block.deepnorm_mlp_scale)
            alpha = float(rec["deepnorm_alpha"])
            attn_out = rec.get("attn_out")
            if attn_out is not None and x_rms > 0.0:
                attn_branch_ratio = attn_scale * _rms(attn_out) / (alpha * x_rms)
            mlp_input = rec.get("mlp_input")
            mlp_out = rec.get("mlp_out")
            if mlp_input is not None and mlp_out is not None:
                mlp_input_rms = _rms(mlp_input)
                if mlp_input_rms > 0.0:
                    mlp_branch_ratio = (
                        mlp_scale * _rms(mlp_out) / (alpha * mlp_input_rms)
                    )
        grad_in_rms = _rms(gx) if gx is not None else math.nan
        grad_out_rms = _rms(gy) if gy is not None else math.nan
        grad_transport = (
            grad_in_rms / grad_out_rms
            if grad_out_rms and math.isfinite(grad_out_rms)
            else math.nan
        )
        ranks = _effective_rank(y)
        layers.append(
            {
                "layer": layer,
                "block_type": rec["block_type"],
                "deepnorm_alpha": rec["deepnorm_alpha"],
                "lns_scale": rec["lns_scale"],
                "input_rms": x_rms,
                "output_rms": y_rms,
                "delta_rms": delta_rms,
                "delta_ratio": delta_rms / x_rms if x_rms > 0.0 else math.nan,
                "attn_scale": attn_scale,
                "mlp_scale": mlp_scale,
                "attn_branch_ratio": attn_branch_ratio,
                "mlp_branch_ratio": mlp_branch_ratio,
                "grad_in_rms": grad_in_rms,
                "grad_out_rms": grad_out_rms,
                "grad_transport": grad_transport,
                **ranks,
                "attn_entropy": rec.get("attn_entropy", math.nan),
                "attn_entropy_norm": rec.get("attn_entropy_norm", math.nan),
                "attn_eff_support_frac": rec.get("attn_eff_support_frac", math.nan),
                "attn_max_prob": rec.get("attn_max_prob", math.nan),
                "attn_score_rms": rec.get("attn_score_rms", math.nan),
            }
        )

    finite_transport = [
        max(1e-30, row["grad_transport"])
        for row in layers
        if math.isfinite(row["grad_transport"])
    ]
    finite_rank_frac = [
        row["entropy_rank"] / model.cfg.d_model
        for row in layers
        if math.isfinite(row["entropy_rank"])
    ]
    finite_attn_entropy = [
        row["attn_entropy_norm"]
        for row in layers
        if math.isfinite(row["attn_entropy_norm"])
    ]
    finite_attn_branch = [
        row["attn_branch_ratio"]
        for row in layers
        if math.isfinite(row["attn_branch_ratio"])
    ]
    finite_mlp_branch = [
        row["mlp_branch_ratio"]
        for row in layers
        if math.isfinite(row["mlp_branch_ratio"])
    ]
    summary = {
        "loss": float(loss.detach()),
        "layers": len(layers),
        "d_model": model.cfg.d_model,
        "block_type": model.cfg.block_type,
        "deepnorm_alpha": model.cfg.resolved_deepnorm_alpha,
        "lns": model.cfg.lns,
        "grad_transport_geomean": math.exp(
            sum(math.log(x) for x in finite_transport) / len(finite_transport)
        )
        if finite_transport
        else math.nan,
        "grad_transport_min": min(finite_transport) if finite_transport else math.nan,
        "grad_transport_max": max(finite_transport) if finite_transport else math.nan,
        "entropy_rank_frac_min": min(finite_rank_frac) if finite_rank_frac else math.nan,
        "entropy_rank_frac_last": finite_rank_frac[-1] if finite_rank_frac else math.nan,
        "attn_entropy_norm_mean": sum(finite_attn_entropy) / len(finite_attn_entropy)
        if finite_attn_entropy
        else math.nan,
        "attn_entropy_norm_min": min(finite_attn_entropy)
        if finite_attn_entropy
        else math.nan,
        "attn_branch_ratio_mean": sum(finite_attn_branch) / len(finite_attn_branch)
        if finite_attn_branch
        else math.nan,
        "mlp_branch_ratio_mean": sum(finite_mlp_branch) / len(finite_mlp_branch)
        if finite_mlp_branch
        else math.nan,
    }
    return {"summary": summary, "layers": layers}


def print_depth_scaling_report(report: dict, sample_layers: int = 8) -> None:
    summary = report["summary"]
    print(
        "depth_probe "
        f"loss={summary['loss']:.4f} block_type={summary['block_type']} "
        f"layers={summary['layers']} d_model={summary['d_model']} "
        f"deepnorm_alpha={summary['deepnorm_alpha']:.6g} "
        f"lns={summary['lns']} "
        f"grad_transport_geomean={summary['grad_transport_geomean']:.4g} "
        f"grad_transport_min={summary['grad_transport_min']:.4g} "
        f"grad_transport_max={summary['grad_transport_max']:.4g} "
        f"rank_frac_min={summary['entropy_rank_frac_min']:.4g} "
        f"rank_frac_last={summary['entropy_rank_frac_last']:.4g} "
        f"attn_entropy_norm_mean={summary['attn_entropy_norm_mean']:.4g} "
        f"attn_entropy_norm_min={summary['attn_entropy_norm_min']:.4g} "
        f"attn_branch_ratio_mean={summary['attn_branch_ratio_mean']:.4g} "
        f"mlp_branch_ratio_mean={summary['mlp_branch_ratio_mean']:.4g}"
    )
    layers = report["layers"]
    if sample_layers <= 0:
        return
    if len(layers) <= sample_layers:
        selected = layers
    else:
        idxs = torch.linspace(0, len(layers) - 1, sample_layers).round().long().tolist()
        selected = [layers[i] for i in dict.fromkeys(idxs)]
    for row in selected:
        print(
            "depth_layer "
            f"layer={row['layer']} in_rms={row['input_rms']:.4g} "
            f"out_rms={row['output_rms']:.4g} delta_ratio={row['delta_ratio']:.4g} "
            f"lns_scale={row['lns_scale']:.4g} "
            f"attn_branch_ratio={row['attn_branch_ratio']:.4g} "
            f"mlp_branch_ratio={row['mlp_branch_ratio']:.4g} "
            f"grad_transport={row['grad_transport']:.4g} "
            f"entropy_rank={row['entropy_rank']:.4g} top_frac={row['top_frac']:.4g} "
            f"attn_entropy_norm={row['attn_entropy_norm']:.4g} "
            f"attn_support_frac={row['attn_eff_support_frac']:.4g}"
        )


def make_depth_parser() -> argparse.ArgumentParser:
    parser = make_parser()
    parser.description = (
        "Probe depth-scaling invariants: forward state scale, backward transport, "
        "and representation effective rank."
    )
    parser.set_defaults(mode="train", no_save=True, skip_sample=True, compile=False)
    parser.add_argument("--probe-split", choices=["train", "val"], default="train")
    parser.add_argument("--probe-json", default="")
    parser.add_argument("--probe-sample-layers", type=int, default=8)
    return parser


def main() -> None:
    args = make_depth_parser().parse_args()
    args.hyperball_update = resolve_hyperball_update(args)
    device, _amp_dtype = configure_runtime(args)
    dataset = load_dataset(args)
    model = build_model(args, dataset, device)
    configure_derf_training(model, args)
    # Reuse the training initializer so the probe sees Scion's target RMS geometry.
    build_optimizer(model, args, device)
    source = BatchSource(
        dataset.train,
        dataset.val,
        args.block_size,
        args.batch_size,
        device,
        train_seed=resolve_data_seed(args),
        val_seed=args.eval_seed if args.eval_seed is not None else args.seed + 1,
    )
    if args.deepnorm_calibrate_branches:
        calib_idx, _ = source.seeded_batch(args.probe_split, resolve_data_seed(args))
        calibration = calibrate_deepnorm_branches(model, calib_idx)
        if calibration:
            print(
                "deepnorm_calibration "
                f"target_delta={calibration['target_delta']:.6g} "
                f"attn_scale_mean={calibration['attn_scale_mean']:.4g} "
                f"mlp_scale_mean={calibration['mlp_scale_mean']:.4g}"
            )
    idx, targets = source.seeded_batch(args.probe_split, resolve_data_seed(args))
    report = collect_depth_scaling_diagnostics(model, idx, targets)
    print_depth_scaling_report(report, args.probe_sample_layers)
    if args.probe_json:
        path = Path(args.probe_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
