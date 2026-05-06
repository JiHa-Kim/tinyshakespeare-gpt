import math

import torch

from scionh.models.gpt import GPT


def _rms_float(x: torch.Tensor) -> float:
    return float(x.detach().float().square().mean().sqrt())


@torch.no_grad()
def calibrate_deepnorm_branches(
    model: GPT, idx: torch.Tensor, eps: float = 1e-12
) -> dict[str, float]:
    if model.cfg.block_type != "deepnorm":
        return {}

    target_delta = 1.0 / math.sqrt(2.0 * model.cfg.n_layer)
    was_training = model.training
    model.eval()
    attn_scales = []
    mlp_scales = []
    attn_ratios_before = []
    mlp_ratios_before = []
    attn_ratios_after = []
    mlp_ratios_after = []
    try:
        x = model.tok_emb(idx)
        for block in model.blocks:
            if block.block_type != "deepnorm":
                x = block(x)
                continue
            if block.post_attn_norm is None or block.post_mlp_norm is None:
                raise RuntimeError("DeepNorm post-norms are missing")

            alpha = float(block.deepnorm_alpha)
            attn_out = block.attn(x)
            attn_den = max(alpha * _rms_float(x), eps)
            attn_ratio = _rms_float(attn_out) / attn_den
            attn_scale = target_delta / max(attn_ratio, eps)
            block.deepnorm_attn_scale.fill_(attn_scale)
            x = block.post_attn_norm(alpha * x + attn_scale * attn_out)

            mlp_out = block.mlp(x)
            mlp_den = max(alpha * _rms_float(x), eps)
            mlp_ratio = _rms_float(mlp_out) / mlp_den
            mlp_scale = target_delta / max(mlp_ratio, eps)
            block.deepnorm_mlp_scale.fill_(mlp_scale)
            x = block.post_mlp_norm(alpha * x + mlp_scale * mlp_out)

            attn_scales.append(attn_scale)
            mlp_scales.append(mlp_scale)
            attn_ratios_before.append(attn_ratio)
            mlp_ratios_before.append(mlp_ratio)
            attn_ratios_after.append(attn_ratio * attn_scale)
            mlp_ratios_after.append(mlp_ratio * mlp_scale)
    finally:
        model.train(was_training)

    return {
        "target_delta": target_delta,
        **_stats(attn_scales, "attn_scale"),
        **_stats(mlp_scales, "mlp_scale"),
        **_stats(attn_ratios_before, "attn_ratio_before"),
        **_stats(mlp_ratios_before, "mlp_ratio_before"),
        **_stats(attn_ratios_after, "attn_ratio_after"),
        **_stats(mlp_ratios_after, "mlp_ratio_after"),
    }


def _stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"{prefix}_mean": sum(values) / len(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def deepnorm_calibration_text(stats: dict[str, float]) -> str:
    if not stats:
        return ""
    return (
        "deepnorm_calibration "
        f"target_delta={stats['target_delta']:.6g} "
        f"attn_scale={stats['attn_scale_mean']:.4g}"
        f"[{stats['attn_scale_min']:.4g},{stats['attn_scale_max']:.4g}] "
        f"mlp_scale={stats['mlp_scale_mean']:.4g}"
        f"[{stats['mlp_scale_min']:.4g},{stats['mlp_scale_max']:.4g}] "
        f"attn_ratio_before={stats['attn_ratio_before_mean']:.4g} "
        f"mlp_ratio_before={stats['mlp_ratio_before_mean']:.4g}"
    )
