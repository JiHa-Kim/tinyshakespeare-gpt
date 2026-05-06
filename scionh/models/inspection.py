import torch

from scionh.models.gpt import Derf, EquivariantLowRankKV, GPT, RMSNorm


def kv_cache_summary(model: GPT) -> dict[str, float | int | str]:
    modules = [m for m in model.modules() if isinstance(m, EquivariantLowRankKV)]
    if not modules:
        attn = model.blocks[0].attn
        original = 2 * attn.n_head * attn.head_dim
        return {
            "type": "full",
            "cache_dim": original,
            "original_cache_dim": original,
            "cache_ratio": 1.0,
            "key_rank": attn.n_head,
            "value_rank": attn.n_head * attn.head_dim,
        }
    module = modules[0]
    return {
        "type": "equivariant-lowrank",
        "cache_dim": module.cache_dim,
        "original_cache_dim": module.original_cache_dim,
        "cache_ratio": module.cache_dim / module.original_cache_dim,
        "key_rank": module.key_rank,
        "value_rank": module.value_rank,
    }


def parameter_summary(model: GPT) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


@torch.no_grad()
def kv_decoder_state(model: GPT) -> dict[str, float]:
    key_values = []
    value_values = []
    for module in model.modules():
        if isinstance(module, EquivariantLowRankKV):
            key_values.append(module.key_decoder.detach().float().flatten())
            value_values.append(module.value_decoder.detach().float().flatten())
    if not key_values:
        return {}
    key = torch.cat(key_values)
    value = torch.cat(value_values)
    return {
        "key_decoder_rms": float(key.square().mean().sqrt()),
        "key_decoder_min": float(key.min()),
        "key_decoder_max": float(key.max()),
        "value_decoder_rms": float(value.square().mean().sqrt()),
        "value_decoder_min": float(value.min()),
        "value_decoder_max": float(value.max()),
    }


def kv_decoder_state_text(state: dict[str, float]) -> str:
    if not state:
        return ""
    return (
        " | kv_decoder "
        f"k_rms={state['key_decoder_rms']:.3g},"
        f"k=[{state['key_decoder_min']:.3g},{state['key_decoder_max']:.3g}],"
        f"v_rms={state['value_decoder_rms']:.3g},"
        f"v=[{state['value_decoder_min']:.3g},{state['value_decoder_max']:.3g}]"
    )


@torch.no_grad()
def derf_state(model: GPT) -> dict[str, float]:
    alphas = []
    shifts = []
    gammas = []
    betas = []
    for module in model.modules():
        if isinstance(module, Derf):
            alphas.append(float(module.alpha))
            shifts.append(float(module.shift))
            gammas.extend(float(x) for x in module.gamma.detach().flatten())
            betas.extend(float(x) for x in module.beta.detach().flatten())
    if not alphas:
        return {}
    return {
        "alpha_mean": sum(alphas) / len(alphas),
        "alpha_min": min(alphas),
        "alpha_max": max(alphas),
        "shift_mean": sum(shifts) / len(shifts),
        "shift_min": min(shifts),
        "shift_max": max(shifts),
        "gamma_mean": sum(gammas) / len(gammas),
        "gamma_min": min(gammas),
        "gamma_max": max(gammas),
        "beta_mean": sum(betas) / len(betas),
        "beta_min": min(betas),
        "beta_max": max(betas),
    }


def derf_state_text(state: dict[str, float]) -> str:
    if not state:
        return ""
    return (
        " | derf "
        f"a={state['alpha_mean']:.3g}"
        f"[{state['alpha_min']:.3g},{state['alpha_max']:.3g}],"
        f"s={state['shift_mean']:.3g}"
        f"[{state['shift_min']:.3g},{state['shift_max']:.3g}],"
        f"g={state['gamma_mean']:.3g}"
        f"[{state['gamma_min']:.3g},{state['gamma_max']:.3g}],"
        f"b={state['beta_mean']:.3g}"
        f"[{state['beta_min']:.3g},{state['beta_max']:.3g}]"
    )


@torch.no_grad()
def rmsnorm_affine_state(model: GPT) -> dict[str, float]:
    values = []
    final_values = []
    block_values = []
    for name, module in model.named_modules():
        if not isinstance(module, RMSNorm) or module.gamma is None:
            continue
        gamma = module.gamma.detach().float().flatten()
        values.append(gamma)
        if name == "norm_f":
            final_values.append(gamma)
        else:
            block_values.append(gamma)
    if not values:
        return {}

    return {
        **_tensor_stats("gamma", values),
        **_tensor_stats("block_gamma", block_values),
        **_tensor_stats("final_gamma", final_values),
    }


def _tensor_stats(prefix: str, parts: list[torch.Tensor]) -> dict[str, float]:
    if not parts:
        return {}
    x = torch.cat(parts)
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_rms": float(x.square().mean().sqrt()),
        f"{prefix}_min": float(x.min()),
        f"{prefix}_max": float(x.max()),
    }


def rmsnorm_affine_state_text(state: dict[str, float]) -> str:
    if not state:
        return ""
    final = ""
    if "final_gamma_rms" in state:
        final = f",final_rms={state['final_gamma_rms']:.3g}"
    return (
        " | norm_affine "
        f"g_mean={state['gamma_mean']:.3g},g_rms={state['gamma_rms']:.3g},"
        f"g=[{state['gamma_min']:.3g},{state['gamma_max']:.3g}]"
        f"{final}"
    )
