import math

import torch

from scionc.models.gpt import GPT
from scionc.optim.parametrization import (
    halving_factor,
    schedule_at_step,
    validate_step_scale,
)
from scionc.optim.scion import ScionC
from scionc.ulmos.core import ColNormULMO, GramNewtonSchulzULMO, RowNormULMO, SignULMO
from scionc.ulmos.streaming_svd import StreamingSVDULMO

DEFAULT_TARGET_RMS = {
    "embed": 0.70,
    "hidden": 0.051,
    "out": 0.022,
}
GROUP_NAMES = tuple(DEFAULT_TARGET_RMS)
DEFAULT_COUNT_INCREMENT = 64 * 256
DEFAULT_MOMENTUM_RETENTION = 0.95
DEFAULT_BETA_HALF_LIFE = -DEFAULT_COUNT_INCREMENT / math.log2(
    DEFAULT_MOMENTUM_RETENTION
)
DEFAULT_SHRINKS = {
    "embed": 0.965,
    "hidden": 0.9883333333333333,
    "out": 0.9965,
}
DEFAULT_SHRINK_HALF_LIVES = {
    name: -DEFAULT_COUNT_INCREMENT / math.log2(shrink)
    for name, shrink in DEFAULT_SHRINKS.items()
}
DEFAULT_STEP_SCALE = 1.0
DEFAULT_BASE_ETA = 3.5e-2


def scale_from_coordinate(
    linear: float | None, log2_value: float | None, label: str
) -> float | None:
    if linear is not None and log2_value is not None:
        raise ValueError(f"set either {label} or log2 {label}, not both")
    if log2_value is None:
        return linear
    if not math.isfinite(log2_value):
        raise ValueError(f"invalid log2 {label}: {log2_value}")
    try:
        return 2.0**log2_value
    except OverflowError as exc:
        raise ValueError(f"invalid log2 {label}: {log2_value}") from exc


def resolve_group_step_scale(args, group: str) -> tuple[float, float]:
    peak = scale_from_coordinate(
        getattr(args, f"step_scale_{group}", None),
        getattr(args, f"log2_step_scale_{group}", None),
        f"{group} step scale",
    )
    if peak is None:
        peak = scale_from_coordinate(
            args.step_scale,
            args.log2_step_scale,
            "global step scale",
        )
    if peak is None:
        peak = DEFAULT_STEP_SCALE

    floor = getattr(args, f"min_step_scale_{group}", None)
    if floor is None:
        floor = args.min_step_scale
    if floor is None:
        floor = 0.1 * peak

    peak = validate_step_scale(peak, f"{group} step scale")
    floor = validate_step_scale(floor, f"{group} minimum step scale")
    if floor > peak:
        raise ValueError(
            f"invalid {group} minimum step scale: {floor}; expected <= peak {peak}"
        )
    return peak, floor


def resolve_group_target_rms(args, group: str) -> float:
    target = getattr(args, f"target_rms_{group}", None)
    if target is None:
        target = getattr(args, "target_rms", None)
    if target is None:
        target = DEFAULT_TARGET_RMS[group]
    if target <= 0.0:
        raise ValueError(f"invalid {group} target RMS: {target}")
    return float(target)


def count_increment(args) -> int:
    return args.batch_size * args.block_size * args.grad_accum


def resolve_group_shrink_half_life(args, group: str) -> float:
    half_life = getattr(args, f"shrink_half_life_{group}", None)
    if half_life is None:
        half_life = args.shrink_half_life
    if half_life is None:
        half_life = DEFAULT_SHRINK_HALF_LIVES[group]
    if half_life <= 0.0:
        raise ValueError(f"invalid {group} shrink half-life: {half_life}")
    return half_life


def apply_auto_group_step_scales(
    opt, summary: dict[str, dict[str, float]], args
) -> str:
    parts = []
    for group in opt.param_groups:
        name = group.get("name", "group")
        l1 = summary.get(name, {}).get("l1")
        if l1 is None or not math.isfinite(l1) or l1 <= 0.0:
            continue

        l1 = args.auto_l1_beta * group.get("auto_l1", l1) + (
            1.0 - args.auto_l1_beta
        ) * l1
        group["auto_l1"] = l1

        step_scale = args.auto_action_scale / (l1 * float(group["base_eta"]))
        step_scale = min(
            max(step_scale, group.get("min_step_scale", 0.0)),
            group.get("max_step_scale", step_scale),
        )
        group["peak_step_scale"] = step_scale
        parts.append(f"{name}={step_scale:.3e}")
    return "auto_step_scale " + ", ".join(parts) if parts else ""


def make_hidden_ulmo(args, work_dtype: torch.dtype):
    if args.hidden_ulmo == "gram-ns":
        return GramNewtonSchulzULMO(steps=args.pe_steps, work_dtype=work_dtype)
    return StreamingSVDULMO(
        steps=args.spi_steps,
        ridge=args.spi_ridge,
        refresh_interval=args.spi_refresh_interval,
        refresh_threshold=args.spi_refresh_threshold,
        iteration=args.spi_iteration,
    )


def input_output_tied(model: GPT) -> bool:
    return model.tok_emb.weight is model.lm_head.weight


def optimizer_io_label(model: GPT) -> str:
    return "tied" if input_output_tied(model) else "untied"


def make_edge_ulmo(kind: str):
    if kind == "colnorm":
        return ColNormULMO(transpose=True)
    if kind == "rownorm":
        return RowNormULMO()
    if kind == "sign":
        return SignULMO()
    raise ValueError(f"unsupported edge ULMO: {kind}")


def hidden_params(model: GPT) -> list[torch.Tensor]:
    skip = {id(model.tok_emb.weight), id(model.lm_head.weight)}
    return [p for p in model.parameters() if p.requires_grad and id(p) not in skip]


def group_schedule_ratio(group: dict, scheduled_step_scale: float) -> float:
    peak_step_scale = float(group["peak_step_scale"])
    if peak_step_scale == 0.0:
        return 0.0
    ratio = scheduled_step_scale / peak_step_scale
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError(
            f"invalid schedule ratio for {group.get('name', 'group')}: {ratio}"
        )
    return ratio


def group_action(group: dict, scheduled_step_scale: float) -> tuple[float, float]:
    peak_scale = float(group["peak_step_scale"])
    validate_step_scale(scheduled_step_scale, f"{group.get('name', 'group')} schedule")
    if peak_scale > 0.0 and scheduled_step_scale > peak_scale * (1.0 + 1e-12):
        raise ValueError(
            f"invalid {group.get('name', 'group')} scheduled scale "
            f"{scheduled_step_scale}; expected <= peak {peak_scale}"
        )

    ratio = group_schedule_ratio(group, scheduled_step_scale)
    shrink = float(group["peak_shrink"]) ** ratio
    return shrink, float(group["base_eta"]) * scheduled_step_scale


@torch.no_grad()
def current_group_rms(group: dict) -> float:
    params = [p.detach().float() for p in group["params"] if p.numel()]
    total = sum(p.numel() for p in params)
    if total <= 0:
        return 0.0
    sq = torch.stack(torch._foreach_norm(params)).square().sum()
    return math.sqrt(float(sq) / total)


@torch.no_grad()
def init_from_actions_(groups: list[dict]) -> None:
    for group in groups:
        ulmo = group["ulmo"]
        for p in group["params"]:
            ulmo.init_(p, float(group["target_rms"]))


def action_group_fields(
    name: str, args, delta_tau: int, memory_beta: float
) -> dict:
    peak_step_scale, min_step_scale = resolve_group_step_scale(args, name)
    shrink_half_life = resolve_group_shrink_half_life(args, name)
    peak_shrink = halving_factor(
        delta_tau,
        shrink_half_life,
        f"{name}_shrink_half_life",
    )
    fields = {
        "target_rms": resolve_group_target_rms(args, name),
        "memory_beta": memory_beta,
        "base_eta": DEFAULT_BASE_ETA,
        "peak_step_scale": peak_step_scale,
        "max_step_scale": peak_step_scale,
        "min_step_scale": min_step_scale,
        "peak_shrink": peak_shrink,
        "shrink_half_life": shrink_half_life,
        "rms_solve": args.rms_solve,
        "beta_half_life": args.beta_half_life,
    }
    shrink, eta = group_action(fields, peak_step_scale)
    fields.update(step_scale=peak_step_scale, shrink=shrink, lr=eta)
    return fields


def optimizer_group_specs(model: GPT, args, work_dtype: torch.dtype):
    tied = input_output_tied(model)
    specs = [
        (
            "embed",
            [model.tok_emb.weight],
            SignULMO() if tied else make_edge_ulmo(args.embed_ulmo),
        ),
        ("hidden", hidden_params(model), make_hidden_ulmo(args, work_dtype)),
    ]
    if not tied:
        specs.append(("out", [model.lm_head.weight], make_edge_ulmo(args.out_ulmo)))
    return specs


@torch.no_grad()
def build_optimizer(model: GPT, args, device: torch.device):
    delta_tau = count_increment(args)
    memory_beta = halving_factor(delta_tau, args.beta_half_life, "beta_half_life")
    work_dtype = torch.float16 if device.type == "cuda" else torch.float32
    groups = [
        {
            "name": name,
            "params": params,
            "ulmo": ulmo,
            **action_group_fields(name, args, delta_tau, memory_beta),
        }
        for name, params, ulmo in optimizer_group_specs(model, args, work_dtype)
        if params
    ]
    init_from_actions_(groups)

    hidden_group = next(group for group in groups if group["name"] == "hidden")
    return ScionC(
        groups,
        lr=hidden_group["lr"],
        readout_mu=args.readout_mu,
        memory_beta=memory_beta,
    )


def format_optimizer_schedule(opt) -> str:
    parts = []
    for group in opt.param_groups:
        name = group.get("name", "group")
        peak_step_scale = group.get("peak_step_scale", group["step_scale"])
        min_step_scale = group.get("min_step_scale", 0.0)
        peak_shrink, peak_eta = group_action(group, peak_step_scale)
        min_shrink, min_eta = group_action(group, min_step_scale)
        target_rms = group.get("target_rms", math.nan)
        shrink_half_life = group.get("shrink_half_life", math.inf)
        parts.append(
            f"{name}=(target_rms={target_rms:g},"
            f"s={peak_step_scale:.3g}->{min_step_scale:.3g},"
            f"eta={peak_eta:.3e}->{min_eta:.3e},"
            f"h_shrink={shrink_half_life:.3g},"
            f"shrink={peak_shrink:.6f}->{min_shrink:.6f})"
        )
    return ", ".join(parts)


@torch.no_grad()
def optimizer_rms_state(opt) -> dict[str, dict[str, float]]:
    out = {}
    for group in opt.param_groups:
        target = float(group.get("target_rms", math.nan))
        current = current_group_rms(group)
        out[group.get("name", f"group{len(out)}")] = {
            "param_rms": current,
            "target_rms": target,
            "rms_ratio": current / target if target and target > 0.0 else math.nan,
        }
    return out


def rms_state_text(state: dict[str, dict[str, float]]) -> str:
    if not state:
        return ""
    parts = [
        f"{name}={values['param_rms']:.3g}/{values['target_rms']:.3g}"
        for name, values in state.items()
    ]
    return " | weight_rms " + "; ".join(parts)


def apply_scheduled_etas(
    opt, step: int, max_steps: int, warmup_steps: int, decay_steps: int
) -> dict[str, float]:
    current_etas = {}
    for group in opt.param_groups:
        peak_step_scale = group.get("peak_step_scale", group["step_scale"])
        min_step_scale = group.get("min_step_scale", 0.0)
        step_scale = schedule_at_step(
            step,
            max_steps,
            peak_step_scale,
            min_step_scale,
            warmup_steps,
            decay_steps,
        )
        shrink, eta = group_action(group, step_scale)
        group["step_scale"] = step_scale
        group["shrink"] = shrink
        group["lr"] = eta
        current_etas[group.get("name", f"group{len(current_etas)}")] = eta
    return current_etas
