import math

import torch

from scionc.models.gpt import GPT
from scionc.optim.parametrization import (
    angular_angle,
    angular_step,
    direction_retention,
    resolve_schedule,
    schedule_at_step,
    scheduled_retention,
)
from scionc.optim.scion import RMSSphere
from scionc.ulmos.core import ColNormULMO, GramNewtonSchulzULMO, RowNormULMO, SignULMO
from scionc.ulmos.streaming_svd import StreamingSVDULMO

DEFAULT_TARGET_RMS = {
    "embed": 0.70,
    "hidden": 0.051,
    "out": 0.022,
}
GROUP_NAMES = tuple(DEFAULT_TARGET_RMS)
DEFAULT_COUNT_INCREMENT = 64 * 256
DEFAULT_DIRECTION_HALF_LIFE = 2.214e5  # processed tokens


def count_increment(args) -> int:
    return args.batch_size * args.block_size * args.grad_accum


def resolve_group_target_rms(args, group: str) -> float:
    target = getattr(args, f"target_rms_{group}", None)
    if target is None:
        target = getattr(args, "target_rms", None)
    if target is None:
        target = DEFAULT_TARGET_RMS[group]
    if target <= 0.0:
        raise ValueError(f"invalid {group} target RMS: {target}")
    return float(target)


def resolve_group_direction_half_life(args, group: str) -> float:
    half_life = getattr(args, f"direction_half_life_{group}", None)
    if half_life is None:
        half_life = args.direction_half_life
    if half_life is None:
        half_life = DEFAULT_DIRECTION_HALF_LIFE
    if half_life <= 0.0:
        raise ValueError(f"invalid {group} direction half-life: {half_life}")
    return float(half_life)


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
def init_and_freeze_radii_(groups: list[dict]) -> None:
    """Initialize weights using geometry, then record R = ‖W‖_rms per block."""
    for group in groups:
        geometry = group["ulmo"].geometry
        target_rms = group.get("target_rms")
        if target_rms is not None:
            for p in group["params"]:
                geometry.init_(p, float(target_rms))


@torch.no_grad()
def current_group_rms(group: dict) -> float:
    params = [p.detach().float() for p in group["params"] if p.numel()]
    total = sum(p.numel() for p in params)
    if total <= 0:
        return 0.0
    sq = torch.stack(torch._foreach_norm(params)).square().sum()
    return math.sqrt(float(sq) / total)


@torch.no_grad()
def build_optimizer(model: GPT, args, device: torch.device):
    delta_tau = count_increment(args)
    work_dtype = torch.float16 if device.type == "cuda" else torch.float32

    groups = []
    for name, params, ulmo in optimizer_group_specs(model, args, work_dtype):
        if not params:
            continue
        half_life = resolve_group_direction_half_life(args, name)
        q = direction_retention(delta_tau, half_life, f"{name}_direction_half_life")
        target_rms = resolve_group_target_rms(args, name)
        groups.append({
            "name": name,
            "params": params,
            "ulmo": ulmo,
            "q": q,
            "q_peak": q,
            "direction_half_life": half_life,
            "target_rms": target_rms,
        })

    init_and_freeze_radii_(groups)

    default_q = groups[0]["q"] if groups else 0.99
    return RMSSphere(groups, q=default_q)


def format_optimizer_schedule(opt) -> str:
    parts = []
    for group in opt.param_groups:
        name = group.get("name", "group")
        q = float(group["q_peak"])
        eps = angular_step(q)
        theta = angular_angle(q)
        h = group.get("direction_half_life", float("nan"))
        target_rms = group.get("target_rms", float("nan"))
        parts.append(
            f"{name}=(h={h:.3g},q={q:.6f},eps={eps:.6f},theta={theta:.6f},"
            f"init_rms={target_rms:g})"
        )
    return ", ".join(parts)


@torch.no_grad()
def optimizer_rms_state(opt) -> dict[str, dict[str, float]]:
    out = {}
    for group in opt.param_groups:
        target_rms = float(group.get("target_rms", math.nan))
        current = current_group_rms(group)
        out[group.get("name", f"group{len(out)}")] = {
            "param_rms": current,
            "init_rms": target_rms,
            "rms_ratio": current / target_rms if target_rms and target_rms > 0.0 else math.nan,
        }
    return out


def rms_state_text(state: dict[str, dict[str, float]]) -> str:
    if not state:
        return ""
    parts = [
        f"{name}={values['param_rms']:.3g}/{values['init_rms']:.3g}"
        for name, values in state.items()
    ]
    return " | weight_rms " + "; ".join(parts)


def apply_scheduled_q(
    opt, step: int, max_steps: int, warmup_steps: int, decay_steps: int,
    schedule_floor: float = 0.0,
) -> dict[str, float]:
    """Apply WSD schedule to the halving exponent of each group.

    The schedule ratio s_t ∈ [floor, 1] scales the halving exponent:
        q_t = q_peak^{s_t}

    Returns the current q per group.
    """
    current_qs = {}
    for group in opt.param_groups:
        q_peak = float(group["q_peak"])
        s_t = schedule_at_step(
            step, max_steps, 1.0, schedule_floor, warmup_steps, decay_steps,
        )
        q = scheduled_retention(q_peak, s_t)
        group["q"] = q
        group["schedule_ratio"] = s_t
        current_qs[group.get("name", f"group{len(current_qs)}")] = q
    return current_qs
