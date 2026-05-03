import math
from dataclasses import dataclass

__all__ = [
    "RMSMatchedAction",
    "ema_atom_correlation_factor",
    "halving_factor",
    "resolve_schedule",
    "rms_matched_eta",
    "rms_matched_radius",
    "schedule_at_step",
    "scheduled_rms_matched_action",
    "scheduled_weight_retention",
    "validate_step_scale",
    "weight_retention_for_rms_eta",
]


@dataclass(frozen=True, slots=True)
class RMSMatchedAction:
    rms_radius: float
    rho: float
    weight_retention: float
    eta: float
    step_scale: float
    atom_correlation: float


def halving_factor(delta_tau: float, half_life: float, name: str) -> float:
    if delta_tau <= 0.0:
        raise ValueError(f"invalid count increment: {delta_tau}")
    if half_life <= 0.0:
        raise ValueError(f"invalid {name}: {half_life}")
    if math.isinf(half_life):
        return 1.0
    return 2.0 ** (-delta_tau / half_life)


def resolve_schedule(
    max_steps: int, warmup_steps: int, decay_steps: int
) -> tuple[int, int, int]:
    if max_steps <= 0:
        raise ValueError(f"invalid max_steps: {max_steps}")
    warmup_steps = max(0, min(warmup_steps, max_steps))
    decay_steps = max(0, min(decay_steps, max_steps - warmup_steps))
    stable_steps = max_steps - warmup_steps - decay_steps
    return warmup_steps, stable_steps, decay_steps


def schedule_at_step(
    step: int,
    max_steps: int,
    peak: float,
    floor: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Return the per-update schedule value for a piecewise-constant update."""
    warmup_steps, stable_steps, decay_steps = resolve_schedule(
        max_steps, warmup_steps, decay_steps
    )

    if warmup_steps > 0 and step < warmup_steps:
        return peak * (step + 1) / warmup_steps

    decay_start = warmup_steps + stable_steps
    if decay_steps == 0 or step < decay_start:
        return peak
    if decay_steps == 1:
        return floor

    progress = (step - decay_start) / (decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return peak + (floor - peak) * progress


def validate_step_scale(scale: float, name: str = "step_scale") -> float:
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(f"invalid {name}: {scale}; expected {name} >= 0")
    return scale


def _active_step_scale(
    scheduled_scale: float, peak_scale: float, name: str = "schedule"
) -> float:
    scheduled_scale = validate_step_scale(scheduled_scale, f"{name} scheduled scale")
    peak_scale = validate_step_scale(peak_scale, f"{name} peak scale")
    if peak_scale == 0.0:
        if scheduled_scale != 0.0:
            raise ValueError(
                f"invalid {name} schedule: scheduled scale exceeds zero peak"
            )
        return 0.0
    if scheduled_scale > peak_scale * (1.0 + 1e-12):
        raise ValueError(
            f"invalid {name} scheduled scale {scheduled_scale}; "
            f"expected <= peak {peak_scale}"
        )
    return scheduled_scale


def scheduled_weight_retention(
    peak_weight_retention: float, step_scale: float, name: str = "weight_retention"
) -> float:
    if not (0.0 < peak_weight_retention <= 1.0):
        raise ValueError(
            f"invalid {name} peak weight retention: {peak_weight_retention}"
        )
    step_scale = validate_step_scale(step_scale, f"{name} step scale")
    return peak_weight_retention**step_scale


def ema_atom_correlation_factor(
    momentum_retention: float, weight_retention: float
) -> float:
    if not (0.0 <= momentum_retention <= 1.0):
        raise ValueError(f"invalid momentum-state retention: {momentum_retention}")
    if not (0.0 < weight_retention <= 1.0):
        raise ValueError(f"invalid weight retention: {weight_retention}")
    product = momentum_retention * weight_retention
    if product >= 1.0:
        return math.inf
    return (1.0 + product) / (1.0 - product)


def rms_matched_radius(
    rms_radius: float,
    momentum_retention: float,
    weight_retention: float,
    atom_sq: float = 1.0,
) -> tuple[float, float]:
    if rms_radius <= 0.0 or not math.isfinite(rms_radius):
        raise ValueError(f"invalid target RMS radius: {rms_radius}")
    if atom_sq <= 0.0 or not math.isfinite(atom_sq):
        raise ValueError(f"invalid atom squared-norm scale: {atom_sq}")
    atom_correlation = ema_atom_correlation_factor(
        momentum_retention, weight_retention
    )
    if weight_retention >= 1.0:
        return math.inf, atom_correlation
    denom = (1.0 - weight_retention) * atom_sq * atom_correlation
    return rms_radius * math.sqrt((1.0 + weight_retention) / denom), atom_correlation


def rms_matched_eta(
    rms_radius: float,
    momentum_retention: float,
    weight_retention: float,
    atom_sq: float = 1.0,
) -> float:
    if weight_retention >= 1.0:
        return 0.0
    if rms_radius <= 0.0 or not math.isfinite(rms_radius):
        raise ValueError(f"invalid target RMS radius: {rms_radius}")
    if atom_sq <= 0.0 or not math.isfinite(atom_sq):
        raise ValueError(f"invalid atom squared-norm scale: {atom_sq}")
    atom_correlation = ema_atom_correlation_factor(
        momentum_retention, weight_retention
    )
    return rms_radius * math.sqrt(
        ((1.0 - weight_retention) * (1.0 + weight_retention))
        / (atom_sq * atom_correlation)
    )


def weight_retention_for_rms_eta(
    rms_radius: float,
    momentum_retention: float,
    eta: float,
    atom_sq: float = 1.0,
) -> float:
    if eta <= 0.0:
        return 1.0
    if rms_radius <= 0.0 or not math.isfinite(rms_radius):
        raise ValueError(f"invalid target RMS radius: {rms_radius}")
    if atom_sq <= 0.0 or not math.isfinite(atom_sq):
        raise ValueError(f"invalid atom squared-norm scale: {atom_sq}")
    if eta >= rms_radius / math.sqrt(atom_sq):
        return 0.0
    if not (0.0 <= momentum_retention <= 1.0):
        raise ValueError(f"invalid momentum-state retention: {momentum_retention}")

    lo = 0.0
    hi = 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rms_matched_eta(rms_radius, momentum_retention, mid, atom_sq) > eta:
            lo = mid
        else:
            hi = mid
    return hi


def scheduled_rms_matched_action(
    rms_radius: float,
    momentum_retention: float,
    peak_weight_retention: float,
    peak_scale: float,
    scheduled_scale: float,
    atom_sq: float = 1.0,
    name: str = "group",
) -> RMSMatchedAction:
    step_scale = _active_step_scale(scheduled_scale, peak_scale, name)
    weight_retention = scheduled_weight_retention(
        peak_weight_retention, step_scale, name
    )
    rho, atom_correlation = rms_matched_radius(
        rms_radius,
        momentum_retention,
        weight_retention,
        atom_sq,
    )
    eta = (1.0 - weight_retention) * rho if math.isfinite(rho) else 0.0
    return RMSMatchedAction(
        rms_radius=rms_radius,
        rho=rho,
        weight_retention=weight_retention,
        eta=eta,
        step_scale=step_scale,
        atom_correlation=atom_correlation,
    )
