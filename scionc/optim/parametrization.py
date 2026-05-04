import math


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
