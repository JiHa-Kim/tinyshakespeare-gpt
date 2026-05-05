import math


def retention_from_half_life(
    delta_tau: float, half_life: float, name: str = "half_life"
) -> float:
    """Compute a per-update EMA retention from a processed-token half-life."""
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
    """WSD schedule scalar at a given step."""
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


def scheduled_learning_rate(
    lr_peak: float,
    schedule_ratio: float,
) -> float:
    """Apply a WSD schedule ratio to the Euclidean pre-retraction step size."""
    if lr_peak < 0.0:
        raise ValueError(f"invalid peak learning rate: {lr_peak}")
    if schedule_ratio <= 0.0:
        return 0.0
    if schedule_ratio >= 1.0:
        return lr_peak
    return schedule_ratio * lr_peak
