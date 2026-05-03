import math

import torch
from torch.optim import Optimizer

__all__ = [
    "LionKCCWDPA",
    "lionk_effective_readout",
    "lionk_linear_correlation_factor",
    "lionk_sign_correlation_factor",
    "sign_",
]

_ASIN_SERIES_TERMS = 48
_SOLVE_ITERS = 56


def sign_(x: torch.Tensor) -> torch.Tensor:
    return x.sign_().neg_()


def lionk_effective_readout(
    beta1: float, beta2: float, nesterov: bool = True
) -> float:
    _validate_betas(beta1, beta2)
    return beta1 * beta2 if nesterov else beta1


def lionk_linear_correlation_factor(
    beta1: float, beta2: float, weight_retention: float, nesterov: bool = True
) -> float:
    _validate_weight_retention(weight_retention)
    _, lag1 = _linear_readout_stats(beta1, beta2, nesterov)
    denom = 1.0 - weight_retention * beta2
    if denom <= 0.0:
        return math.inf
    return 1.0 + 2.0 * weight_retention * lag1 / denom


def lionk_sign_correlation_factor(
    beta1: float, beta2: float, weight_retention: float, nesterov: bool = True
) -> float:
    _validate_weight_retention(weight_retention)
    _, lag1 = _linear_readout_stats(beta1, beta2, nesterov)
    lag1 = max(-1.0, min(1.0, lag1))
    if abs(lag1) < 1e-15 or weight_retention == 0.0:
        return 1.0

    # For jointly Gaussian readouts,
    # E[sign(Z_t) sign(Z_{t-k})] = 2/pi * asin(corr_k).
    # With corr_k = lag1 * beta2^(k-1), summing the asin power series gives
    # sum_k a^k asin(lag1 beta2^(k-1))
    # = sum_j c_j a lag1^(2j+1) / (1 - a beta2^(2j+1)).
    total = 0.0
    coeff = 1.0
    lag_power = lag1
    beta_power = beta2
    beta2_sq = beta2 * beta2
    lag1_sq = lag1 * lag1
    for j in range(_ASIN_SERIES_TERMS):
        denom = 1.0 - weight_retention * beta_power
        if denom <= 0.0:
            return math.inf
        term = coeff * weight_retention * lag_power / denom
        total += term
        if abs(term) <= 1e-14 * max(1.0, abs(total)):
            break
        coeff *= ((2 * j + 1) ** 2) / ((2 * j + 2) * (2 * j + 3))
        lag_power *= lag1_sq
        beta_power *= beta2_sq
    return 1.0 + (4.0 / math.pi) * total


def _validate_betas(beta1: float, beta2: float) -> None:
    if not (0.0 <= beta1 <= 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError(f"invalid betas: {(beta1, beta2)}")


def _validate_weight_retention(weight_retention: float) -> None:
    if not (0.0 <= weight_retention <= 1.0):
        raise ValueError(f"invalid weight_retention: {weight_retention}")


def _linear_readout_stats(
    beta1: float, beta2: float, nesterov: bool
) -> tuple[float, float]:
    _validate_betas(beta1, beta2)
    b = lionk_effective_readout(beta1, beta2, nesterov)
    nu0 = (1.0 - b) ** 2 + b * b * (1.0 - beta2) / (1.0 + beta2)
    if nu0 <= 0.0:
        return 0.0, 0.0
    nu1 = b * (1.0 - beta2) * (1.0 + beta2 - b) / (1.0 + beta2)
    return nu0, nu1 / nu0


def _correlation_factor(
    mode: str,
    dir_fn,
    beta1: float,
    beta2: float,
    weight_retention: float,
    nesterov: bool,
) -> float:
    if mode == "auto":
        mode = "sign" if dir_fn is sign_ else "linear"
    if mode == "none":
        return 1.0
    if mode == "linear":
        return lionk_linear_correlation_factor(
            beta1, beta2, weight_retention, nesterov=nesterov
        )
    if mode == "sign":
        return lionk_sign_correlation_factor(
            beta1, beta2, weight_retention, nesterov=nesterov
        )
    raise ValueError(f"invalid correlation mode: {mode}")


def _stationary_decay_complement(
    additive_scale: float,
    rms_radius: float,
    direction_sq: float,
    beta1: float,
    beta2: float,
    nesterov: bool,
    correlation: str,
    dir_fn,
    mask_sq_fraction: float,
) -> tuple[float, float, float]:
    if additive_scale <= 0.0:
        return 0.0, 1.0, 1.0
    if rms_radius <= 0.0 or not math.isfinite(rms_radius):
        raise ValueError(f"invalid rms_radius: {rms_radius}")
    if direction_sq <= 0.0 or not math.isfinite(direction_sq):
        raise ValueError(f"invalid direction_sq: {direction_sq}")
    if not (0.0 < mask_sq_fraction <= 1.0):
        raise ValueError(f"invalid mask_sq_fraction: {mask_sq_fraction}")

    scale_sq = additive_scale * additive_scale * direction_sq / (rms_radius**2)

    def residual(decay: float) -> float:
        masked_retention = 1.0 - mask_sq_fraction * decay
        corr = _correlation_factor(
            correlation, dir_fn, beta1, beta2, masked_retention, nesterov
        )
        return mask_sq_fraction * decay * (2.0 - decay) - scale_sq * corr

    if residual(1.0) <= 0.0:
        corr = _correlation_factor(
            correlation,
            dir_fn,
            beta1,
            beta2,
            1.0 - mask_sq_fraction,
            nesterov,
        )
        return 1.0, 0.0, corr

    lo = 0.0
    hi = 1.0
    for _ in range(_SOLVE_ITERS):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    decay = hi
    retention = 1.0 - decay
    corr_retention = 1.0 - mask_sq_fraction * decay
    corr = _correlation_factor(
        correlation, dir_fn, beta1, beta2, corr_retention, nesterov
    )
    return decay, retention, corr


class LionKCCWDPA(Optimizer):
    """
    Lion-K with retention/radius weight control, optional cautious masking,
    and optional parameter averaging.

    The PyTorch `lr` field is the additive direction scale gamma:
        z <- z + gamma * U.

    Weight control is expressed by an active-coordinate retention zeta:
        z <- z - (1 - zeta) * z + gamma * U
    for unmasked decay, and
        z <- z - (1 - zeta) * P*z + gamma * U
    for cautious decay. If `rms_radius` is supplied, zeta is derived from the
    stationary RMS balance. If `weight_retention` is supplied, that fixed zeta
    is used directly.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.99),
        dir_fn=sign_,
        phi: float = 0.0,
        weight_retention: float | None = None,
        rms_radius: float | None = None,
        direction_sq: float = 1.0,
        mask_sq_fraction: float = 1.0,
        correlation: str = "auto",
        cwd: bool = False,
        nesterov: bool = True,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"invalid lr: {lr}")
        beta1, beta2 = betas
        _validate_betas(beta1, beta2)
        if not (0.0 <= phi <= 1.0):
            raise ValueError(f"invalid phi: {phi}")
        if weight_retention is not None:
            _validate_weight_retention(weight_retention)
        if rms_radius is not None and (
            rms_radius <= 0.0 or not math.isfinite(rms_radius)
        ):
            raise ValueError(f"invalid rms_radius: {rms_radius}")
        if direction_sq <= 0.0 or not math.isfinite(direction_sq):
            raise ValueError(f"invalid direction_sq: {direction_sq}")
        if not (0.0 < mask_sq_fraction <= 1.0):
            raise ValueError(f"invalid mask_sq_fraction: {mask_sq_fraction}")
        if correlation not in {"auto", "none", "linear", "sign"}:
            raise ValueError(f"invalid correlation mode: {correlation}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")

        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                dir_fn=dir_fn,
                phi=phi,
                fixed_weight_retention=weight_retention,
                weight_retention=weight_retention,
                rms_radius=rms_radius,
                direction_sq=direction_sq,
                mask_sq_fraction=mask_sq_fraction,
                correlation=correlation,
                atom_correlation=math.nan,
                cwd=cwd,
                nesterov=nesterov,
                eps=eps,
                weight_decay=weight_decay,
                _pa_denom=0.0,
            ),
        )
        for group in self.param_groups:
            self._decay_complement(group)

    def _decay_complement(self, group: dict) -> float:
        fixed_retention = group.get("fixed_weight_retention")
        if fixed_retention is not None:
            _validate_weight_retention(float(fixed_retention))
            retention = float(fixed_retention)
            decay = 1.0 - retention
            corr_retention = (
                1.0 - float(group["mask_sq_fraction"]) * decay
                if group.get("cwd", False)
                else retention
            )
            beta1, beta2 = group["betas"]
            group["weight_retention"] = retention
            group["atom_correlation"] = _correlation_factor(
                str(group["correlation"]),
                group["dir_fn"],
                beta1,
                beta2,
                corr_retention,
                bool(group["nesterov"]),
            )
            return decay

        lr = float(group["lr"])
        rms_radius = group.get("rms_radius")
        if rms_radius is None:
            decay = lr * float(group.get("weight_decay", 0.0))
            if decay > 1.0:
                raise ValueError(f"weight decay complement exceeds one: {decay}")
            group["weight_retention"] = 1.0 - decay
            group["atom_correlation"] = 1.0
            return decay

        beta1, beta2 = group["betas"]
        decay, retention, corr = _stationary_decay_complement(
            additive_scale=lr,
            rms_radius=float(rms_radius),
            direction_sq=float(group["direction_sq"]),
            beta1=beta1,
            beta2=beta2,
            nesterov=bool(group["nesterov"]),
            correlation=str(group["correlation"]),
            dir_fn=group["dir_fn"],
            mask_sq_fraction=float(group["mask_sq_fraction"])
            if group.get("cwd", False)
            else 1.0,
        )
        group["weight_retention"] = retention
        group["atom_correlation"] = corr
        return decay

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            dir_fn = group["dir_fn"]
            phi = float(group["phi"])
            cwd = bool(group["cwd"])
            nesterov = bool(group["nesterov"])
            set_dir_param = getattr(dir_fn, "set_param", None)
            batch_dir = getattr(dir_fn, "batch", None)
            decay = self._decay_complement(group)

            if phi:
                group["_pa_denom"] += lr * lr
                c = (lr * lr) / group["_pa_denom"]
            else:
                c = 0.0

            entries = []
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("LionKCCWDPA does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["m"] = g.detach().clone(memory_format=torch.preserve_format)
                    state["z"] = p.detach().clone(memory_format=torch.preserve_format)
                    if phi:
                        state["x"] = p.detach().clone(
                            memory_format=torch.preserve_format
                        )

                m = state["m"]
                z = state["z"]

                # Reuse grad storage as scratch to avoid per-step allocations.
                if nesterov:
                    m.lerp_(g, 1.0 - beta2)
                    if beta1 == 1.0:
                        g.copy_(m)
                    elif beta1 != 0.0:
                        g.lerp_(m, beta1)
                else:
                    if beta1 != 0.0:
                        tmp = state.get("tmp")
                        if tmp is None:
                            tmp = state["tmp"] = m.detach().clone(
                                memory_format=torch.preserve_format
                            )
                        else:
                            tmp.copy_(m)
                    m.lerp_(g, 1.0 - beta2)
                    if beta1 == 1.0:
                        g.copy_(tmp)
                    elif beta1 != 0.0:
                        g.mul_(1.0 - beta1).add_(tmp, alpha=beta1)

                entries.append((p, g, z, state))

            if batch_dir is None:
                updates = []
                for p, g, _, _ in entries:
                    if set_dir_param is not None:
                        set_dir_param(p)
                    updates.append(dir_fn(g))
            else:
                updates = batch_dir(
                    [g for _, g, _, _ in entries],
                    [p for p, _, _, _ in entries],
                )

            for (p, _, z, state), u in zip(entries, updates, strict=True):
                if decay:
                    if cwd:
                        mask = (p * u > 0).to(dtype=p.dtype)
                        z.addcmul_(p, mask, value=-decay)
                    else:
                        z.add_(p, alpha=-decay)

                z.add_(u, alpha=lr)

                if phi:
                    x = state["x"]
                    x.lerp_(z, c)
                    if phi == 1.0:
                        p.copy_(x)
                    else:
                        p.copy_(z).lerp_(x, phi)
                else:
                    p.copy_(z)

        return loss
