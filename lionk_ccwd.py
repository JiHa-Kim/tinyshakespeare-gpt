import torch
from torch.optim import Optimizer

__all__ = [
    "sign_",
    "lionk_S",
    "corrected_eta",
    "consume_step_stats",
    "LionKCCWDPA",
]


def sign_(x: torch.Tensor) -> torch.Tensor:
    return x.sign_().neg_()


def lionk_S(beta1: float, beta2: float, nesterov: bool = True) -> float:
    beta_eff = beta1 * beta2 if nesterov else beta1
    return (1.0 + beta2) / (
        ((1.0 - beta_eff) ** 2) * (1.0 + beta2) + (beta_eff * beta_eff) * (1.0 - beta2)
    )


def corrected_eta(
    lr: float,
    theta2: float,
    cu2: float = 1.0,
    S: float = 1.0,
    q: float = 1.0,
    eps: float = 1e-12,
) -> float:
    return (lr * lr * cu2 * S) / (2.0 * max(q, eps) * max(theta2, eps))


def _stat_add(stats: dict, name: str, value: torch.Tensor) -> None:
    value = value.detach()
    current = stats.get(name)
    stats[name] = value if current is None else current + value


def _accumulate_step_stats(group: dict, entries, updates, lr: float) -> None:
    if not group.get("track_stats"):
        return
    stats = group.setdefault("_step_stats", {"steps": 0, "params": 0, "numel": 0})
    stats["steps"] += 1
    stats["params"] += len(entries)

    for (p, g, _, _, raw_g), u in zip(entries, updates, strict=True):
        stats["numel"] += p.numel()
        g32 = g.float()
        raw32 = g32 if raw_g is None else raw_g.float()
        u32 = u.float()
        p32 = p.float()
        update_sq = u32.square().sum()
        descent = -(g32 * u32).sum()
        raw_descent = -(raw32 * u32).sum()
        _stat_add(stats, "grad_sq", g32.square().sum())
        _stat_add(stats, "raw_grad_sq", raw32.square().sum())
        _stat_add(stats, "update_sq", update_sq)
        _stat_add(stats, "param_sq", p32.square().sum())
        _stat_add(stats, "descent", descent)
        _stat_add(stats, "lr_descent", descent * lr)
        _stat_add(stats, "raw_descent", raw_descent)
        _stat_add(stats, "lr_raw_descent", raw_descent * lr)
        _stat_add(stats, "lr2_update_sq", update_sq * (lr * lr))


def consume_step_stats(optimizer: Optimizer, eps: float = 1e-12) -> dict[str, dict]:
    out = {}
    for group in optimizer.param_groups:
        stats = group.pop("_step_stats", None)
        if not stats:
            continue
        grad_sq = float(stats.get("grad_sq", 0.0))
        raw_grad_sq = float(stats.get("raw_grad_sq", grad_sq))
        update_sq = float(stats.get("update_sq", 0.0))
        param_sq = float(stats.get("param_sq", 0.0))
        descent = float(stats.get("descent", 0.0))
        lr_descent = float(stats.get("lr_descent", 0.0))
        raw_descent = float(stats.get("raw_descent", descent))
        lr_raw_descent = float(stats.get("lr_raw_descent", lr_descent))
        lr2_update_sq = float(stats.get("lr2_update_sq", 0.0))
        numel = max(int(stats["numel"]), 1)
        out[group.get("name", "group")] = {
            "steps": int(stats["steps"]),
            "params": int(stats["params"]),
            "grad_rms": (grad_sq / numel) ** 0.5,
            "raw_grad_rms": (raw_grad_sq / numel) ** 0.5,
            "update_rms": (update_sq / numel) ** 0.5,
            "param_rms": (param_sq / numel) ** 0.5,
            "descent": descent,
            "lr_descent": lr_descent,
            "raw_descent": raw_descent,
            "lr_raw_descent": lr_raw_descent,
            "lr2_update_sq": lr2_update_sq,
            "cos": descent / ((grad_sq * update_sq) ** 0.5 + eps),
            "raw_cos": raw_descent / ((raw_grad_sq * update_sq) ** 0.5 + eps),
        }
    return out


class LionKCCWDPA(Optimizer):
    """
    Lion-K with corrected decoupled decay, optional cautious masking,
    and optional primal averaging.

    Parameters are kept at the gradient-eval point
        y_t = (1 - phi) z_t + phi x_t.

    `dir_fn` maps the momentum proxy to a negative update direction.
    `eta` applies fixed decoupled decay; otherwise it is derived from `theta2`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.99),
        dir_fn=sign_,
        phi: float = 0.0,
        eta: float | None = None,
        theta2: float | None = None,
        cu2: float = 1.0,
        S: float | None = None,
        q: float = 1.0,
        cwd: bool = False,
        nesterov: bool = True,
        eps: float = 1e-12,
    ):
        if lr <= 0.0:
            raise ValueError(f"invalid lr: {lr}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 <= 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"invalid betas: {betas}")
        if not (0.0 <= phi <= 1.0):
            raise ValueError(f"invalid phi: {phi}")

        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                dir_fn=dir_fn,
                phi=phi,
                eta=eta,
                theta2=theta2,
                cu2=cu2,
                S=S,
                q=q,
                cwd=cwd,
                nesterov=nesterov,
                eps=eps,
                _pa_denom=0.0,
                track_stats=False,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            dir_fn = group["dir_fn"]
            phi = group["phi"]
            cwd = group["cwd"]
            nesterov = group["nesterov"]
            set_dir_param = getattr(dir_fn, "set_param", None)
            batch_dir = getattr(dir_fn, "batch", None)

            if phi:
                group["_pa_denom"] += lr * lr
                c = (lr * lr) / group["_pa_denom"]
            else:
                c = 0.0

            eta = group["eta"]
            theta2 = group["theta2"]
            if eta is None:
                if theta2 is None:
                    eta = lr * group.get("weight_decay", 0.0)
                else:
                    S = group["S"]
                    if S is None:
                        S = group["S"] = lionk_S(beta1, beta2, nesterov=nesterov)
                    eta = corrected_eta(
                        lr=lr,
                        theta2=theta2,
                        cu2=group["cu2"],
                        S=S,
                        q=group["q"] if cwd else 1.0,
                        eps=group["eps"],
                    )

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
                raw_g = (
                    g.detach().clone(memory_format=torch.preserve_format)
                    if group.get("_line_probe")
                    else None
                )

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

                entries.append((p, g, z, state, raw_g))

            if batch_dir is None:
                updates = []
                for p, g, _, _, _ in entries:
                    if set_dir_param is not None:
                        set_dir_param(p)
                    updates.append(dir_fn(g))
            else:
                updates = batch_dir(
                    [g for _, g, _, _, _ in entries],
                    [p for p, _, _, _, _ in entries],
                )

            _accumulate_step_stats(group, entries, updates, lr)

            for (p, _, z, state, _), u in zip(entries, updates, strict=True):

                if eta:
                    if cwd:
                        z.addcmul_(p, (p * u > 0).to(dtype=p.dtype), value=-eta)
                    else:
                        z.add_(p, alpha=-eta)

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
