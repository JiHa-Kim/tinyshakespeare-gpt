import math

import torch
from torch.optim import Optimizer


class Hyperball(Optimizer):
    """
    Hyperball optimizer.

    Each controlled block lives on a fixed-radius RMS sphere.  The radius
    R = ‖W₀‖_rms is frozen on the first optimizer step.  The learning rate
    sets the Euclidean pre-retraction movement; state retention β sets momentum
    memory.
    The default update is the fixed-radius RMSNorm retraction:

        Ŵ_{t+1} = rmsnorm(Ŵ_t + lr rmsnorm(V_t))

    under this codebase's convention that ULMOs return descent directions.
    The `slerp` update projects the atom to the tangent space and applies the
    sphere exponential map:

        Ŵ_{t+1} = cos(lr) Ŵ_t + sin(lr) U_t

    where lr is the angular step in radians and U_t is the unit RMS tangent
    descent direction.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        beta: float = 0.95,
        ulmo=None,
        update_rule: str = "retract",
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"invalid beta: {beta}")
        if update_rule not in {"retract", "slerp"}:
            raise ValueError(f"invalid update_rule: {update_rule}")

        super().__init__(
            params, dict(lr=lr, beta=beta, ulmo=ulmo, update_rule=update_rule)
        )
        for group in self.param_groups:
            update = group.get("update_rule", update_rule)
            if update not in {"retract", "slerp"}:
                raise ValueError(f"invalid update_rule: {update}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            entries = self._collect_entries(group)
            if not entries:
                continue

            lr = float(group["lr"])
            if lr == 0.0:
                continue

            update_rule = group.get("update_rule", "retract")
            updates = self._updates(group["ulmo"], entries)

            for (_, p, _), u in zip(entries, updates, strict=True):
                state = self.state[p]
                R = state["R"]
                if R <= 0.0:
                    continue

                d = p.numel()
                w_hat = p.data / R

                if update_rule == "retract":
                    u_rms = (u.square().sum() / d).sqrt()
                    if float(u_rms) <= 0.0:
                        continue
                    w_hat = w_hat + lr * (u / u_rms)
                    w_hat_rms = (w_hat.square().sum() / d).sqrt()
                    w_hat = w_hat / w_hat_rms
                    p.data.copy_(R * w_hat)
                    continue

                inner_rms = (u * w_hat).sum() / d
                tangent = u - inner_rms * w_hat
                tangent_rms = (tangent.square().sum() / d).sqrt()
                if float(tangent_rms) <= 0.0:
                    continue

                tangent_unit = tangent / tangent_rms
                w_hat = math.cos(lr) * w_hat + math.sin(lr) * tangent_unit
                w_hat_rms = (w_hat.square().sum() / d).sqrt()
                w_hat = w_hat / w_hat_rms
                p.data.copy_(R * w_hat)

        return loss

    def _collect_entries(self, group) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        beta = float(group["beta"])
        entries = []
        for param_index, p in enumerate(group["params"]):
            g = p.grad
            if g is None:
                continue
            if g.is_sparse:
                raise RuntimeError("Hyperball does not support sparse gradients")

            state = self.state[p]
            if "m" not in state:
                state["m"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                # Freeze radius on first step: R = ‖W₀‖_rms
                d = p.numel()
                state["R"] = float((p.data.square().sum() / d).sqrt())

            m = state["m"]
            m.lerp_(g, 1.0 - beta)
            entries.append((param_index, p, m))
        return entries

    def _updates(
        self, ulmo, entries: list[tuple[int, torch.Tensor, torch.Tensor]]
    ) -> list[torch.Tensor]:
        if ulmo is None:
            # Euclidean direction: just return momentum (ULMO negation handled
            # by the projection step — the ULMO convention is descent direction)
            return [m.clone() for _, _, m in entries]

        batch_dir = getattr(ulmo, "batch", None)
        if batch_dir is not None:
            return batch_dir([m for _, _, m in entries], [p for _, p, _ in entries])

        set_ulmo_param = getattr(ulmo, "set_param", None)
        updates = []
        for _, p, m in entries:
            if set_ulmo_param is not None:
                set_ulmo_param(p)
            updates.append(ulmo(m))
        return updates
