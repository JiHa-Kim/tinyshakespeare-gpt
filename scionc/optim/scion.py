import math

import torch
from torch.optim import Optimizer


class RMSSphere(Optimizer):
    """
    RMS-Sphere optimizer.

    Each controlled block lives on a fixed-radius RMS sphere.  The radius
    R = ‖W₀‖_rms is frozen on the first optimizer step.  A single direction
    retention q sets the angular movement; momentum retention β = q.

    The spherical update is:

        Ŵ_{t+1} = q Ŵ_t + √(1 − q²) U_t

    where U_t is the unit RMS tangent descent direction obtained by projecting
    the ULMO atom of the momentum state onto the tangent space of the sphere.
    """

    def __init__(self, params, q: float = 0.99, ulmo=None):
        if not (0.0 < q <= 1.0):
            raise ValueError(f"invalid q: {q}")

        super().__init__(params, dict(q=q, ulmo=ulmo))

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

            q = float(group["q"])
            eps_move = math.sqrt(max(0.0, 1.0 - q * q))
            if eps_move == 0.0:
                continue

            updates = self._updates(group["ulmo"], entries)

            for (_, p, _), u in zip(entries, updates, strict=True):
                state = self.state[p]
                R = state["R"]
                if R <= 0.0:
                    continue

                d = p.numel()
                w_hat = p.data / R

                # Tangent projection: D = u − ⟨u, ŵ⟩_rms ŵ
                inner_rms = (u * w_hat).sum() / d
                D = u - inner_rms * w_hat
                D_rms = (D.square().sum() / d).sqrt()

                if float(D_rms) > 0.0:
                    U = D / D_rms
                    w_hat = q * w_hat + eps_move * U
                    # Numerical re-projection to the constraint manifold
                    w_hat_rms = (w_hat.square().sum() / d).sqrt()
                    w_hat = w_hat / w_hat_rms
                    p.data.copy_(R * w_hat)

        return loss

    def _collect_entries(
        self, group
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        q = float(group["q"])  # β = q (tied)
        entries = []
        for param_index, p in enumerate(group["params"]):
            g = p.grad
            if g is None:
                continue
            if g.is_sparse:
                raise RuntimeError("RMSSphere does not support sparse gradients")

            state = self.state[p]
            if "m" not in state:
                state["m"] = torch.zeros_like(
                    g, memory_format=torch.preserve_format
                )
                # Freeze radius on first step: R = ‖W₀‖_rms
                d = p.numel()
                state["R"] = float((p.data.square().sum() / d).sqrt())

            m = state["m"]
            m.lerp_(g, 1.0 - q)
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
