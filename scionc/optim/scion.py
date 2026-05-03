import torch
from torch.optim import Optimizer

__all__ = ["ScionC"]


class ScionC(Optimizer):
    """
    Minimal ScionC optimizer.

    Each group supplies a ULMO. The default constrained SCG update is:

        m <- beta * m + (1 - beta) * grad
        p <- zeta * p + (1 - zeta) * rho * ulmo(m)

    Internally, the additive scale `(1 - zeta) * rho` is stored in `lr` to
    match PyTorch optimizer conventions. The group field `weight_retention`
    stores `zeta`. `memory_beta` is the momentum-state retention. `readout_mu`
    optionally blends the current gradient and the updated momentum state before
    the ULMO.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        memory_beta: float = 0.95,
        readout_mu: float = 1.0,
        ulmo=None,
        weight_retention: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not (0.0 <= memory_beta < 1.0):
            raise ValueError(f"invalid memory_beta: {memory_beta}")
        if not (0.0 <= readout_mu <= 1.0):
            raise ValueError(f"invalid readout_mu: {readout_mu}")
        if not (0.0 < weight_retention <= 1.0):
            raise ValueError(f"invalid weight_retention: {weight_retention}")

        super().__init__(
            params,
            dict(
                lr=lr,
                memory_beta=memory_beta,
                readout_mu=readout_mu,
                ulmo=ulmo,
                weight_retention=weight_retention,
            ),
        )

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
            weight_retention = float(group["weight_retention"])
            if lr == 0.0:
                if weight_retention != 1.0:
                    for p, _ in entries:
                        p.mul_(weight_retention)
                continue

            updates = self._updates(group["ulmo"], entries)
            for (p, _), u in zip(entries, updates, strict=True):
                if weight_retention != 1.0:
                    p.mul_(weight_retention)
                p.add_(u, alpha=lr)

        return loss

    def _collect_entries(self, group) -> list[tuple[torch.Tensor, torch.Tensor]]:
        memory_beta = group["memory_beta"]
        readout_mu = group["readout_mu"]
        entries = []
        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            if g.is_sparse:
                raise RuntimeError("ScionC does not support sparse gradients")

            state = self.state[p]
            if not state:
                state["m"] = torch.zeros_like(
                    g, memory_format=torch.preserve_format
                )

            m = state["m"]
            m.lerp_(g, 1.0 - memory_beta)
            if readout_mu == 1.0:
                g.copy_(m)
            elif readout_mu != 0.0:
                g.lerp_(m, readout_mu)
            entries.append((p, g))
        return entries

    def _updates(
        self, ulmo, entries: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> list[torch.Tensor]:
        if ulmo is None:
            raise ValueError("ScionC requires a ULMO for every parameter group")

        batch_dir = getattr(ulmo, "batch", None)
        if batch_dir is not None:
            return batch_dir([g for _, g in entries], [p for p, _ in entries])

        set_ulmo_param = getattr(ulmo, "set_param", None)
        updates = []
        for p, g in entries:
            if set_ulmo_param is not None:
                set_ulmo_param(p)
            updates.append(ulmo(g))
        return updates
