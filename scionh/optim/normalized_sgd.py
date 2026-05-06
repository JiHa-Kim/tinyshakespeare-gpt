import torch


class NormalizedSGD:
    """Scion-style normalized descent for small unconstrained parameter groups."""

    def __init__(self, params, lr: float, beta: float, eps: float = 1e-12):
        self.params = [p for p in params if p.requires_grad]
        self.lr_peak = float(lr)
        self.lr = float(lr)
        self.beta = float(beta)
        self.eps = eps
        self.state: dict[torch.Tensor, torch.Tensor] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        total = 0
        sq = None
        updates = []
        for p in self.params:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("NormalizedSGD does not support sparse gradients")
            m = self.state.get(p)
            if m is None:
                m = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p] = m
            m.lerp_(p.grad, 1.0 - self.beta)
            value = m.float().square().sum()
            sq = value if sq is None else sq + value
            total += p.numel()
            updates.append((p, m))

        if not updates or sq is None or total <= 0:
            return
        rms = (sq / total).sqrt()
        if float(rms) <= self.eps:
            return
        for p, m in updates:
            p.add_(m / rms.to(dtype=m.dtype), alpha=-self.lr)
