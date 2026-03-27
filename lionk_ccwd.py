import torch
from torch.optim import Optimizer


__all__ = ['sign_', 'lionk_S', 'corrected_eta', 'LionKCCWDPA']


def sign_(x: torch.Tensor) -> torch.Tensor:
    return x.sign_().neg_()


def lionk_S(beta1: float, beta2: float, nesterov: bool = True) -> float:
    beta_eff = beta1 * beta2 if nesterov else beta1
    return (1.0 + beta2) / (
        ((1.0 - beta_eff) ** 2) * (1.0 + beta2) +
        (beta_eff * beta_eff) * (1.0 - beta2)
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


class LionKCCWDPA(Optimizer):
    """
    General Lion-K with corrected decoupled decay, optional cautious masking,
    and optional primal averaging.

    Live params are always the gradient-eval point
        y_t = (1 - phi) z_t + phi x_t.

    Each param group may provide:
        dir_fn: maps v_t -> negative direction u_t
        eta: fixed multiplicative decay per step
    or enough info to derive eta from corrected_eta(...).
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
            raise ValueError(f'invalid lr: {lr}')
        beta1, beta2 = betas
        if not (0.0 <= beta1 <= 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f'invalid betas: {betas}')
        if not (0.0 <= phi <= 1.0):
            raise ValueError(f'invalid phi: {phi}')

        defaults = dict(
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
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            dir_fn = group['dir_fn']
            phi = group['phi']
            cwd = group['cwd']

            if phi:
                group['_pa_denom'] += lr * lr
                c = (lr * lr) / group['_pa_denom']
            else:
                c = 0.0

            eta = group['eta']
            if eta is None:
                theta2 = group['theta2']
                if theta2 is None:
                    eta = 0.0
                else:
                    S = group['S']
                    if S is None:
                        S = lionk_S(beta1, beta2, nesterov=group['nesterov'])
                    eta = corrected_eta(
                        lr=lr,
                        theta2=theta2,
                        cu2=group['cu2'],
                        S=S,
                        q=group['q'] if cwd else 1.0,
                        eps=group['eps'],
                    )

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError('LionKCCWDPA does not support sparse gradients')

                state = self.state[p]
                if not state:
                    state['m'] = g.detach().clone(memory_format=torch.preserve_format)
                    state['z'] = p.detach().clone(memory_format=torch.preserve_format)
                    if phi:
                        state['x'] = p.detach().clone(memory_format=torch.preserve_format)

                m = state['m']
                z = state['z']

                m.lerp_(g, 1.0 - beta2)

                if beta1 == 1.0:
                    v = g.copy_(m)
                elif beta1 == 0.0:
                    v = g
                else:
                    v = g.mul_(1.0 - beta1).add_(m, alpha=beta1)

                u = dir_fn(v)

                if eta:
                    if cwd:
                        mask = (p * u) > 0
                        z.addcmul_(p, mask.to(dtype=p.dtype), value=-eta)
                    else:
                        z.add_(p, alpha=-eta)

                z.add_(u, alpha=lr)

                if phi:
                    x = state['x']
                    x.lerp_(z, c)
                    if phi == 1.0:
                        p.copy_(x)
                    else:
                        p.copy_(z).lerp_(x, phi)
                else:
                    p.copy_(z)

        return loss

    @torch.no_grad()
    def copy_fast_to_params_(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p)
                if state:
                    p.copy_(state['z'])

    @torch.no_grad()
    def copy_average_to_params_(self):
        for group in self.param_groups:
            if not group['phi']:
                continue
            for p in group['params']:
                state = self.state.get(p)
                if state:
                    p.copy_(state['x'])
