import math
import torch
import torch.nn as nn

from lionk_ccwd import LionKCCWDPA, lionk_S


__all__ = [
    'RMSLMO',
    'ColNormLMO',
    'RowNormLMO',
    'SpectralLMO',
    'init_colnorm_',
    'init_rownorm_',
    'init_semiorthogonal_',
    'scion_transfer_lr',
    'ScionC',
]


_PE = (
    (8.28721201814563 / 1.01, -23.595886519098837 / (1.01 ** 3), 17.300387312530933 / (1.01 ** 5)),
    (4.107059111542203 / 1.01, -2.9478499167379106 / (1.01 ** 3), 0.5448431082926601 / (1.01 ** 5)),
    (3.9486908534822946 / 1.01, -2.908902115962949 / (1.01 ** 3), 0.5518191394370137 / (1.01 ** 5)),
    (3.3184196573706015 / 1.01, -2.488488024314874 / (1.01 ** 3), 0.51004894012372 / (1.01 ** 5)),
    (2.300652019954817 / 1.01, -1.6689039845747493 / (1.01 ** 3), 0.4188073119525673 / (1.01 ** 5)),
    (1.891301407787398 / 1.01, -1.2679958271945868 / (1.01 ** 3), 0.37680408948524835 / (1.01 ** 5)),
    (1.8750014808534479 / 1.01, -1.2500016453999487 / (1.01 ** 3), 0.3750001645474248 / (1.01 ** 5)),
    (1.875, -1.25, 0.375),
)


def polar_express_uvt(
    g: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    work_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if g.ndim != 2:
        raise ValueError('polar_express_uvt expects a 2D tensor')

    if work_dtype is None:
        work_dtype = torch.bfloat16 if g.is_cuda else g.dtype

    x = g.to(work_dtype)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.mT

    x = x / (torch.linalg.matrix_norm(x, ord='fro') * 1.01 + eps)

    n = len(_PE)
    for i in range(steps):
        a, b, c = _PE[i if i < n else n - 1]
        A = x @ x.mT
        AX = A @ x
        AAX = A @ AX
        x.mul_(a).add_(AX, alpha=b).add_(AAX, alpha=c)

    if transposed:
        x = x.mT
    return x.to(g.dtype)


class RMSLMO:
    __slots__ = ('radius', 'eps')
    def __init__(self, radius: float = 1.0, eps: float = 1e-12):
        self.radius = radius
        self.eps = eps
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.div_(x.square().mean().add_(self.eps).sqrt_()).mul_(-self.radius)


class ColNormLMO:
    __slots__ = ('radius', 'eps')
    def __init__(self, radius: float = 50.0, eps: float = 1e-12):
        self.radius = radius
        self.eps = eps
    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        d_out = w.size(0)
        return w.div_(torch.linalg.vector_norm(w, dim=0, keepdim=True).clamp_min_(self.eps)).mul_(-self.radius * math.sqrt(d_out))


class RowNormLMO:
    __slots__ = ('radius', 'eps')
    def __init__(self, radius: float = 3000.0, eps: float = 1e-12):
        self.radius = radius
        self.eps = eps
    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        d_in = w.size(1)
        return w.div_(torch.linalg.vector_norm(w, dim=1, keepdim=True).clamp_min_(self.eps)).mul_(-self.radius / math.sqrt(d_in))


class SpectralLMO:
    __slots__ = ('radius', 'steps', 'eps', 'work_dtype', 'input_like')
    def __init__(self, radius: float = 50.0, steps: int = 5, eps: float = 1e-7, work_dtype: torch.dtype | None = None, input_like: bool = False):
        self.radius = radius
        self.steps = steps
        self.eps = eps
        self.work_dtype = work_dtype
        self.input_like = input_like
    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        d_out, d_in = v.shape
        scale = math.sqrt(d_out / d_in)
        if self.input_like:
            scale = max(1.0, scale)
        u = polar_express_uvt(v, steps=self.steps, eps=self.eps, work_dtype=self.work_dtype)
        return u.mul_(-self.radius * scale)


def init_colnorm_(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    nn.init.normal_(w, mean=0.0, std=1.0)
    return w.div_(torch.linalg.vector_norm(w, dim=0, keepdim=True).clamp_min_(eps))


def init_rownorm_(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    nn.init.normal_(w, mean=0.0, std=1.0)
    return w.div_(torch.linalg.vector_norm(w, dim=1, keepdim=True).clamp_min_(eps))


def init_semiorthogonal_(w: torch.Tensor) -> torch.Tensor:
    return nn.init.orthogonal_(w)


def scion_transfer_lr(base_lr: float, mT: float = 1.0, mL: float = 1.0, alpha: float = 0.5) -> dict[str, float]:
    return {
        'embed': base_lr * (mT ** -0.5),
        'hidden': base_lr * (mT ** -0.5) * (mL ** (alpha - 1.0)),
        'out': base_lr * (mT ** -0.5),
    }


class ScionC(LionKCCWDPA):
    """
    ScionC specialization.

    Important defaults for this repo:
      - primal averaging OFF by default: phi = 0.0
      - CWD OFF by default
      - tune a single global lr first; radii live in the LMOs
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta2: float = 0.95,
        dir_fn=None,
        phi: float = 0.0,
        eta: float | None = None,
        theta2: float | None = None,
        cu2: float = 1.0,
        S: float | None = None,
        q: float = 1.0,
        cwd: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=(1.0, beta2),
            dir_fn=dir_fn,
            phi=phi,
            eta=eta,
            theta2=theta2,
            cu2=cu2,
            S=lionk_S(1.0, beta2) if S is None else S,
            q=q,
            cwd=cwd,
            nesterov=True,
            eps=eps,
        )
