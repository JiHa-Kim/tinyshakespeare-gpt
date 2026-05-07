import torch

from scionh.ulmos.core import (
    ULMOGeometry,
    _capped_response_weights,
    _power_alignment,
    _power_effective_rank,
    _power_response_weights,
    _singular_alignment,
    _singular_effective_rank,
    _singular_stable_rank,
)

_SVDGroupKey = tuple[tuple[int, ...], torch.dtype, torch.device]
_SVDItem = tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, bool, float]


def _normalize_columns(
    x: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_scale = (
        torch.linalg.vector_norm(x, dim=-2, keepdim=True).clamp_min(eps).reciprocal()
    )
    return x * inv_scale, inv_scale


def _batch_alignment(
    sigma: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    denom = torch.linalg.vector_norm(sigma, dim=-1) * torch.linalg.vector_norm(
        weights, dim=-1
    )
    tiny = torch.finfo(denom.dtype).tiny
    return (sigma * weights).sum(dim=-1) / denom.clamp_min(tiny)


def _batch_effective_rank(weights: torch.Tensor, eps: float) -> torch.Tensor:
    mass = weights.square()
    total = mass.sum(dim=-1, keepdim=True)
    probs = (mass / total.clamp_min(eps)).clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy.exp()


def _batch_stable_rank(weights: torch.Tensor, eps: float) -> torch.Tensor:
    mass = weights.square()
    total = mass.sum(dim=-1)
    peak = mass.amax(dim=-1)
    return total / peak.clamp_min(eps)


def _capped_response_weights_batch(
    sigma: torch.Tensor,
    target_rank: int,
    eps: float,
    search_steps: int,
) -> torch.Tensor:
    if sigma.ndim == 1:
        return _capped_response_weights(sigma, target_rank, eps, search_steps)

    n = sigma.size(-1)
    rank = max(1, min(int(target_rank), n))
    tiny = torch.finfo(sigma.dtype).tiny
    sigma = sigma.clamp_min(0.0)
    positive = sigma > 0.0
    active_count = positive.sum(dim=-1)
    active = active_count > 0

    if rank <= 1:
        norm = torch.linalg.vector_norm(sigma, dim=-1, keepdim=True)
        weights = sigma / norm.clamp_min(tiny)
        return torch.where(active.unsqueeze(-1), weights, torch.zeros_like(weights))

    cap = sigma.new_tensor(1.0 / (rank ** 0.5))

    # Exact batched water-filling solve for ||min(lambda * sigma, cap)||_2 = 1.
    # If the requested stable rank exceeds the nonzero support, the best feasible
    # response is uniform over that support.
    sorted_sigma, order = sigma.sort(dim=-1, descending=True)
    sorted_positive = sorted_sigma > 0.0
    uniform = sorted_positive.to(dtype=sigma.dtype) / active_count.to(
        dtype=sigma.dtype
    ).clamp_min(1).sqrt().unsqueeze(-1)

    sq = sorted_sigma.square()
    prefix = sq.cumsum(dim=-1)
    prefix = torch.cat([sq.new_zeros((*sq.shape[:-1], 1)), prefix], dim=-1)
    total = prefix[..., -1:]
    candidates = torch.arange(rank, device=sigma.device)
    remaining = (1.0 - candidates.to(dtype=sigma.dtype) / rank).clamp_min(tiny)
    tail = total - prefix.index_select(-1, candidates)
    lam = (remaining / tail.clamp_min(tiny)).sqrt()

    next_sigma = sorted_sigma.index_select(-1, candidates)
    prev_sigma = torch.cat(
        [sorted_sigma.new_full((*sorted_sigma.shape[:-1], 1), float("inf")),
         sorted_sigma[..., : rank - 1]],
        dim=-1,
    )
    valid = (lam * next_sigma <= cap) & (lam * prev_sigma >= cap)
    choice = valid.to(dtype=torch.int64).argmax(dim=-1, keepdim=True)
    lam_star = lam.gather(-1, choice)
    capped = torch.minimum(sorted_sigma * lam_star, cap)
    capped = capped / torch.linalg.vector_norm(capped, dim=-1, keepdim=True).clamp_min(
        tiny
    )
    sorted_weights = torch.where(
        (active_count <= rank).unsqueeze(-1),
        uniform,
        capped,
    )
    weights = torch.zeros_like(sorted_weights).scatter(-1, order, sorted_weights)
    return torch.where(active.unsqueeze(-1), weights, torch.zeros_like(weights))


class StreamingSVDULMO:
    """
    Spectral ULMO based on one or more streaming power-iteration steps.

    The cached V basis is stored per parameter via `set_param`, which is called
    by optimizers when the ULMO exposes that hook.
    """

    __slots__ = (
        "steps",
        "ridge",
        "refresh_interval",
        "refresh_threshold",
        "iteration",
        "eps",
        "work_dtype",
        "geometry",
        "states",
        "_param_key",
        "stats",
    )

    def __init__(
        self,
        steps: int = 1,
        ridge: float = 1e-3,
        refresh_interval: int = 25,
        refresh_threshold: float = 0.10,
        iteration: str = "scqr2",
        eps: float = 1e-12,
        work_dtype: torch.dtype | None = torch.float32,
        input_like: bool = False,
    ):
        if steps <= 0:
            raise ValueError(f"invalid steps: {steps}")
        if ridge < 0.0:
            raise ValueError(f"invalid ridge: {ridge}")
        if refresh_interval < 0:
            raise ValueError(f"invalid refresh_interval: {refresh_interval}")
        if refresh_threshold < 0.0:
            raise ValueError(f"invalid refresh_threshold: {refresh_threshold}")
        if iteration not in {"scqr2", "norm-power"}:
            raise ValueError(f"invalid iteration: {iteration}")

        self.steps = steps
        self.ridge = ridge
        self.refresh_interval = refresh_interval
        self.refresh_threshold = refresh_threshold
        self.iteration = iteration
        self.eps = eps
        self.work_dtype = work_dtype
        self.geometry = ULMOGeometry("spectral", input_like=input_like)
        self.states = {}
        self._param_key = None
        self.stats = {"calls": 0, "steps": 0, "refreshes": 0, "quality_checks": 0}

    def set_param(self, p: torch.Tensor) -> None:
        self._param_key = id(p)

    def _state_key(self, x: torch.Tensor) -> tuple:
        base = self._param_key
        if base is None:
            base = ("shape", tuple(x.shape))
        return (base, tuple(x.shape), x.dtype, x.device)

    def _resolve_work_dtype(self, x: torch.Tensor) -> torch.dtype:
        if self.work_dtype is not None:
            return self.work_dtype
        if x.dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return x.dtype

    def _ridge_scale(self, gram: torch.Tensor) -> torch.Tensor:
        scale = gram[..., 0, 0].abs()
        return torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(
            self.eps
        )

    def _cholesky_fast(self, gram: torch.Tensor, ridge: float) -> torch.Tensor:
        if gram.dtype not in {torch.float32, torch.float64}:
            gram = gram.float()
        scale = self._ridge_scale(gram)
        shifted = (gram + gram.mT).mul(0.5)
        diag = shifted.diagonal(dim1=-2, dim2=-1)
        shift = ridge * scale
        if shift.ndim:
            shift = shift.unsqueeze(-1)
        diag.add_(shift)
        r, _ = torch.linalg.cholesky_ex(shifted, upper=True, check_errors=False)
        return r

    def _solve_right(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if x.dtype != r.dtype:
            x = x.to(r.dtype)
        return torch.linalg.solve_triangular(r, x, upper=True, left=False)

    def _scqr_once_fast(self, x: torch.Tensor, ridge: float) -> torch.Tensor:
        r = self._cholesky_fast(x.mT @ x, ridge)
        return self._solve_right(x, r)

    def _qr(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled, _ = _normalize_columns(x, self.eps)
        return self._scqr_once_fast(x_scaled, self.ridge)

    def _v_step_scqr2(
        self, m: torch.Tensor, v: torch.Tensor, check_refresh: bool = False
    ) -> torch.Tensor:
        mv = m @ v
        mv_scaled, inv_scale = _normalize_columns(mv, self.eps)

        # Direct Gram from M @ V preserves positive semidefiniteness better than
        # the algebraically equivalent V.T @ (M.T @ M @ V) in finite precision.
        a_scaled = (m.mT @ mv) * inv_scale
        gram1 = mv_scaled.mT @ mv_scaled
        exact_qr = False
        if check_refresh:
            if self.refresh_threshold == 0.0:
                exact_qr = True
            else:
                quality = gram1.detach().clone()
                quality.diagonal(dim1=-2, dim2=-1).zero_()
                n = quality.size(-1)
                rms = torch.sqrt(
                    quality.square().sum(dim=(-2, -1)) / max(n * (n - 1), 1)
                )
                self.stats["quality_checks"] += int(rms.numel())
                exact_qr = bool((rms > self.refresh_threshold).any())
        r1 = self._cholesky_fast(gram1, self.ridge)
        b = self._solve_right(a_scaled, r1)
        if exact_qr:
            self.stats["refreshes"] += b.shape[0] if b.ndim == 3 else 1
            return torch.linalg.qr(b.float(), mode="reduced").Q.to(dtype=v.dtype)
        return self._qr(b.to(dtype=v.dtype)).to(dtype=v.dtype)

    def _v_step_norm_power(
        self, m: torch.Tensor, v: torch.Tensor, check_refresh: bool = False
    ) -> torch.Tensor:
        mv = m @ v
        mv_scaled, inv_scale = _normalize_columns(mv, self.eps)
        b = (m.mT @ mv) * inv_scale

        exact_qr = False
        if check_refresh:
            if self.refresh_threshold == 0.0:
                exact_qr = True
            else:
                quality = mv_scaled.mT @ mv_scaled
                n = quality.size(-1)
                offdiag_sq = quality.square().sum(dim=(-2, -1))
                offdiag_sq = offdiag_sq - quality.diagonal(
                    dim1=-2, dim2=-1
                ).square().sum(dim=-1)
                rms = torch.sqrt(offdiag_sq.clamp_min_(0.0) / max(n * (n - 1), 1))
                self.stats["quality_checks"] += int(rms.numel())
                exact_qr = bool((rms > self.refresh_threshold).any())

        if exact_qr:
            self.stats["refreshes"] += b.shape[0] if b.ndim == 3 else 1
            return torch.linalg.qr(b.float(), mode="reduced").Q.to(dtype=v.dtype)
        return self._qr(b.to(dtype=v.dtype)).to(dtype=v.dtype)

    def _v_step(
        self, m: torch.Tensor, v: torch.Tensor, check_refresh: bool = False
    ) -> torch.Tensor:
        if self.iteration == "norm-power":
            return self._v_step_norm_power(m, v, check_refresh)
        return self._v_step_scqr2(m, v, check_refresh)

    def _basis_for(self, p: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        key = (id(p), tuple(m.shape), m.dtype, m.device)
        v = self.states.get(key)
        if v is None or v.shape != (m.size(-1), m.size(-1)):
            v = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
        return v

    def _store_basis_for(
        self, p: torch.Tensor, m: torch.Tensor, v: torch.Tensor
    ) -> None:
        key = (id(p), tuple(m.shape), m.dtype, m.device)
        self.states[key] = v.detach()

    def final_stats(self) -> dict:
        return dict(self.stats)

    def batch(
        self, tensors: list[torch.Tensor], params: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor | None] = [None] * len(tensors)
        groups: dict[_SVDGroupKey, list[_SVDItem]] = {}
        self.stats["steps"] += 1
        check_refresh = (
            self.refresh_interval > 0
            and self.stats["steps"] % self.refresh_interval == 0
        )

        for i, (x, p) in enumerate(zip(tensors, params, strict=True)):
            if x.ndim != 2:
                out[i] = self(x)
                continue
            work_dtype = self._resolve_work_dtype(x)
            m = x.to(work_dtype)
            transposed = m.size(0) < m.size(1)
            if transposed:
                m = m.mT
            key = (tuple(m.shape), m.dtype, m.device)
            groups.setdefault(key, []).append(
                (i, x, p, m, transposed, self.geometry.scale(x))
            )

        for items in groups.values():
            m_batch = torch.stack([item[3] for item in items])
            v_batch = torch.stack([self._basis_for(item[2], item[3]) for item in items])

            for _ in range(self.steps):
                v_batch = self._v_step(m_batch, v_batch, check_refresh)

            mv = m_batch @ v_batch
            raw_sigma = torch.linalg.vector_norm(mv, dim=-2)
            sigma = raw_sigma.clamp_min(self.eps)
            y_batch = self._response(mv, sigma, v_batch, raw_sigma)

            for j, (i, x, p, m, transposed, scale) in enumerate(items):
                v = v_batch[j]
                self._store_basis_for(p, m, v)
                y = y_batch[j].mT if transposed else y_batch[j]
                out[i] = y.to(dtype=x.dtype).mul_(-scale)
                self.stats["calls"] += 1

        if any(x is None for x in out):
            raise RuntimeError("batched StreamingSVDULMO missed an output")
        return out

    def _response(
        self,
        mv: torch.Tensor,
        sigma: torch.Tensor,
        v: torch.Tensor,
        raw_sigma: torch.Tensor,
    ) -> torch.Tensor:
        del raw_sigma
        return (mv * sigma.reciprocal().unsqueeze(-2)) @ v.mT

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("StreamingSVDULMO expects a 2D tensor")

        work_dtype = self._resolve_work_dtype(x)
        m = x.to(work_dtype)
        transposed = m.size(0) < m.size(1)
        if transposed:
            m = m.mT
        self.stats["steps"] += 1
        check_refresh = (
            self.refresh_interval > 0
            and self.stats["steps"] % self.refresh_interval == 0
        )

        key = self._state_key(m)
        v = self.states.get(key)
        if v is None or v.shape != (m.size(1), m.size(1)):
            v = torch.eye(m.size(1), dtype=m.dtype, device=m.device)

        for _ in range(self.steps):
            v = self._v_step(m, v, check_refresh)
        self.states[key] = v.detach()

        mv = m @ v
        raw_sigma = torch.linalg.vector_norm(mv, dim=0)
        sigma = raw_sigma.clamp_min(self.eps)
        out = self._response(mv, sigma, v, raw_sigma)
        if transposed:
            out = out.mT
        self.stats["calls"] += 1
        return out.to(dtype=x.dtype).mul_(-self.geometry.scale(x))


class StreamingSpectralShapeULMO(StreamingSVDULMO):
    __slots__ = (
        "mode",
        "alpha",
        "target_alignment",
        "target_effective_rank",
        "target_stable_rank",
        "rank_frac",
        "search_steps",
        "_shape_stats",
    )

    def __init__(
        self,
        mode: str = "power",
        alpha: float = 0.5,
        target_alignment: float = 0.95,
        target_effective_rank: float = 0.0,
        target_stable_rank: float = 0.0,
        rank_frac: float = 0.5,
        search_steps: int = 24,
        **kwargs,
    ):
        if mode not in {"power", "power-rho", "power-erank", "cap-rho", "cap-rank"}:
            raise ValueError(f"invalid streaming spectral shape mode: {mode}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"invalid power alpha: {alpha}")
        if not (0.0 < target_alignment <= 1.0):
            raise ValueError(f"invalid target alignment: {target_alignment}")
        if target_effective_rank < 0.0:
            raise ValueError(f"invalid target effective rank: {target_effective_rank}")
        if target_stable_rank < 0.0:
            raise ValueError(f"invalid target stable rank: {target_stable_rank}")
        if rank_frac < 0.0:
            raise ValueError(f"invalid rank fraction: {rank_frac}")
        if search_steps <= 0:
            raise ValueError(f"invalid search steps: {search_steps}")
        super().__init__(**kwargs)
        self.mode = mode
        self.alpha = alpha
        self.target_alignment = target_alignment
        self.target_effective_rank = target_effective_rank
        self.target_stable_rank = target_stable_rank
        self.rank_frac = rank_frac
        self.search_steps = search_steps
        self.stats.update(
            {
                "active_calls": 0,
                "effective_rank_sum": 0.0,
                "stable_rank_sum": 0.0,
                "alignment_sum": 0.0,
            }
        )
        if mode.startswith("power"):
            self.stats["alpha_sum"] = 0.0
        else:
            self.stats["rank_sum"] = 0.0
        self._shape_stats = {}

    def _stat_add(self, name: str, value: torch.Tensor) -> None:
        value = value.detach()
        current = self._shape_stats.get(name)
        if current is None:
            self._shape_stats[name] = value.clone()
            return
        if current.device != value.device:
            value = value.to(current.device)
        current.add_(value)

    def final_stats(self) -> dict:
        out = dict(self.stats)
        for key, value in self._shape_stats.items():
            x = float(value.detach().cpu())
            out[key] = int(round(x)) if key == "active_calls" else x
        return out

    def _target_erank(self, sigma: torch.Tensor) -> float:
        if self.target_effective_rank > 0.0:
            target = self.target_effective_rank
        else:
            target = self.rank_frac * sigma.numel()
        return max(1.0, min(float(target), float(sigma.numel())))

    def _target_rank(self, sigma: torch.Tensor) -> int:
        if self.target_stable_rank > 0.0:
            target = self.target_stable_rank
        else:
            target = self.rank_frac * sigma.numel()
        return max(1, min(int(round(target)), sigma.numel()))

    def _alpha_for_alignment(self, sigma: torch.Tensor) -> float:
        if _power_alignment(sigma, 0.0, self.eps) >= self.target_alignment:
            return 0.0
        if _power_alignment(sigma, 1.0, self.eps) < self.target_alignment:
            return 1.0

        lo = 0.0
        hi = 1.0
        for _ in range(self.search_steps):
            mid = 0.5 * (lo + hi)
            if _power_alignment(sigma, mid, self.eps) >= self.target_alignment:
                hi = mid
            else:
                lo = mid
        return hi

    def _alpha_for_effective_rank(self, sigma: torch.Tensor) -> float:
        target = self._target_erank(sigma)
        if _power_effective_rank(sigma, 1.0, self.eps) >= target:
            return 1.0
        if _power_effective_rank(sigma, 0.0, self.eps) < target:
            return 0.0

        lo = 0.0
        hi = 1.0
        for _ in range(self.search_steps):
            mid = 0.5 * (lo + hi)
            if _power_effective_rank(sigma, mid, self.eps) >= target:
                lo = mid
            else:
                hi = mid
        return lo

    def _rank_for_alignment(self, sigma: torch.Tensor) -> int:
        lo = 1
        hi = sigma.numel()
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            weights = _capped_response_weights(
                sigma,
                mid,
                self.eps,
                self.search_steps,
            )
            if _singular_alignment(sigma, weights, self.eps) >= self.target_alignment:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _weights_one(self, sigma: torch.Tensor) -> torch.Tensor:
        if self.mode == "power":
            alpha = self.alpha
            weights = _power_response_weights(sigma, alpha, self.eps)
            self._record_power_stats(sigma, weights, alpha)
            return weights
        if self.mode == "power-rho":
            alpha = self._alpha_for_alignment(sigma)
            weights = _power_response_weights(sigma, alpha, self.eps)
            self._record_power_stats(sigma, weights, alpha)
            return weights
        if self.mode == "power-erank":
            alpha = self._alpha_for_effective_rank(sigma)
            weights = _power_response_weights(sigma, alpha, self.eps)
            self._record_power_stats(sigma, weights, alpha)
            return weights

        if self.mode == "cap-rho":
            rank = self._rank_for_alignment(sigma)
        else:
            rank = self._target_rank(sigma)
        weights = _capped_response_weights(
            sigma,
            rank,
            self.eps,
            self.search_steps,
        )
        self._record_cap_stats(sigma, weights, rank)
        return weights

    def _weights(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            return self._weights_one(sigma)
        if self.mode == "power":
            if self.alpha <= self.eps:
                weights = torch.ones_like(sigma)
            else:
                weights = sigma.clamp_min(0.0).pow(self.alpha)
            self._record_power_stats_batch(sigma, weights, self.alpha)
            return weights
        if self.mode == "cap-rank":
            rank = self._target_rank(sigma[0])
            weights = _capped_response_weights_batch(
                sigma,
                rank,
                self.eps,
                self.search_steps,
            )
            self._record_cap_stats_batch(sigma, weights, rank)
            return weights
        return torch.stack([self._weights_one(row) for row in sigma])

    def _active_mask(self, sigma: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (
            torch.linalg.vector_norm(sigma, dim=-1) > self.eps
        ) & (torch.linalg.vector_norm(weights, dim=-1) > self.eps)

    def _record_power_stats_batch(
        self,
        sigma: torch.Tensor,
        weights: torch.Tensor,
        alpha: float,
    ) -> None:
        active = self._active_mask(sigma, weights)
        active_f = active.to(dtype=torch.float32)
        active_count = active_f.sum()
        self._stat_add("active_calls", active_count)
        self._stat_add("alpha_sum", active_count * float(alpha))
        self._stat_add(
            "effective_rank_sum",
            (_batch_effective_rank(weights, self.eps) * active_f).sum(),
        )
        self._stat_add(
            "stable_rank_sum",
            (_batch_stable_rank(weights, self.eps) * active_f).sum(),
        )
        self._stat_add(
            "alignment_sum",
            (_batch_alignment(sigma, weights) * active_f).sum(),
        )

    def _record_cap_stats_batch(
        self,
        sigma: torch.Tensor,
        weights: torch.Tensor,
        rank: int,
    ) -> None:
        active = self._active_mask(sigma, weights)
        active_f = active.to(dtype=torch.float32)
        active_count = active_f.sum()
        self._stat_add("active_calls", active_count)
        self._stat_add("rank_sum", active_count * float(rank))
        self._stat_add(
            "effective_rank_sum",
            (_batch_effective_rank(weights, self.eps) * active_f).sum(),
        )
        self._stat_add(
            "stable_rank_sum",
            (_batch_stable_rank(weights, self.eps) * active_f).sum(),
        )
        self._stat_add(
            "alignment_sum",
            (_batch_alignment(sigma, weights) * active_f).sum(),
        )

    def _record_power_stats(
        self,
        sigma: torch.Tensor,
        weights: torch.Tensor,
        alpha: float,
    ) -> None:
        if (
            float(torch.linalg.vector_norm(sigma)) <= self.eps
            or float(torch.linalg.vector_norm(weights)) <= self.eps
        ):
            return
        self.stats["active_calls"] += 1
        self.stats["alpha_sum"] += float(alpha)
        self.stats["effective_rank_sum"] += _singular_effective_rank(
            weights,
            self.eps,
        )
        self.stats["stable_rank_sum"] += _singular_stable_rank(weights, self.eps)
        self.stats["alignment_sum"] += _singular_alignment(sigma, weights, self.eps)

    def _record_cap_stats(
        self,
        sigma: torch.Tensor,
        weights: torch.Tensor,
        rank: int,
    ) -> None:
        if (
            float(torch.linalg.vector_norm(sigma)) <= self.eps
            or float(torch.linalg.vector_norm(weights)) <= self.eps
        ):
            return
        self.stats["active_calls"] += 1
        self.stats["rank_sum"] += float(rank)
        self.stats["effective_rank_sum"] += _singular_effective_rank(
            weights,
            self.eps,
        )
        self.stats["stable_rank_sum"] += _singular_stable_rank(weights, self.eps)
        self.stats["alignment_sum"] += _singular_alignment(sigma, weights, self.eps)

    def _response(
        self,
        mv: torch.Tensor,
        sigma: torch.Tensor,
        v: torch.Tensor,
        raw_sigma: torch.Tensor,
    ) -> torch.Tensor:
        weights = self._weights(raw_sigma)
        return (mv * sigma.reciprocal().unsqueeze(-2) * weights.unsqueeze(-2)) @ v.mT
