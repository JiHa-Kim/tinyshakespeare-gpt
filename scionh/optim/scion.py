import math
from collections import defaultdict

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

            if update_rule == "retract":
                self._retract_group_(entries, updates, lr)
                continue

            for (_, p, _), u in zip(entries, updates, strict=True):
                state = self.state[p]
                R = state["R"]
                if R <= 0.0:
                    continue

                d = p.numel()
                w_hat = p.data / R

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
        momenta = []
        grads = []
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
            momenta.append(m)
            grads.append(g)
            entries.append((param_index, p, m))
        if momenta:
            torch._foreach_lerp_(momenta, grads, 1.0 - beta)
        return entries

    def _retract_group_(
        self,
        entries: list[tuple[int, torch.Tensor, torch.Tensor]],
        updates: list[torch.Tensor],
        lr: float,
    ) -> None:
        params = []
        dirs = []
        radius_dims = []
        for (_, p, _), u in zip(entries, updates, strict=True):
            R = self.state[p]["R"]
            if R <= 0.0:
                continue
            params.append(p.data)
            dirs.append(u)
            radius_dims.append((float(R), math.sqrt(p.numel())))
        if not params:
            return

        groups = defaultdict(list)
        for i, p in enumerate(params):
            key = (tuple(p.shape), p.dtype, p.device)
            groups[key].append(i)

        if len(groups) < len(params):
            self._retract_shape_groups_(params, dirs, radius_dims, lr, groups)
            return

        foreach_groups = defaultdict(list)
        for i, p in enumerate(params):
            foreach_groups[(p.dtype, p.device)].append(i)
        for indices in foreach_groups.values():
            self._retract_foreach_(
                [params[i] for i in indices],
                [dirs[i] for i in indices],
                [radius_dims[i] for i in indices],
                lr,
            )

    def _retract_foreach_(
        self,
        params: list[torch.Tensor],
        dirs: list[torch.Tensor],
        radius_dims: list[tuple[float, float]],
        lr: float,
    ) -> None:
        tiny = torch.finfo(dirs[0].dtype).tiny
        dir_norms = torch._foreach_norm(dirs)
        one = dir_norms[0].new_ones(())
        step_scales = [
            norm.new_tensor(lr * R * sqrt_d)
            / torch.where(norm > 0, norm.clamp_min(tiny), one)
            for norm, (R, sqrt_d) in zip(dir_norms, radius_dims, strict=True)
        ]
        deltas = torch._foreach_mul(dirs, step_scales)
        candidates = torch._foreach_add(params, deltas)
        candidate_norms = torch._foreach_norm(candidates)
        retract_scales = [
            norm.new_tensor(R * sqrt_d) / norm.clamp_min(tiny)
            for norm, (R, sqrt_d) in zip(candidate_norms, radius_dims, strict=True)
        ]
        retracted = torch._foreach_mul(candidates, retract_scales)
        torch._foreach_copy_(params, retracted)

    def _retract_shape_groups_(
        self,
        params: list[torch.Tensor],
        dirs: list[torch.Tensor],
        radius_dims: list[tuple[float, float]],
        lr: float,
        groups,
    ) -> None:
        for indices in groups.values():
            if len(indices) == 1:
                i = indices[0]
                self._retract_single_(params[i], dirs[i], radius_dims[i], lr)
                continue

            p_batch = torch.stack([params[i] for i in indices])
            u_batch = torch.stack([dirs[i] for i in indices])
            radii = p_batch.new_tensor([radius_dims[i][0] for i in indices])
            sqrt_d = p_batch.new_tensor([radius_dims[i][1] for i in indices])
            tiny = torch.finfo(u_batch.dtype).tiny
            view = (len(indices),) + (1,) * (u_batch.ndim - 1)

            u_norm = torch.linalg.vector_norm(u_batch, dim=tuple(range(1, u_batch.ndim)))
            safe_u_norm = torch.where(
                u_norm > 0, u_norm.clamp_min(tiny), torch.ones_like(u_norm)
            )
            step_scale = (lr * radii * sqrt_d / safe_u_norm).view(view)
            candidate = p_batch + step_scale * u_batch
            candidate_norm = torch.linalg.vector_norm(
                candidate, dim=tuple(range(1, candidate.ndim))
            )
            retract_scale = (radii * sqrt_d / candidate_norm.clamp_min(tiny)).view(view)
            retracted = retract_scale * candidate
            torch._foreach_copy_(
                [params[i] for i in indices],
                list(retracted.unbind(0)),
            )

    def _retract_single_(
        self,
        param: torch.Tensor,
        direction: torch.Tensor,
        radius_dim: tuple[float, float],
        lr: float,
    ) -> None:
        radius, sqrt_d = radius_dim
        tiny = torch.finfo(direction.dtype).tiny
        u_norm = torch.linalg.vector_norm(direction)
        if float(u_norm) <= 0.0:
            return
        u_norm = u_norm.clamp_min(tiny)
        candidate = param + direction * (lr * radius * sqrt_d / u_norm)
        candidate_norm = torch.linalg.vector_norm(candidate).clamp_min(tiny)
        param.copy_(candidate * (radius * sqrt_d / candidate_norm))

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
