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
        candidate = param + direction * (lr * radius * sqrt_d / u_norm.clamp_min(tiny))
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


class ScheduleFreeHyperball(Hyperball):
    """Schedule-free fixed-radius Hyperball.

    The trainable tensor stores the Schedule-Free query point.  Each parameter
    also keeps the raw iterate z and averaged/evaluation iterate x in optimizer
    state:

        z_t = Retr_R(z_{t-1} + lr * atom(grad at y_{t-1}))
        x_t = (1 - c_t) x_{t-1} + c_t z_t
        y_t = beta_t x_t + (1 - beta_t) z_t

    The ambient x recurrence is deliberately not projected: projecting it would
    destroy the linear averaging identity used by Schedule-Free's prediction
    argument.  By default only the ScionH base iterate z lives on the fixed-RMS
    sphere.  The `geodesic` geometry is the fixed-RMS analogue: x and y are
    computed by spherical interpolation on the same RMS sphere.
    """

    _SF_GEOMETRIES = {"ambient", "geodesic"}

    def __init__(
        self,
        params,
        lr: float = 0.01,
        beta: float = 0.95,
        ulmo=None,
        update_rule: str = "retract",
        sf_beta: float = 0.9,
        sf_beta_final: float | None = None,
        sf_beta_anneal_steps: int = 0,
        sf_r: float = 0.0,
        sf_c_warmup_steps: int = 0,
        sf_weight_lr_power: float = 2.0,
        sf_polyak: bool = False,
        sf_polyak_beta: float = 0.9,
        sf_polyak_f_star: float = 0.0,
        sf_polyak_eps: float = 1e-12,
        sf_polyak_max_scale: float = 0.0,
        sf_geometry: str = "ambient",
    ):
        if update_rule != "retract":
            raise ValueError(
                "ScheduleFreeHyperball currently supports update_rule='retract'"
            )
        if not (0.0 <= sf_beta < 1.0):
            raise ValueError(f"invalid schedule-free beta: {sf_beta}")
        if sf_beta_final is None:
            sf_beta_final = sf_beta
        if not (0.0 <= sf_beta_final < 1.0):
            raise ValueError(f"invalid schedule-free final beta: {sf_beta_final}")
        if sf_beta_anneal_steps < 0:
            raise ValueError(
                f"invalid schedule-free beta anneal steps: {sf_beta_anneal_steps}"
            )
        if sf_c_warmup_steps < 0:
            raise ValueError(
                f"invalid schedule-free c warmup steps: {sf_c_warmup_steps}"
            )
        if sf_weight_lr_power < 0.0:
            raise ValueError(
                f"invalid schedule-free weight lr power: {sf_weight_lr_power}"
            )
        if not (0.0 <= sf_polyak_beta < 1.0):
            raise ValueError(f"invalid schedule-free Polyak beta: {sf_polyak_beta}")
        if sf_polyak_eps <= 0.0:
            raise ValueError(f"invalid schedule-free Polyak eps: {sf_polyak_eps}")
        if sf_polyak_max_scale < 0.0:
            raise ValueError(
                f"invalid schedule-free Polyak max scale: {sf_polyak_max_scale}"
            )
        if sf_geometry not in self._SF_GEOMETRIES:
            valid = ", ".join(sorted(self._SF_GEOMETRIES))
            raise ValueError(f"invalid schedule-free geometry: {sf_geometry}; {valid}")

        self._sf_polyak_loss: float | None = None
        super().__init__(
            params,
            lr=lr,
            beta=beta,
            ulmo=ulmo,
            update_rule=update_rule,
        )
        for group in self.param_groups:
            group.setdefault("sf_beta", sf_beta)
            group.setdefault("sf_beta_final", sf_beta_final)
            group.setdefault("sf_beta_anneal_steps", sf_beta_anneal_steps)
            group.setdefault("sf_r", sf_r)
            group.setdefault("sf_c_warmup_steps", sf_c_warmup_steps)
            group.setdefault("sf_weight_lr_power", sf_weight_lr_power)
            group.setdefault("sf_polyak", sf_polyak)
            group.setdefault("sf_polyak_beta", sf_polyak_beta)
            group.setdefault("sf_polyak_f_star", sf_polyak_f_star)
            group.setdefault("sf_polyak_eps", sf_polyak_eps)
            group.setdefault("sf_polyak_max_scale", sf_polyak_max_scale)
            group.setdefault("sf_geometry", sf_geometry)
            group.setdefault("sf_polyak_ema", 0.0)
            group.setdefault("sf_polyak_step", 0)
            group.setdefault("sf_polyak_scale", 1.0)
            group.setdefault("sf_polyak_slope", 0.0)
            group.setdefault("sf_polyak_numerator", 0.0)
            group.setdefault("sf_weight_sum", 0.0)
            group.setdefault("sf_step", 0)
            group.setdefault("sf_lr_max", torch.finfo(torch.float32).eps)
            group["base_update"] = "schedule-free-retract"

    def set_polyak_loss_(self, loss: float | None) -> None:
        self._sf_polyak_loss = None if loss is None else float(loss)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group_work = []
        for group in self.param_groups:
            entries = self._collect_entries(group)
            if not entries:
                continue

            lr = float(group["lr"])
            if lr == 0.0:
                updates = [torch.zeros_like(m) for _, _, m in entries]
            else:
                updates = self._updates(group["ulmo"], entries)
            group_work.append((group, entries, updates, lr))

        polyak_scale = self._polyak_scale(group_work)
        for group, entries, updates, lr in group_work:
            effective_lr = lr * polyak_scale
            group["sf_effective_lr"] = effective_lr
            self._schedule_free_group_(group, entries, updates, effective_lr)
            group["sf_step"] = int(group.get("sf_step", 0)) + 1

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
                raise RuntimeError(
                    "ScheduleFreeHyperball does not support sparse gradients"
                )

            state = self.state[p]
            if "m" not in state:
                state["m"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                d = p.numel()
                state["R"] = float((p.data.square().sum() / d).sqrt())
                state["x"] = p.detach().clone(memory_format=torch.preserve_format)
                state["z"] = p.detach().clone(memory_format=torch.preserve_format)

            m = state["m"]
            momenta.append(m)
            grads.append(g)
            entries.append((param_index, p, m))
        if momenta:
            torch._foreach_lerp_(momenta, grads, 1.0 - beta)
        return entries

    def _schedule_free_group_(
        self,
        group: dict,
        entries: list[tuple[int, torch.Tensor, torch.Tensor]],
        updates: list[torch.Tensor],
        lr: float,
    ) -> None:
        z_tensors = []
        radius_dims = []
        active_entries = []
        active_updates = []
        for entry, update in zip(entries, updates, strict=True):
            _, p, _ = entry
            state = self.state[p]
            R = state["R"]
            if R <= 0.0:
                continue
            z_tensors.append(state["z"])
            radius_dims.append((float(R), math.sqrt(p.numel())))
            active_entries.append(entry)
            active_updates.append(update)

        if not active_entries:
            return

        if lr != 0.0:
            self._retract_tensors_(z_tensors, active_updates, radius_dims, lr)
        self._update_average_and_query_(group, active_entries, lr)

    def _retract_tensors_(
        self,
        tensors: list[torch.Tensor],
        dirs: list[torch.Tensor],
        radius_dims: list[tuple[float, float]],
        lr: float,
    ) -> None:
        groups = defaultdict(list)
        for i, tensor in enumerate(tensors):
            groups[(tuple(tensor.shape), tensor.dtype, tensor.device)].append(i)

        if len(groups) < len(tensors):
            self._retract_shape_groups_(tensors, dirs, radius_dims, lr, groups)
            return

        foreach_groups = defaultdict(list)
        for i, tensor in enumerate(tensors):
            foreach_groups[(tensor.dtype, tensor.device)].append(i)
        for indices in foreach_groups.values():
            self._retract_foreach_(
                [tensors[i] for i in indices],
                [dirs[i] for i in indices],
                [radius_dims[i] for i in indices],
                lr,
            )

    def _update_average_and_query_(
        self,
        group: dict,
        entries: list[tuple[int, torch.Tensor, torch.Tensor]],
        lr: float,
    ) -> None:
        step = int(group.get("sf_step", 0)) + 1
        c = self._schedule_free_c(group, step, lr)
        beta = self._schedule_free_beta(group, step)
        geometry = self._schedule_free_geometry(group)
        if geometry == "geodesic":
            self._update_geodesic_average_and_query_(entries, c, beta)
            return

        for _, p, _ in entries:
            state = self.state[p]
            x = state["x"]
            z = state["z"]
            if c >= 1.0:
                x.copy_(z)
            else:
                x.lerp_(z, c)
            p.data.copy_(x).mul_(beta).add_(z, alpha=1.0 - beta)

    def _update_geodesic_average_and_query_(
        self,
        entries: list[tuple[int, torch.Tensor, torch.Tensor]],
        c: float,
        beta: float,
    ) -> None:
        y_weight = 1.0 - beta * (1.0 - c)
        if c >= 1.0:
            x_tensors = []
            params = []
            z_tensors = []
            for _, p, _ in entries:
                state = self.state[p]
                x_tensors.append(state["x"])
                params.append(p.data)
                z_tensors.append(state["z"])
            torch._foreach_copy_(x_tensors, z_tensors)
            torch._foreach_copy_(params, z_tensors)
            return

        groups = defaultdict(list)
        for i, (_, p, _) in enumerate(entries):
            groups[(tuple(p.shape), p.dtype, p.device)].append(i)

        if len(groups) < len(entries):
            for indices in groups.values():
                if len(indices) == 1:
                    i = indices[0]
                    _, p, _ = entries[i]
                    state = self.state[p]
                    self._slerp_rms_pair_(
                        state["x"],
                        p.data,
                        state["x"],
                        state["z"],
                        c,
                        y_weight,
                        float(state["R"]),
                    )
                    continue
                self._slerp_rms_pair_batch_(entries, indices, c, y_weight)
            return

        for _, p, _ in entries:
            state = self.state[p]
            self._slerp_rms_pair_(
                state["x"],
                p.data,
                state["x"],
                state["z"],
                c,
                y_weight,
                float(state["R"]),
            )

    def _schedule_free_geometry(self, group: dict) -> str:
        geometry = group.get("sf_geometry", "ambient")
        if geometry not in self._SF_GEOMETRIES:
            valid = ", ".join(sorted(self._SF_GEOMETRIES))
            raise ValueError(f"invalid schedule-free geometry: {geometry}; {valid}")
        return str(geometry)

    def _schedule_free_c(self, group: dict, step: int, lr: float) -> float:
        if step <= int(group.get("sf_c_warmup_steps", 0)):
            return 1.0

        lr_max = max(float(group.get("sf_lr_max", 0.0)), abs(float(lr)))
        group["sf_lr_max"] = lr_max
        r = float(group.get("sf_r", 0.0))
        lr_power = float(group.get("sf_weight_lr_power", 2.0))
        weight = (float(step) ** r) * (lr_max ** lr_power)
        weight = max(weight, torch.finfo(torch.float32).eps)
        weight_sum = float(group.get("sf_weight_sum", 0.0)) + weight
        group["sf_weight_sum"] = weight_sum
        return weight / weight_sum

    def _polyak_scale(self, group_work: list[tuple[dict, list, list, float]]) -> float:
        if not group_work or not bool(group_work[0][0].get("sf_polyak", False)):
            return 1.0
        if self._sf_polyak_loss is None or not math.isfinite(self._sf_polyak_loss):
            return 1.0

        numerator = self._sf_polyak_loss - float(
            group_work[0][0].get("sf_polyak_f_star", 0.0)
        )
        slope = 0.0
        for group, entries, updates, lr in group_work:
            correction, group_slope = self._polyak_group_terms(group, entries, updates)
            numerator += correction
            slope += group_slope

        group0 = group_work[0][0]
        beta = float(group0.get("sf_polyak_beta", 0.9))
        eps = float(group0.get("sf_polyak_eps", 1e-12))
        step = int(group0.get("sf_polyak_step", 0)) + 1
        ema = beta * float(group0.get("sf_polyak_ema", 0.0))
        ema += (1.0 - beta) * max(slope, 0.0)
        denom = ema / max(1.0 - beta**step, eps)
        scale = max(numerator, 0.0) / max(denom, eps)
        max_scale = float(group0.get("sf_polyak_max_scale", 0.0))
        if max_scale > 0.0:
            scale = min(scale, max_scale)
        if not math.isfinite(scale):
            scale = 1.0

        for group, _, _, _ in group_work:
            group["sf_polyak_ema"] = ema
            group["sf_polyak_step"] = step
            group["sf_polyak_scale"] = scale
            group["sf_polyak_slope"] = slope
            group["sf_polyak_numerator"] = numerator
        return scale

    def _polyak_group_terms(
        self,
        group: dict,
        entries: list[tuple[int, torch.Tensor, torch.Tensor]],
        updates: list[torch.Tensor],
    ) -> tuple[float, float]:
        correction = 0.0
        slope = 0.0
        base_lr = abs(float(group.get("lr_peak", group.get("lr", 1.0))))
        for (_, p, _), update in zip(entries, updates, strict=True):
            grad = p.grad
            if grad is None:
                continue
            state = self.state[p]
            z = state.get("z")
            R = state.get("R")
            if z is None or R is None or R <= 0.0:
                continue

            grad_f = grad.detach().float()
            correction += float(
                (grad_f * (z.detach().float() - p.detach().float())).sum()
            )

            update_f = update.detach().float()
            update_norm = float(torch.linalg.vector_norm(update_f))
            if update_norm <= 0.0:
                continue

            radius = float(R) * math.sqrt(p.numel())
            if radius <= 0.0:
                continue
            atom_unit = update_f / update_norm
            z_unit = z.detach().float() / radius
            radial = (atom_unit * z_unit).sum()
            tangent = atom_unit - radial * z_unit
            pairing = -float((grad_f * tangent).sum())
            slope += max(pairing, 0.0) * base_lr * radius
        return correction, slope

    def _schedule_free_beta(self, group: dict, step: int) -> float:
        beta_initial = float(group.get("sf_beta", 0.9))
        beta_final = float(group.get("sf_beta_final", beta_initial))
        anneal_steps = int(group.get("sf_beta_anneal_steps", 0))
        if anneal_steps <= 0 or beta_initial == beta_final:
            return beta_initial
        tau = min(float(step) / anneal_steps, 1.0)
        log_one_minus = (1.0 - tau) * math.log1p(-beta_initial)
        log_one_minus += tau * math.log1p(-beta_final)
        return 1.0 - math.exp(log_one_minus)

    def _project_rms_(self, tensor: torch.Tensor, radius: float) -> None:
        if radius <= 0.0 or tensor.numel() == 0:
            return
        tiny = torch.finfo(tensor.dtype).tiny
        target = radius * math.sqrt(tensor.numel())
        norm = torch.linalg.vector_norm(tensor).clamp_min(tiny)
        tensor.mul_(target / norm)

    def _slerp_coefficients(
        self,
        theta: torch.Tensor,
        sin_theta: torch.Tensor,
        weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = sin_theta.new_tensor(1e-6)
        safe_sin = sin_theta.clamp_min(eps)
        small = sin_theta.abs() <= eps
        left = torch.where(
            small,
            theta.new_tensor(1.0 - weight),
            torch.sin((1.0 - weight) * theta) / safe_sin,
        )
        right = torch.where(
            small,
            theta.new_tensor(weight),
            torch.sin(weight * theta) / safe_sin,
        )
        return left, right

    def _slerp_rms_pair_(
        self,
        x_dest: torch.Tensor,
        y_dest: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        x_weight: float,
        y_weight: float,
        radius: float,
    ) -> None:
        if radius <= 0.0 or x_dest.numel() == 0:
            x_dest.copy_(start)
            y_dest.copy_(start)
            return

        target = radius * math.sqrt(x_dest.numel())
        inv_target = 1.0 / target
        u = start.detach().float() * inv_target
        v = end.detach().float() * inv_target
        dot = (u * v).sum().clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        x_left, x_right = self._slerp_coefficients(theta, sin_theta, x_weight)
        y_left, y_right = self._slerp_coefficients(theta, sin_theta, y_weight)
        x_result = target * (x_left * u + x_right * v)
        y_result = target * (y_left * u + y_right * v)
        x_dest.copy_(x_result.to(dtype=x_dest.dtype))
        y_dest.copy_(y_result.to(dtype=y_dest.dtype))

    def _slerp_rms_pair_batch_(
        self,
        entries: list[tuple[int, torch.Tensor, torch.Tensor]],
        indices: list[int],
        x_weight: float,
        y_weight: float,
    ) -> None:
        xs = []
        ys = []
        starts = []
        ends = []
        radii = []
        for i in indices:
            _, p, _ = entries[i]
            state = self.state[p]
            xs.append(state["x"])
            ys.append(p.data)
            starts.append(state["x"])
            ends.append(state["z"])
            radii.append(float(state["R"]))

        start = torch.stack([tensor.detach().float() for tensor in starts])
        end = torch.stack([tensor.detach().float() for tensor in ends])
        target = start.new_tensor(radii) * math.sqrt(starts[0].numel())
        view = (len(indices),) + (1,) * starts[0].ndim
        target_view = target.view(view)
        u = start / target_view
        v = end / target_view
        dot = (u * v).flatten(1).sum(dim=1).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        x_left, x_right = self._slerp_coefficients(theta, sin_theta, x_weight)
        y_left, y_right = self._slerp_coefficients(theta, sin_theta, y_weight)
        x_view = x_left.view(view) * u + x_right.view(view) * v
        y_view = y_left.view(view) * u + y_right.view(view) * v
        x_result = target_view * x_view
        y_result = target_view * y_view
        torch._foreach_copy_(xs, list(x_result.to(dtype=xs[0].dtype).unbind(0)))
        torch._foreach_copy_(ys, list(y_result.to(dtype=ys[0].dtype).unbind(0)))

    def _slerp_rms_(
        self,
        dest: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        weight: float,
        radius: float,
    ) -> None:
        if weight <= 0.0:
            dest.copy_(start)
            self._project_rms_(dest, radius)
            return
        if weight >= 1.0:
            dest.copy_(end)
            self._project_rms_(dest, radius)
            return
        if radius <= 0.0 or dest.numel() == 0:
            dest.copy_(start)
            return

        target = radius * math.sqrt(dest.numel())
        inv_target = 1.0 / target
        u = start.detach().float() * inv_target
        v = end.detach().float() * inv_target
        dot = (u * v).sum().clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        left, right = self._slerp_coefficients(theta, sin_theta, weight)
        result = target * (left * u + right * v)
        dest.copy_(result.to(dtype=dest.dtype))

    @torch.no_grad()
    def eval_parameters_(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                x = state.get("x")
                if x is None:
                    continue
                p.data.copy_(x)

    @torch.no_grad()
    def train_parameters_(self) -> None:
        for group in self.param_groups:
            beta = self._schedule_free_beta(
                group, max(int(group.get("sf_step", 0)), 1)
            )
            geometry = self._schedule_free_geometry(group)
            for p in group["params"]:
                state = self.state[p]
                x = state.get("x")
                z = state.get("z")
                R = state.get("R")
                if x is None or z is None or R is None:
                    continue
                if geometry == "geodesic":
                    self._slerp_rms_(p.data, z, x, beta, float(R))
                    continue
                p.data.copy_(x).mul_(beta).add_(z, alpha=1.0 - beta)
