import math
from dataclasses import dataclass

import torch

from scionh.ulmos.core import gram_newton_schulz_polar
from scionh.models.gpt import GPT


@dataclass
class ConvergenceItem:
    name: str
    group: str
    opt_group: dict
    param: torch.Tensor
    ulmo: object
    state: dict


_SpectralPowerKey = tuple[int, str]
_SpectralPowerGroupKey = tuple[torch.device, tuple[int, int], int]
_SpectralPowerGroupItem = tuple[_SpectralPowerKey, torch.Tensor, bool]
_PrevState = tuple[torch.Tensor, torch.Tensor] | None
_ConvergenceRecord = tuple[ConvergenceItem, torch.Tensor, _PrevState]
_SpectralDualRequest = tuple[int, ConvergenceItem, torch.Tensor]
_SpectralDualGroupKey = tuple[torch.device, tuple[int, int]]
_NuclearRequest = tuple[int, torch.Tensor]
_NuclearGroupKey = tuple[torch.device, tuple[int, int]]
_SpectralOracleResult = tuple[float, torch.Tensor]
_STREAMING_POWER_COLD_STEPS = 4
_STREAMING_POWER_WARM_STEPS = 1
_NUCLEAR_SUPPORT_STEPS = 7


class StreamingSpectralNormEstimator:
    def __init__(
        self,
        eps: float,
        cold_steps: int = _STREAMING_POWER_COLD_STEPS,
        warm_steps: int = _STREAMING_POWER_WARM_STEPS,
    ):
        self.eps = eps
        self.cold_steps = cold_steps
        self.warm_steps = warm_steps
        self.vectors: dict[_SpectralPowerKey, torch.Tensor] = {}

    @torch.no_grad()
    def estimate(
        self, requests: list[tuple[_SpectralPowerKey, torch.Tensor]]
    ) -> dict[_SpectralPowerKey, float]:
        results: dict[_SpectralPowerKey, float] = {}
        groups: dict[_SpectralPowerGroupKey, list[_SpectralPowerGroupItem]] = {}

        for key, x in requests:
            if x.ndim != 2 or x.numel() == 0 or x.device.type != "cuda":
                results[key] = spectral_norm_power(x, self.eps)
                continue
            vector = self.vectors.get(key)
            warm = (
                vector is not None
                and vector.device == x.device
                and vector.numel() == x.size(1)
            )
            group_key = (
                x.device,
                (x.size(0), x.size(1)),
                self.warm_steps if warm else self.cold_steps,
            )
            groups.setdefault(group_key, []).append((key, x.detach(), warm))

        for (_, _, steps), items in groups.items():
            x_batch = torch.stack([x.float() for _, x, _ in items]).contiguous()
            v_batch = torch.stack(
                [
                    self.vectors[key].to(x_batch.device, dtype=torch.float32)
                    if warm
                    else torch.ones(x_batch.size(2), device=x_batch.device)
                    for key, _, warm in items
                ]
            ).unsqueeze(-1)
            v_batch = self._normalize(v_batch)

            for _ in range(steps):
                u_batch = self._normalize(torch.bmm(x_batch, v_batch))
                v_batch = self._normalize(torch.bmm(x_batch.transpose(1, 2), u_batch))

            sigma = torch.linalg.vector_norm(
                torch.bmm(x_batch, v_batch), dim=1
            ).squeeze(-1)
            for (key, _, _), value, vector in zip(
                items, sigma.detach().cpu().tolist(), v_batch.squeeze(-1)
            ):
                results[key] = max(float(value), self.eps)
                self.vectors[key] = vector.detach()

        return results

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(self.eps)


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def spectral_ulmo_scale(x: torch.Tensor, ulmo) -> float:
    return float(ulmo.geometry.scale(x))


def is_spectral_ulmo(ulmo) -> bool:
    return bool(ulmo.geometry.is_spectral)


def uses_global_spectral_support(ulmo) -> bool:
    return bool(ulmo.geometry.kind == "spectral")


def dual_norm(
    x: torch.Tensor, ulmo, eps: float = 1e-12, param: torch.Tensor | None = None
) -> float:
    custom = getattr(ulmo, "dual_norm", None)
    if custom is not None:
        return float(custom(x, eps))
    x = x.float()
    geometry = ulmo.geometry
    if geometry.is_spectral:
        return spectral_nuclear_support_estimate(x, eps) * spectral_ulmo_scale(x, ulmo)
    return float(geometry.dual_norm(x, eps))


def spectral_norm_power(x: torch.Tensor, eps: float = 1e-12, steps: int = 4) -> float:
    x = x.float()
    if x.ndim != 2 or x.numel() == 0:
        return float(torch.linalg.vector_norm(x).clamp_min(eps))
    v = torch.ones(x.size(1), dtype=x.dtype, device=x.device)
    v = v / torch.linalg.vector_norm(v).clamp_min(eps)
    for _ in range(steps):
        u = x @ v
        u_norm = torch.linalg.vector_norm(u)
        if float(u_norm) <= eps:
            return 0.0
        u = u / u_norm
        v = x.mT @ u
        v_norm = torch.linalg.vector_norm(v)
        if float(v_norm) <= eps:
            return 0.0
        v = v / v_norm
    return float(torch.linalg.vector_norm(x @ v).clamp_min(eps))


@torch.no_grad()
def spectral_nuclear_support_batch(
    batch: torch.Tensor,
    steps: int = _NUCLEAR_SUPPORT_STEPS,
    eps: float = 1e-7,
    work_dtype: torch.dtype | None = torch.float16,
) -> torch.Tensor:
    polar = gram_newton_schulz_polar(
        batch, steps, eps, work_dtype, 1.05, compile_scale=False
    ).float()
    return (batch.float() * polar).sum(dim=(-2, -1)).abs()


def spectral_nuclear_support_estimate(
    x: torch.Tensor, eps: float = 1e-12, steps: int = _NUCLEAR_SUPPORT_STEPS
) -> float:
    x = x.float()
    if x.ndim != 2 or x.numel() == 0:
        return float(torch.linalg.vector_norm(x))
    work = x.mT if x.size(0) < x.size(1) else x
    return float(
        spectral_nuclear_support_batch(
            work.unsqueeze(0),
            steps,
            1e-7,
            torch.float16 if work.is_cuda else torch.float32,
        ).clamp_min(eps)
    )


def primal_norm(x: torch.Tensor, ulmo, eps: float = 1e-12) -> float:
    custom = getattr(ulmo, "primal_norm", None)
    if custom is not None:
        return float(custom(x, eps))
    x = x.float()
    geometry = ulmo.geometry
    if geometry.is_spectral:
        return spectral_norm_power(x, eps) / spectral_ulmo_scale(x, ulmo)
    return float(geometry.primal_norm(x, eps))


def activation_stats_from_input(
    x: torch.Tensor, eps: float = 1e-12
) -> dict[str, object]:
    flat = x.detach().reshape(-1, x.size(-1)).float()
    gram = (flat.mT @ flat).float()
    fro_sq = gram.diagonal().sum().clamp_min(eps)
    op_sq = torch.linalg.eigvalsh(gram).amax().clamp_min(eps)
    return {
        "fro": float(fro_sq.sqrt()),
        "op": float(op_sq.sqrt()),
        "stable_rank": float(fro_sq / op_sq),
        "gram": gram.detach(),
        "kind": "linear",
    }


def activation_norms_from_input(x: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    stats = activation_stats_from_input(x, eps)
    return {
        "fro": float(stats["fro"]),
        "op": float(stats["op"]),
        "stable_rank": float(stats["stable_rank"]),
    }


def stable_rank_from_input(x: torch.Tensor, eps: float = 1e-12) -> float:
    return activation_norms_from_input(x, eps)["stable_rank"]


def token_indicator_stats(tokens: torch.Tensor, eps: float = 1e-12) -> dict[str, object]:
    flat = tokens.detach().reshape(-1)
    total = int(flat.numel())
    if total <= 0:
        counts = torch.zeros(0, dtype=torch.float32, device=tokens.device)
        return {
            "fro": 0.0,
            "op": eps**0.5,
            "stable_rank": 0.0,
            "counts": counts,
            "kind": "embedding",
        }
    counts = torch.bincount(flat.to(dtype=torch.long)).float()
    max_count = float(counts.max()) if counts.numel() else 0.0
    fro_sq = float(total)
    op_sq = max(max_count, eps)
    return {
        "fro": fro_sq**0.5,
        "op": op_sq**0.5,
        "stable_rank": fro_sq / op_sq,
        "counts": counts.detach(),
        "kind": "embedding",
    }


def token_indicator_norms(tokens: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    stats = token_indicator_stats(tokens, eps)
    return {
        "fro": float(stats["fro"]),
        "op": float(stats["op"]),
        "stable_rank": float(stats["stable_rank"]),
    }


class ConvergenceProbe:
    def __init__(self, model: GPT, opt, args):
        self.interval = args.convergence_interval
        self.action_scale = args.convergence_action_scale
        self.eps = 1e-12
        self.active = False
        self.keep_prev = False
        self.keep_prev_gpu = False
        self.prev: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.prev_gpu: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.input_norms: dict[int, dict[str, object]] = {}
        self.summary: dict[str, dict[str, float]] = {}
        self.details: list[dict[str, float | str | list[int]]] = []
        self.spectral_norms = StreamingSpectralNormEstimator(self.eps)
        self.support_steps = max(
            1,
            int(getattr(args, "convergence_support_steps", _NUCLEAR_SUPPORT_STEPS)),
        )
        self.items = self._items(model, opt, args.convergence_probe)

    def _items(self, model: GPT, opt, probe: str) -> list[ConvergenceItem]:
        groups = {
            id(p): (
                group.get("name", "group"),
                group,
                group["ulmo"],
                opt.state[p],
            )
            for group in opt.param_groups
            for p in group["params"]
        }
        keep = self._probe_names(model) if probe == "representative" else None
        items = []
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) not in groups:
                continue
            if keep is not None and name not in keep:
                continue
            group, opt_group, ulmo, state = groups[id(p)]
            items.append(ConvergenceItem(name, group, opt_group, p, ulmo, state))
        return items

    def _probe_names(self, model: GPT) -> set[str]:
        names = {"tok_emb.weight", "lm_head.weight"}
        block_ids = sorted({0, len(model.blocks) // 2, len(model.blocks) - 1})
        suffixes = (
            "attn.qkv.weight",
            "attn.proj.weight",
            "mlp.up_gate.weight",
            "mlp.down.weight",
        )
        for block_id in block_ids:
            names.update(f"blocks.{block_id}.{suffix}" for suffix in suffixes)
        return names

    def start_step(self, step: int) -> None:
        self.active = self.interval > 0 and step % self.interval == 0
        self.keep_prev = self.active or (
            self.interval > 0 and (step + 1) % self.interval == 0
        )
        self.keep_prev_gpu = self.interval > 0 and (step + 1) % self.interval == 0
        if self.active:
            self.input_norms.clear()
            self.details = []

    def register_hooks(self, model: GPT):
        selected = {id(item.param) for item in self.items}
        handles = []
        for module in model.modules():
            if not isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
                continue
            weight = getattr(module, "weight", None)
            if weight is None or id(weight) not in selected:
                continue
            handles.append(module.register_forward_pre_hook(self._make_hook(module, weight)))
        return handles

    def _make_hook(self, module: torch.nn.Module, weight: torch.Tensor):
        def hook(module, inputs):
            if not (self.active and module.training and torch.is_grad_enabled()):
                return
            with torch.no_grad():
                if isinstance(module, torch.nn.Embedding):
                    self.input_norms[id(weight)] = token_indicator_stats(
                        inputs[0], self.eps
                    )
                else:
                    self.input_norms[id(weight)] = activation_stats_from_input(
                        inputs[0], self.eps
                    )

        return hook

    def _previous_tensor(
        self,
        item: ConvergenceItem,
        previous: _PrevState,
        index: int,
        shape: torch.Size,
        device: torch.device,
    ) -> torch.Tensor | None:
        previous_gpu = self.prev_gpu.get(id(item.param))
        if previous_gpu is not None and previous_gpu[index].shape == shape:
            return previous_gpu[index]
        if previous is None or previous[index].shape != shape:
            return None
        return previous[index].to(device, non_blocking=True)

    def _streaming_dparam_norms(
        self, records: list[_ConvergenceRecord]
    ) -> dict[int, float]:
        requests: list[tuple[_SpectralPowerKey, torch.Tensor]] = []
        scales: dict[_SpectralPowerKey, float] = {}
        for item, _, previous in records:
            if not self._can_stream_spectral_norm(item, item.param):
                continue
            key = (id(item.param), "dparam")
            prev_param = self._previous_tensor(
                item, previous, 1, item.param.shape, item.param.device
            )
            if prev_param is None:
                continue
            requests.append((key, item.param.detach().float() - prev_param))
            scales[key] = spectral_ulmo_scale(item.param, item.ulmo)
        estimates = self.spectral_norms.estimate(requests)
        return {
            key[0]: value / max(scales[key], self.eps)
            for key, value in estimates.items()
        }

    def _spectral_grad_dual_norms(
        self, records: list[_ConvergenceRecord]
    ) -> dict[int, float]:
        requests = [
            (id(item.param), item, grad.detach())
            for item, grad, _ in records
            if self._can_stream_spectral_norm(item, grad)
        ]
        return self._spectral_dual_norms(requests)

    def _spectral_dgrad_dual_norms(
        self, records: list[_ConvergenceRecord]
    ) -> dict[int, float]:
        requests: list[_SpectralDualRequest] = []
        for item, grad, previous in records:
            if not self._can_stream_spectral_norm(item, grad):
                continue
            prev_grad = self._previous_tensor(
                item, previous, 0, grad.shape, grad.device
            )
            if prev_grad is None:
                continue
            requests.append((id(item.param), item, grad.detach().float() - prev_grad))
        return self._spectral_dual_norms(requests)

    def _spectral_dual_norms(
        self, requests: list[_SpectralDualRequest]
    ) -> dict[int, float]:
        results: dict[int, float] = {}
        groups: dict[_SpectralDualGroupKey, list[_SpectralDualRequest]] = {}
        scales: dict[int, float] = {}

        for key, item, x in requests:
            work = x.detach().float()
            if work.size(0) < work.size(1):
                work = work.mT
            scales[key] = spectral_ulmo_scale(x, item.ulmo)
            groups.setdefault((work.device, tuple(work.shape)), []).append(
                (key, item, work)
            )

        for items in groups.values():
            batch = torch.stack([x for _, _, x in items]).contiguous()
            trace_norm = spectral_nuclear_support_batch(
                batch,
                self.support_steps,
                1e-7,
                torch.float16 if batch.is_cuda else torch.float32,
            )
            for (key, _, _), value in zip(
                items, trace_norm.detach().cpu().tolist(), strict=True
            ):
                results[key] = max(float(value) * scales[key], 0.0)
        return results

    def _effective_momentum(
        self, item: ConvergenceItem, grad: torch.Tensor
    ) -> torch.Tensor:
        beta = float(item.opt_group.get("beta", 0.0))
        previous = item.state.get("m")
        if previous is None:
            return grad.detach() * (1.0 - beta)
        return previous.detach().to(device=grad.device, dtype=grad.dtype).lerp(
            grad.detach(), 1.0 - beta
        )

    def _momentum_spectral_oracle_stats(
        self, records: list[_ConvergenceRecord]
    ) -> dict[int, _SpectralOracleResult]:
        requests: list[tuple[int, torch.Tensor, bool]] = []
        for item, grad, _ in records:
            momentum = self._effective_momentum(item, grad)
            if momentum.ndim != 2 or momentum.numel() == 0:
                continue
            work = momentum.detach().float()
            transposed = work.size(0) < work.size(1)
            if work.size(0) < work.size(1):
                work = work.mT
            requests.append((id(item.param), work, transposed))
        return self._spectral_oracle_stats(requests)

    def _spectral_oracle_stats(
        self, requests: list[tuple[int, torch.Tensor, bool]]
    ) -> dict[int, _SpectralOracleResult]:
        results: dict[int, _SpectralOracleResult] = {}
        groups: dict[_NuclearGroupKey, list[tuple[int, torch.Tensor, bool]]] = {}
        for key, work, transposed in requests:
            groups.setdefault((work.device, tuple(work.shape)), []).append(
                (key, work, transposed)
            )

        for items in groups.values():
            batch = torch.stack([x for _, x, _ in items]).contiguous()
            polar = gram_newton_schulz_polar(
                batch,
                self.support_steps,
                1e-7,
                torch.float16 if batch.is_cuda else torch.float32,
                1.05,
                compile_scale=False,
            ).float()
            trace_norm = (batch.float() * polar).sum(dim=(-2, -1)).abs()
            for (key, _, transposed), value, polar_item in zip(
                items, trace_norm.detach().cpu().tolist(), polar, strict=True
            ):
                atom = polar_item.mT if transposed else polar_item
                results[key] = (max(float(value), 0.0), atom.detach())
        return results

    def _nuclear_norms(self, requests: list[_NuclearRequest]) -> dict[int, float]:
        results: dict[int, float] = {}
        groups: dict[_NuclearGroupKey, list[_NuclearRequest]] = {}
        for key, work in requests:
            groups.setdefault((work.device, tuple(work.shape)), []).append((key, work))

        for items in groups.values():
            batch = torch.stack([x for _, x in items]).contiguous()
            trace_norm = spectral_nuclear_support_batch(
                batch,
                self.support_steps,
                1e-7,
                torch.float16 if batch.is_cuda else torch.float32,
            )
            for (key, _), value in zip(
                items, trace_norm.detach().cpu().tolist(), strict=True
            ):
                results[key] = max(float(value), 0.0)
        return results

    def _can_stream_spectral_norm(self, item: ConvergenceItem, x: torch.Tensor) -> bool:
        return (
            uses_global_spectral_support(item.ulmo)
            and x.ndim == 2
            and x.numel() > 0
            and x.device.type == "cuda"
        )

    def _has_previous(self, item: ConvergenceItem) -> bool:
        key = id(item.param)
        return key in self.prev or key in self.prev_gpu

    def _cpu_tensor(
        self, x: torch.Tensor, current: torch.Tensor | None
    ) -> torch.Tensor:
        return current if current is not None else x.detach().float().cpu()

    def _append_change_stats(
        self,
        stats: dict[str, list[float]],
        item: ConvergenceItem,
        grad: torch.Tensor,
        previous: _PrevState,
        grad_dual: float,
        lr: float,
        spectral_dgrad: dict[int, float],
        streaming_dparam: dict[int, float],
        current_grad: torch.Tensor | None,
        current_param: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self._has_previous(item):
            return current_grad, current_param

        key = id(item.param)
        dgrad = spectral_dgrad.get(key)
        if dgrad is None and previous is not None:
            prev_grad, _ = previous
            current_grad = self._cpu_tensor(grad, current_grad)
            dgrad = dual_norm(
                current_grad - prev_grad,
                item.ulmo,
                self.eps,
                item.param,
            )

        dparam = streaming_dparam.get(key)
        if dparam is None and previous is not None:
            _, prev_param = previous
            current_param = self._cpu_tensor(item.param, current_param)
            dparam = primal_norm(current_param - prev_param, item.ulmo, self.eps)

        if (
            dgrad is None
            or dparam is None
            or dparam <= self.eps
            or grad_dual <= self.eps
        ):
            return current_grad, current_param

        l1hat = (dgrad / dparam) / grad_dual
        self._append(stats, "l1", l1hat)
        self._append(stats, "lr_pred", self.action_scale / l1hat)
        self._append(stats, "action_eff", lr * l1hat)
        return current_grad, current_param

    def _spectral_reference_scale(
        self, item: ConvergenceItem, x: torch.Tensor
    ) -> float:
        if x.ndim != 2 or x.size(1) <= 0:
            return 1.0
        scale = math.sqrt(x.size(0) / x.size(1))
        if item.name == "tok_emb.weight":
            return max(1.0, scale)
        return scale

    def _sign_atom(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().sign().mul(-1.0 / max(x.size(1), 1))

    def _colnorm_atom(self, item: ConvergenceItem, x: torch.Tensor) -> torch.Tensor:
        transpose = item.group in {"embed", "out"}
        oriented = x.detach().float().mT if transpose else x.detach().float()
        norms = torch.linalg.vector_norm(oriented, dim=0, keepdim=True).clamp_min(
            self.eps
        )
        atom = oriented.div(norms).mul(-math.sqrt(max(oriented.size(0), 1)))
        return atom.mT if transpose else atom

    def _frobenius_atom(self, x: torch.Tensor) -> torch.Tensor:
        work = x.detach().float()
        norm = torch.linalg.vector_norm(work).clamp_min(self.eps)
        return work.mul(-math.sqrt(max(work.numel(), 1)) / norm)

    def _activation_curvature(
        self, item: ConvergenceItem, atom: torch.Tensor
    ) -> float | None:
        input_norms = self.input_norms.get(id(item.param))
        if input_norms is None or atom.ndim != 2:
            return None

        z = atom.detach().float()
        kind = input_norms.get("kind")
        if kind == "embedding":
            counts = input_norms.get("counts")
            if not isinstance(counts, torch.Tensor):
                return None
            counts = counts.to(device=z.device, dtype=torch.float32)
            rows = min(z.size(0), counts.numel())
            if rows <= 0:
                return 0.0
            value = z[:rows].square().sum(dim=1).mul(counts[:rows]).sum()
            return float(value.clamp_min(0.0))

        gram = input_norms.get("gram")
        if not isinstance(gram, torch.Tensor) or gram.size(0) != z.size(1):
            return None
        gram = gram.to(device=z.device, dtype=torch.float32)
        value = (z @ gram).mul(z).sum()
        return float(value.clamp_min(0.0))

    def _oracle_prediction_stats(
        self,
        item: ConvergenceItem,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        spectral: _SpectralOracleResult | None,
    ) -> dict[str, float]:
        if grad.ndim != 2 or momentum.ndim != 2:
            return {}

        atoms = {
            "sign": self._sign_atom(momentum),
            "colnorm": self._colnorm_atom(item, momentum),
            "fro": self._frobenius_atom(momentum),
        }
        if spectral is not None:
            _, polar = spectral
            scale = self._spectral_reference_scale(item, momentum)
            atoms["spectral"] = polar.to(device=momentum.device).mul(-scale)

        grad_f = grad.detach().float()
        mom_f = momentum.detach().float()
        grad_norm = float(torch.linalg.vector_norm(grad_f))
        mom_norm = float(torch.linalg.vector_norm(mom_f))
        out: dict[str, float] = {}

        for name, atom in atoms.items():
            curvature = self._activation_curvature(item, atom)
            atom_norm = float(torch.linalg.vector_norm(atom))
            if curvature is None or curvature <= self.eps or atom_norm <= self.eps:
                continue

            loss_pair = -float((grad_f * atom).sum())
            mom_pair = -float((mom_f * atom).sum())
            loss_pred = max(loss_pair, 0.0) ** 2 / max(curvature, self.eps)
            mom_pred = max(mom_pair, 0.0) ** 2 / max(curvature, self.eps)
            out[f"loss_pair_{name}"] = loss_pair
            out[f"mom_pair_{name}"] = mom_pair
            out[f"loss_pred_{name}"] = loss_pred
            out[f"mom_pred_{name}"] = mom_pred
            out[f"loss_align_{name}"] = loss_pair / (
                grad_norm * atom_norm + self.eps
            )
            out[f"mom_align_{name}"] = mom_pair / (mom_norm * atom_norm + self.eps)

        base = out.get("loss_pred_fro")
        if base is not None and base > self.eps:
            for name in ("spectral", "colnorm", "sign"):
                value = out.get(f"loss_pred_{name}")
                if value is not None:
                    out[f"loss_pred_{name}_over_fro"] = value / base
        mom_base = out.get("mom_pred_fro")
        if mom_base is not None and mom_base > self.eps:
            for name in ("spectral", "colnorm", "sign"):
                value = out.get(f"mom_pred_{name}")
                if value is not None:
                    out[f"mom_pred_{name}_over_fro"] = value / mom_base
        return out

    def _typed_policy_oracle(self, item: ConvergenceItem) -> str | None:
        if item.group == "embed":
            return "colnorm"
        if item.group == "hidden":
            return "spectral"
        if item.group == "out":
            return "sign"
        return None

    def _append_paper_stats(
        self,
        stats: dict[str, list[float]],
        item: ConvergenceItem,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        spectral: _SpectralOracleResult | None,
    ) -> None:
        input_norms = self.input_norms.get(id(item.param))
        if input_norms is None or momentum.ndim != 2 or spectral is None:
            return
        momentum_nuclear, _ = spectral
        momentum_fro = float(torch.linalg.vector_norm(momentum.detach().float()))
        grad_fro = float(torch.linalg.vector_norm(grad.detach().float()))
        if momentum_fro <= self.eps:
            return
        momentum_nr = (momentum_nuclear / momentum_fro) ** 2
        input_fro = float(input_norms["fro"])
        input_op = float(input_norms["op"])
        input_sr = float(input_norms["stable_rank"])
        ratio = momentum_nr / max(input_sr, self.eps)
        pred_fro_bound = momentum_fro * momentum_fro / max(input_op * input_op, self.eps)
        pred_spectral_bound = (
            momentum_nuclear * momentum_nuclear / max(input_fro * input_fro, self.eps)
        )
        grad_mom_cos = float((grad.detach().float() * momentum.detach().float()).sum())
        grad_mom_cos /= max(grad_fro * momentum_fro, self.eps)

        self._append(stats, "momentum_nuclear_rank", momentum_nr)
        self._append(stats, "input_stable_rank", input_sr)
        self._append(stats, "paper_ratio_momentum", ratio)
        self._append(stats, "spec_ratio", ratio)
        self._append(stats, "pred_fro_bound_momentum", pred_fro_bound)
        self._append(stats, "pred_spectral_bound_momentum", pred_spectral_bound)
        self._append(stats, "grad_momentum_cos", grad_mom_cos)

        predictions = self._oracle_prediction_stats(item, grad, momentum, spectral)
        for key, value in predictions.items():
            self._append(stats, key, value)
        typed_policy = self._typed_policy_oracle(item)
        if typed_policy is not None:
            for prefix in (
                "loss_pair",
                "mom_pair",
                "loss_pred",
                "mom_pred",
                "loss_align",
                "mom_align",
            ):
                value = predictions.get(f"{prefix}_{typed_policy}")
                if value is not None:
                    self._append(stats, f"{prefix}_typed_policy", value)
            for prefix in ("loss_pred", "mom_pred"):
                value = predictions.get(f"{prefix}_{typed_policy}_over_fro")
                if value is not None:
                    self._append(stats, f"{prefix}_typed_policy_over_fro", value)

        detail = {
            "name": item.name,
            "group": item.group,
            "shape": list(momentum.shape),
            "paper_matrix": "post_ema_momentum",
            "typed_policy_oracle": typed_policy or "",
            "momentum_nuclear": momentum_nuclear,
            "momentum_fro": momentum_fro,
            "momentum_nuclear_rank": momentum_nr,
            "grad_fro": grad_fro,
            "grad_momentum_cos": grad_mom_cos,
            "input_fro": input_fro,
            "input_op": input_op,
            "input_stable_rank": input_sr,
            "paper_ratio_momentum": ratio,
            "pred_fro_bound_momentum": pred_fro_bound,
            "pred_spectral_bound_momentum": pred_spectral_bound,
        }
        if typed_policy is not None:
            detail["loss_pred_typed_policy_over_fro"] = predictions.get(
                f"loss_pred_{typed_policy}_over_fro", float("nan")
            )
            detail["mom_pred_typed_policy_over_fro"] = predictions.get(
                f"mom_pred_{typed_policy}_over_fro", float("nan")
            )
        detail.update(predictions)
        self.details.append(detail)

    def _append_report_stats(
        self,
        grouped: dict[str, dict[str, list[float]]],
        item: ConvergenceItem,
        grad: torch.Tensor,
        previous: _PrevState,
        current_lrs: dict[str, float],
        streaming_dparam: dict[int, float],
        spectral_gdual: dict[int, float],
        spectral_dgrad: dict[int, float],
        spectral_momentum: dict[int, _SpectralOracleResult],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        key = id(item.param)
        stats = grouped.setdefault(item.group, {})
        current_grad = None
        current_param = None
        momentum = self._effective_momentum(item, grad)
        grad_dual = spectral_gdual.get(key)
        if grad_dual is None:
            current_grad = self._cpu_tensor(grad, current_grad)
            grad_dual = dual_norm(current_grad, item.ulmo, self.eps, item.param)
        spectral = spectral_momentum.get(key)
        if spectral is not None and uses_global_spectral_support(item.ulmo):
            mom_dual = spectral[0] * spectral_ulmo_scale(momentum, item.ulmo)
        else:
            mom_dual = dual_norm(momentum, item.ulmo, self.eps, item.param)

        lr = current_lrs.get(item.group, float("nan"))
        self._append(stats, "gdual", grad_dual)
        self._append(stats, "mdual", mom_dual)
        self._append(stats, "lr", lr)
        current_grad, current_param = self._append_change_stats(
            stats,
            item,
            grad,
            previous,
            grad_dual,
            lr,
            spectral_dgrad,
            streaming_dparam,
            current_grad,
            current_param,
        )
        self._append_paper_stats(stats, item, grad, momentum, spectral)
        return current_grad, current_param

    def _store_previous(
        self,
        item: ConvergenceItem,
        grad: torch.Tensor,
        streamable: bool,
        current_grad: torch.Tensor | None,
        current_param: torch.Tensor | None,
    ) -> None:
        key = id(item.param)
        if self.keep_prev_gpu and streamable:
            self.prev_gpu[key] = (
                grad.detach().float().clone(),
                item.param.detach().float().clone(),
            )
            self.prev.pop(key, None)
            return

        self.prev_gpu.pop(key, None)
        if not self.keep_prev:
            self.prev.pop(key, None)
            return

        current_grad = self._cpu_tensor(grad, current_grad)
        current_param = self._cpu_tensor(item.param, current_param)
        self.prev[key] = (current_grad.clone(), current_param.clone())

    def capture(self, step: int, current_lrs: dict[str, float]) -> str:
        report = self.active
        if not report and not self.keep_prev:
            self.summary = {}
            self.active = False
            return ""
        if not report:
            self.summary = {}

        grouped: dict[str, dict[str, list[float]]] = {}
        records = []
        for item in self.items:
            grad = item.param.grad
            if grad is None:
                continue
            records.append((item, grad, self.prev.get(id(item.param))))

        streaming_dparam = self._streaming_dparam_norms(records) if report else {}
        spectral_gdual = self._spectral_grad_dual_norms(records) if report else {}
        spectral_dgrad = self._spectral_dgrad_dual_norms(records) if report else {}
        spectral_momentum = (
            self._momentum_spectral_oracle_stats(records) if report else {}
        )

        for item, grad, previous in records:
            streamable = self._can_stream_spectral_norm(item, item.param)
            current_grad = None
            current_param = None

            if report:
                current_grad, current_param = self._append_report_stats(
                    grouped,
                    item,
                    grad,
                    previous,
                    current_lrs,
                    streaming_dparam,
                    spectral_gdual,
                    spectral_dgrad,
                    spectral_momentum,
                )

            self._store_previous(item, grad, streamable, current_grad, current_param)
        self.active = False
        return self._format(step, grouped) if report else ""

    def _append(self, stats: dict[str, list[float]], name: str, value: float) -> None:
        if math.isfinite(value):
            stats.setdefault(name, []).append(value)

    def _format(self, step: int, grouped: dict[str, dict[str, list[float]]]) -> str:
        parts = []
        self.summary = {}
        for name, stats in grouped.items():
            group_summary: dict[str, float] = {}
            lr = median(stats.get("lr", []))
            group_summary["lr"] = lr
            fields = [f"lr={lr:.6f}"]
            if stats.get("l1"):
                l1 = median(stats["l1"])
                action_eff = median(stats["action_eff"])
                lr_pred = median(stats["lr_pred"])
                group_summary.update(
                    {
                        "l1": l1,
                        "action_eff": action_eff,
                        "lr_pred": lr_pred,
                    }
                )
                fields.append(f"L1={l1:.2e}")
                fields.append(f"act={action_eff:.2f}")
                fields.append(f"lr*={lr_pred:.2e}")
            gdual = median(stats.get("gdual", []))
            group_summary["gdual"] = gdual
            fields.append(f"g*={gdual:.2e}")
            mdual = median(stats.get("mdual", []))
            group_summary["mdual"] = mdual
            fields.append(f"m*={mdual:.2e}")
            if stats.get("spec_ratio"):
                spec_ratio = median(stats["spec_ratio"])
                group_summary["spec_ratio"] = spec_ratio
                fields.append(f"Rspec={spec_ratio:.2f}")
            if stats.get("momentum_nuclear_rank"):
                momentum_nr = median(stats["momentum_nuclear_rank"])
                input_sr = median(stats.get("input_stable_rank", []))
                paper_ratio = median(stats.get("paper_ratio_momentum", []))
                group_summary.update(
                    {
                        "momentum_nuclear_rank": momentum_nr,
                        "input_stable_rank": input_sr,
                        "paper_ratio_momentum": paper_ratio,
                    }
                )
                fields.append(f"nrM={momentum_nr:.2f}")
                fields.append(f"srA={input_sr:.2f}")
                fields.append(f"nr/sr={paper_ratio:.2f}")
            for stat_name, label in (
                ("loss_pred_typed_policy_over_fro", "policy/F"),
                ("loss_pred_spectral_over_fro", "predS/F"),
                ("loss_pred_colnorm_over_fro", "predC/F"),
                ("loss_pred_sign_over_fro", "predSign/F"),
            ):
                if stats.get(stat_name):
                    value = median(stats[stat_name])
                    group_summary[stat_name] = value
                    fields.append(f"{label}={value:.2f}")
            if stats.get("grad_momentum_cos"):
                grad_mom_cos = median(stats["grad_momentum_cos"])
                group_summary["grad_momentum_cos"] = grad_mom_cos
                fields.append(f"gMcos={grad_mom_cos:.2f}")
            self.summary[name] = group_summary
            parts.append(f"{name}: " + ",".join(fields))
        if not parts:
            return ""
        return f"conv_stats step {step:5d} | " + "; ".join(parts)
