import math
from dataclasses import dataclass

import torch

from scionc.ulmos.core import gram_newton_schulz_polar
from scionc.models.gpt import GPT


@dataclass
class ConvergenceItem:
    name: str
    group: str
    opt_group: dict
    param: torch.Tensor
    ulmo: object


_SpectralPowerKey = tuple[int, str]
_SpectralPowerGroupKey = tuple[torch.device, tuple[int, int], int]
_SpectralPowerGroupItem = tuple[_SpectralPowerKey, torch.Tensor, bool]
_PrevState = tuple[torch.Tensor, torch.Tensor] | None
_ConvergenceRecord = tuple[ConvergenceItem, torch.Tensor, _PrevState]
_SpectralDualRequest = tuple[int, ConvergenceItem, torch.Tensor]
_SpectralDualGroupKey = tuple[torch.device, tuple[int, int]]
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
        return x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(
            self.eps
        )


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


def dual_norm(
    x: torch.Tensor, ulmo, eps: float = 1e-12, param: torch.Tensor | None = None
) -> float:
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
    polar = gram_newton_schulz_polar(batch, steps, eps, work_dtype, 1.05).float()
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
    x = x.float()
    geometry = ulmo.geometry
    if geometry.is_spectral:
        return spectral_norm_power(x, eps) / spectral_ulmo_scale(x, ulmo)
    return float(geometry.primal_norm(x, eps))


def stable_rank_from_input(x: torch.Tensor, eps: float = 1e-12) -> float:
    flat = x.detach().reshape(-1, x.size(-1)).float()
    gram = (flat.mT @ flat).float()
    fro_sq = gram.diagonal().sum().clamp_min(eps)
    op_sq = torch.linalg.eigvalsh(gram).amax().clamp_min(eps)
    return float(fro_sq / op_sq)


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
        self.input_sr: dict[int, float] = {}
        self.summary: dict[str, dict[str, float]] = {}
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
            group, opt_group, ulmo = groups[id(p)]
            items.append(ConvergenceItem(name, group, opt_group, p, ulmo))
        return items

    def _probe_names(self, model: GPT) -> set[str]:
        names = {"tok_emb.weight", "lm_head.weight"}
        block_ids = sorted({0, len(model.blocks) // 2, len(model.blocks) - 1})
        suffixes = (
            "attn.q.weight",
            "attn.k.weight",
            "attn.v.weight",
            "attn.proj.weight",
            "mlp.gate.weight",
            "mlp.up.weight",
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
            self.input_sr.clear()

    def register_hooks(self, model: GPT):
        selected = {id(item.param) for item in self.items}
        handles = []
        for module in model.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = getattr(module, "weight", None)
            if weight is None or id(weight) not in selected:
                continue
            handles.append(module.register_forward_pre_hook(self._make_hook(weight)))
        return handles

    def _make_hook(self, weight: torch.Tensor):
        def hook(module, inputs):
            if not (self.active and module.training and torch.is_grad_enabled()):
                return
            with torch.no_grad():
                self.input_sr[id(weight)] = stable_rank_from_input(inputs[0], self.eps)

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
            prev_grad = self._previous_tensor(item, previous, 0, grad.shape, grad.device)
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

    def _can_stream_spectral_norm(self, item: ConvergenceItem, x: torch.Tensor) -> bool:
        return (
            is_spectral_ulmo(item.ulmo)
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
        q: float,
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

        if dgrad is None or dparam is None or dparam <= self.eps or grad_dual <= self.eps:
            return current_grad, current_param

        l1hat = (dgrad / dparam) / grad_dual
        self._append(stats, "l1", l1hat)
        self._append(stats, "eta_pred", self.action_scale / l1hat)
        self._append(stats, "action_eff", q * l1hat)
        return current_grad, current_param

    def _append_spec_ratio(
        self,
        stats: dict[str, list[float]],
        item: ConvergenceItem,
        grad: torch.Tensor,
        grad_dual: float,
    ) -> None:
        input_sr = self.input_sr.get(id(item.param))
        if input_sr is None or grad.ndim != 2 or not is_spectral_ulmo(item.ulmo):
            return

        scale = max(spectral_ulmo_scale(grad, item.ulmo), self.eps)
        fro_sq = float(grad.detach().float().square().sum().clamp_min(self.eps))
        nuc = grad_dual / scale
        self._append(stats, "spec_ratio", (nuc * nuc / fro_sq) / max(input_sr, self.eps))

    def _append_report_stats(
        self,
        grouped: dict[str, dict[str, list[float]]],
        item: ConvergenceItem,
        grad: torch.Tensor,
        previous: _PrevState,
        current_qs: dict[str, float],
        streaming_dparam: dict[int, float],
        spectral_gdual: dict[int, float],
        spectral_dgrad: dict[int, float],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        key = id(item.param)
        stats = grouped.setdefault(item.group, {})
        current_grad = None
        current_param = None
        grad_dual = spectral_gdual.get(key)
        if grad_dual is None:
            current_grad = self._cpu_tensor(grad, current_grad)
            grad_dual = dual_norm(current_grad, item.ulmo, self.eps, item.param)

        q = current_qs.get(item.group, float("nan"))
        self._append(stats, "gdual", grad_dual)
        self._append(stats, "q", q)
        current_grad, current_param = self._append_change_stats(
            stats,
            item,
            grad,
            previous,
            grad_dual,
            q,
            spectral_dgrad,
            streaming_dparam,
            current_grad,
            current_param,
        )
        self._append_spec_ratio(stats, item, grad, grad_dual)
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

    def capture(self, step: int, current_qs: dict[str, float]) -> str:
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
                    current_qs,
                    streaming_dparam,
                    spectral_gdual,
                    spectral_dgrad,
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
            q = median(stats.get("q", []))
            group_summary["q"] = q
            fields = [f"q={q:.6f}"]
            if stats.get("l1"):
                l1 = median(stats["l1"])
                action_eff = median(stats["action_eff"])
                eta_pred = median(stats["eta_pred"])
                group_summary.update(
                    {
                        "l1": l1,
                        "action_eff": action_eff,
                        "eta_pred": eta_pred,
                    }
                )
                fields.append(f"L1={l1:.2e}")
                fields.append(f"act={action_eff:.2f}")
                fields.append(f"eta*={eta_pred:.2e}")
            gdual = median(stats.get("gdual", []))
            group_summary["gdual"] = gdual
            fields.append(f"g*={gdual:.2e}")
            if stats.get("spec_ratio"):
                spec_ratio = median(stats["spec_ratio"])
                group_summary["spec_ratio"] = spec_ratio
                fields.append(f"Rspec={spec_ratio:.2f}")
            self.summary[name] = group_summary
            parts.append(f"{name}: " + ",".join(fields))
        if not parts:
            return ""
        return f"conv_stats step {step:5d} | " + "; ".join(parts)
