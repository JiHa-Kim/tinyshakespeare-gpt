from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch.optim import Optimizer

__all__ = [
    "StepStatSnapshot",
    "accumulate_step_stats",
    "capture_step_stats",
    "consume_step_stats",
]


@dataclass
class StepStatSnapshot:
    group: str
    items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def capture_step_stats(optimizer: Optimizer) -> list[StepStatSnapshot]:
    snapshots = []
    for group in optimizer.param_groups:
        items = []
        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            items.append(
                (
                    p,
                    p.detach().clone(memory_format=torch.preserve_format),
                    g.detach().clone(memory_format=torch.preserve_format),
                )
            )
        if items:
            snapshots.append(StepStatSnapshot(group.get("name", "group"), items))
    return snapshots


def _stat_add(stats: dict, name: str, value: torch.Tensor) -> None:
    value = value.detach()
    current = stats.get(name)
    stats[name] = value if current is None else current + value


@torch.no_grad()
def accumulate_step_stats(
    accum: dict[str, dict], snapshots: Iterable[StepStatSnapshot]
) -> None:
    for snapshot in snapshots:
        stats = accum.setdefault(
            snapshot.group, {"steps": 0, "params": 0, "numel": 0}
        )
        stats["steps"] += 1
        stats["params"] += len(snapshot.items)

        for p, p_before, grad in snapshot.items:
            grad = grad.float()
            delta = p.detach().float() - p_before.float()
            p32 = p_before.float()
            update_sq = delta.square().sum()
            grad_sq = grad.square().sum()
            param_sq = p32.square().sum()
            descent = -(grad * delta).sum()
            param_grad = (p32 * grad).sum()
            param_update = (p32 * delta).sum()
            stats["numel"] += p.numel()
            _stat_add(stats, "grad_sq", grad_sq)
            _stat_add(stats, "update_sq", update_sq)
            _stat_add(stats, "param_sq", param_sq)
            _stat_add(stats, "descent", descent)
            _stat_add(stats, "param_grad", param_grad)
            _stat_add(stats, "param_update", param_update)
            _stat_add(stats, "grad_abs", grad.abs().sum())
            _stat_add(stats, "update_abs", delta.abs().sum())
            _stat_add(stats, "param_abs", p32.abs().sum())
            _stat_add(stats, "grad_fourth", grad.square().square().sum())
            _stat_add(stats, "update_fourth", delta.square().square().sum())
            _stat_add(stats, "param_fourth", p32.square().square().sum())
            _stat_add(stats, "lr_descent", descent)
            _stat_add(stats, "raw_descent", descent)
            _stat_add(stats, "lr_raw_descent", descent)
            _stat_add(stats, "lr2_update_sq", update_sq)


def consume_step_stats(
    accum: dict[str, dict], eps: float = 1e-12
) -> dict[str, dict[str, float]]:
    out = {}
    for name, stats in list(accum.items()):
        grad_sq = float(stats.get("grad_sq", 0.0))
        update_sq = float(stats.get("update_sq", 0.0))
        param_sq = float(stats.get("param_sq", 0.0))
        descent = float(stats.get("descent", 0.0))
        param_grad = float(stats.get("param_grad", 0.0))
        param_update = float(stats.get("param_update", 0.0))
        lr_descent = float(stats.get("lr_descent", descent))
        raw_descent = float(stats.get("raw_descent", descent))
        lr_raw_descent = float(stats.get("lr_raw_descent", descent))
        lr2_update_sq = float(stats.get("lr2_update_sq", update_sq))
        numel = max(int(stats["numel"]), 1)
        grad_rms = (grad_sq / numel) ** 0.5
        update_rms = (update_sq / numel) ** 0.5
        param_rms = (param_sq / numel) ** 0.5
        grad_abs = float(stats.get("grad_abs", 0.0)) / numel
        update_abs = float(stats.get("update_abs", 0.0)) / numel
        param_abs = float(stats.get("param_abs", 0.0)) / numel
        grad_kurt = numel * float(stats.get("grad_fourth", 0.0)) / (grad_sq**2 + eps)
        update_kurt = (
            numel * float(stats.get("update_fourth", 0.0)) / (update_sq**2 + eps)
        )
        param_kurt = (
            numel * float(stats.get("param_fourth", 0.0)) / (param_sq**2 + eps)
        )
        out[name] = {
            "steps": int(stats["steps"]),
            "params": int(stats["params"]),
            "grad_rms": grad_rms,
            "raw_grad_rms": grad_rms,
            "update_rms": update_rms,
            "param_rms": param_rms,
            "grad_abs_mean": grad_abs,
            "update_abs_mean": update_abs,
            "param_abs_mean": param_abs,
            "grad_kurtosis": grad_kurt,
            "update_kurtosis": update_kurt,
            "param_kurtosis": param_kurt,
            "descent": descent,
            "lr_descent": lr_descent,
            "raw_descent": raw_descent,
            "lr_raw_descent": lr_raw_descent,
            "lr2_update_sq": lr2_update_sq,
            "cos": descent / ((grad_sq * update_sq) ** 0.5 + eps),
            "raw_cos": raw_descent / ((grad_sq * update_sq) ** 0.5 + eps),
            "param_grad_cos": param_grad / ((param_sq * grad_sq) ** 0.5 + eps),
            "param_update_cos": param_update / ((param_sq * update_sq) ** 0.5 + eps),
            "update_grad_rms": update_rms / (grad_rms + eps),
            "grad_param_rms": grad_rms / (param_rms + eps),
            "update_param_rms": update_rms / (param_rms + eps),
            "grad_abs_rms": grad_abs / (grad_rms + eps),
            "update_abs_rms": update_abs / (update_rms + eps),
            "param_abs_rms": param_abs / (param_rms + eps),
        }
    accum.clear()
    return out
