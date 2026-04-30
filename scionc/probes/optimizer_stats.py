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
            descent = -(grad * delta).sum()
            stats["numel"] += p.numel()
            _stat_add(stats, "grad_sq", grad.square().sum())
            _stat_add(stats, "update_sq", update_sq)
            _stat_add(stats, "param_sq", p32.square().sum())
            _stat_add(stats, "descent", descent)
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
        lr_descent = float(stats.get("lr_descent", descent))
        raw_descent = float(stats.get("raw_descent", descent))
        lr_raw_descent = float(stats.get("lr_raw_descent", descent))
        lr2_update_sq = float(stats.get("lr2_update_sq", update_sq))
        numel = max(int(stats["numel"]), 1)
        out[name] = {
            "steps": int(stats["steps"]),
            "params": int(stats["params"]),
            "grad_rms": (grad_sq / numel) ** 0.5,
            "raw_grad_rms": (grad_sq / numel) ** 0.5,
            "update_rms": (update_sq / numel) ** 0.5,
            "param_rms": (param_sq / numel) ** 0.5,
            "descent": descent,
            "lr_descent": lr_descent,
            "raw_descent": raw_descent,
            "lr_raw_descent": lr_raw_descent,
            "lr2_update_sq": lr2_update_sq,
            "cos": descent / ((grad_sq * update_sq) ** 0.5 + eps),
            "raw_cos": raw_descent / ((grad_sq * update_sq) ** 0.5 + eps),
        }
    accum.clear()
    return out
