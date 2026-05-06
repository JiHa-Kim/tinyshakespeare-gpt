import math
from contextlib import nullcontext

import torch

from scionh.models.gpt import BatchSource, GPT


def amp_ctx(amp_dtype: torch.dtype | None):
    return (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )


@torch.inference_mode()
def estimate_loss(
    model: GPT,
    source: BatchSource,
    eval_iters: int,
    amp_dtype: torch.dtype | None,
    splits=("train", "val"),
    fixed_batches: dict[str, tuple[tuple[torch.Tensor, torch.Tensor], ...]]
    | None = None,
):
    was_training = model.training
    model.eval()
    out = {}
    with amp_ctx(amp_dtype):
        for split in splits:
            total = 0.0
            batches = fixed_batches.get(split) if fixed_batches else None
            if batches is None:
                batches = tuple(source.get(split) for _ in range(eval_iters))
            for xb, yb in batches:
                _, loss = model(xb, yb)
                total += float(loss)
            out[split] = total / len(batches)
    model.train(was_training)
    return out


def update_logit_stats(
    acc: dict[str, float], logits: torch.Tensor, targets: torch.Tensor
) -> None:
    values = logits.detach().float()
    token_count = values.numel() // values.size(-1)
    centered = values - values.mean(dim=-1, keepdim=True)
    log_probs = torch.log_softmax(values, dim=-1)
    probs = log_probs.exp()
    top2 = torch.topk(values, k=min(2, values.size(-1)), dim=-1).values
    top_margin = (
        top2[..., 0] - top2[..., 1]
        if top2.size(-1) > 1
        else torch.zeros_like(top2[..., 0])
    )
    target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    acc["tokens"] += float(token_count)
    acc["logit_var"] += float(centered.square().mean(dim=-1).sum())
    acc["logit_margin"] += float(top_margin.sum())
    acc["softmax_entropy"] += float((-(probs * log_probs).sum(dim=-1)).sum())
    acc["softmax_max_prob"] += float(probs.max(dim=-1).values.sum())
    acc["target_prob"] += float(target_probs.sum())


def finalize_logit_stats(acc: dict[str, float]) -> dict[str, float]:
    tokens = max(acc.get("tokens", 0.0), 1.0)
    return {
        "logit_std": math.sqrt(max(0.0, acc["logit_var"] / tokens)),
        "logit_margin": acc["logit_margin"] / tokens,
        "softmax_entropy": acc["softmax_entropy"] / tokens,
        "softmax_max_prob": acc["softmax_max_prob"] / tokens,
        "target_prob": acc["target_prob"] / tokens,
    }


@torch.inference_mode()
def estimate_val_metrics(
    model: GPT,
    source: BatchSource,
    eval_iters: int,
    amp_dtype: torch.dtype | None,
    collect_logit_stats: bool = False,
    fixed_batches: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
) -> tuple[float, dict[str, float]]:
    was_training = model.training
    model.eval()
    total = 0.0
    logit_acc = {
        "tokens": 0.0,
        "logit_var": 0.0,
        "logit_margin": 0.0,
        "softmax_entropy": 0.0,
        "softmax_max_prob": 0.0,
        "target_prob": 0.0,
    }
    batches = fixed_batches
    if batches is None:
        batches = tuple(source.get("val") for _ in range(eval_iters))
    with amp_ctx(amp_dtype):
        for xb, yb in batches:
            logits, loss = model(xb, yb)
            total += float(loss)
            if collect_logit_stats:
                update_logit_stats(logit_acc, logits, yb)
    model.train(was_training)
    stats = finalize_logit_stats(logit_acc) if collect_logit_stats else {}
    return total / len(batches), stats
