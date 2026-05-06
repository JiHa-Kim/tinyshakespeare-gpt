import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from scionh.compile_env import ensure_compile_env
from scionh.optim.parametrization import (
    resolve_schedule,
    schedule_at_step,
)
from scionh.optim.auxiliary import (
    build_derf_optimizers,
    build_kv_decoder_optimizer,
    configure_derf_training,
    step_derf_optimizers,
    zero_derf_optimizers,
)
from scionh.optim.setup import (
    DEFAULT_HYPERBALL_UPDATE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_STATE_HALF_LIFE,
    GROUP_NAMES,
    OPTIMIZER_NAME,
    apply_scheduled_lr,
    build_optimizer,
    count_increment,
    format_optimizer_schedule,
    optimizer_io_label,
    optimizer_rms_state,
    resolve_hyperball_update,
    rms_state_text,
)
from scionh.models.deepnorm import (
    calibrate_deepnorm_branches,
    deepnorm_calibration_text,
)
from scionh.models.gpt import (
    GPT,
    BatchSource,
)
from scionh.models.inspection import (
    derf_state,
    derf_state_text,
    kv_cache_summary,
    kv_decoder_state,
    kv_decoder_state_text,
    parameter_summary,
    rmsnorm_affine_state,
    rmsnorm_affine_state_text,
)
from scionh.probes.convergence import ConvergenceProbe
from scionh.probes.line import (
    apply_line_scale,
    capture_params,
    capture_rng,
    finish_line_snapshot,
    line_curve_stats,
    line_curve_text,
    line_probe_stats,
    line_probe_text,
    parse_line_scales,
    restore_rng,
)
from scionh.probes.optimizer_stats import (
    accumulate_step_stats,
    capture_step_stats,
    consume_step_stats,
    step_stats_text,
)
from scionh.training.checkpoints import (
    load_checkpoint,
    save_checkpoint,
    save_eval_checkpoint,
)
from scionh.training.runtime import (
    build_model,
    configure_runtime,
    fixed_eval_batches,
    load_dataset,
    resolve_compile_seed,
    resolve_data_seed,
    resolve_eval_seed,
)


def sync_now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def cuda_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "alloc_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "total_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
    }


def cuda_memory_text(device: torch.device) -> str:
    stats = cuda_memory_stats(device)
    if not stats:
        return ""
    return (
        f" | cuda_alloc {stats['alloc_gb']:.2f}G"
        f" | cuda_reserved {stats['reserved_gb']:.2f}G"
        f" | cuda_max_reserved {stats['max_reserved_gb']:.2f}G"
    )


def jsonable(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


class MetricsLogger:
    def __init__(self, path: str, run_name: str = "") -> None:
        self.run_name = run_name
        self.file = None
        if path:
            metrics_path = Path(path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = metrics_path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: str, **values) -> None:
        if self.file is None:
            return
        record = {"event": event, "run_name": self.run_name, **values}
        self.file.write(json.dumps(jsonable(record), separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


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


def maybe_compile(
    model: GPT,
    source: BatchSource,
    args,
    amp_dtype: torch.dtype | None,
    device: torch.device,
):
    if not (args.compile and hasattr(torch, "compile")):
        return model, 0.0
    ensure_compile_env()
    model = torch.compile(model)
    xb, yb = source.seeded_batch("train", resolve_compile_seed(args))
    t0 = sync_now(device)
    model.zero_grad(set_to_none=True)
    with amp_ctx(amp_dtype):
        _, loss = model(xb, yb)
    loss.backward()
    model.zero_grad(set_to_none=True)
    was_training = model.training
    model.eval()
    with torch.no_grad(), amp_ctx(amp_dtype):
        model(xb, yb)
    model.train(was_training)
    return model, sync_now(device) - t0


def resolve_training_schedule(args) -> tuple[int, int, int]:
    warmup_steps = (
        args.warmup_iters
        if args.warmup_iters >= 0
        else round(args.warmup_frac * args.max_iters)
    )
    decay_steps = (
        args.decay_iters
        if args.decay_iters >= 0
        else round(args.decay_frac * args.max_iters)
    )
    return resolve_schedule(args.max_iters, warmup_steps, decay_steps)


def line_probe_active(args, step: int) -> bool:
    return (
        args.track_line_probe
        and args.grad_accum == 1
        and args.line_probe_interval > 0
        and step % args.line_probe_interval == 0
    )


def run_line_probe(
    model,
    step: int,
    batch,
    rng_before,
    loss_before: float | None,
    params_before,
    curve_scales: list[float],
    line_stats: dict[str, dict],
    amp_dtype: torch.dtype | None,
    device: torch.device,
) -> dict:
    if batch is None or rng_before is None or loss_before is None:
        return {}

    rng_after = capture_rng(device)
    curve_losses = []
    curve_values = {}
    if params_before is None:
        restore_rng(rng_before, device)
        with torch.no_grad(), amp_ctx(amp_dtype):
            _, loss_after = model(*batch)
        restore_rng(rng_after, device)
        loss_after_value = float(loss_after.detach())
    else:
        snapshot = finish_line_snapshot(params_before)
        for scale in curve_scales:
            apply_line_scale(snapshot, scale)
            restore_rng(rng_before, device)
            with torch.no_grad(), amp_ctx(amp_dtype):
                _, curve_loss = model(*batch)
            curve_losses.append((scale, float(curve_loss.detach())))
        apply_line_scale(snapshot, 1.0)
        restore_rng(rng_after, device)
        loss_after_value = min(curve_losses, key=lambda item: abs(item[0] - 1.0))[1]
        curve_values = line_curve_stats(curve_losses)
        curve_text = line_curve_text(step, curve_losses)
        if curve_text:
            print(curve_text)

    probe_values = line_probe_stats(loss_before, loss_after_value, line_stats)
    line_text = line_probe_text(step, loss_before, loss_after_value, line_stats)
    if line_text:
        print(line_text)
    return {
        "loss_before": loss_before,
        "loss_after": loss_after_value,
        "probe": probe_values,
        "curve": curve_values,
        "curve_losses": [
            {"scale": scale, "loss": loss} for scale, loss in curve_losses
        ],
        "step_stats": line_stats,
    }


def train(args):
    args.hyperball_update = resolve_hyperball_update(args)
    device, amp_dtype = configure_runtime(args)
    metrics = MetricsLogger(args.metrics_jsonl, args.run_name)
    line_curve_scales = parse_line_scales(args.line_curve_scales)
    if line_curve_scales:
        args.track_line_probe = True

    dataset = load_dataset(args)
    raw_model = build_model(args, dataset, device)
    configure_derf_training(raw_model, args)
    source = BatchSource(
        dataset.train,
        dataset.val,
        args.block_size,
        args.batch_size,
        device,
        train_seed=resolve_data_seed(args),
        val_seed=resolve_eval_seed(args),
    )
    eval_batches = fixed_eval_batches(args, source)
    opt = build_optimizer(raw_model, args, device)
    deepnorm_calibration = {}
    if args.deepnorm_calibrate_branches:
        idx, _targets = source.seeded_batch("train", resolve_data_seed(args))
        deepnorm_calibration = calibrate_deepnorm_branches(raw_model, idx)
        text = deepnorm_calibration_text(deepnorm_calibration)
        if text:
            print(text)
    derf_opts = build_derf_optimizers(raw_model, args)
    kv_decoder_opt = build_kv_decoder_optimizer(raw_model, args)
    conv_probe = (
        ConvergenceProbe(raw_model, opt, args) if args.track_convergence_stats else None
    )
    if conv_probe is not None:
        conv_probe.register_hooks(raw_model)
        if args.compile:
            print("compile_disabled_for_convergence_stats")
            args.compile = False
    model, compile_seconds = maybe_compile(raw_model, source, args, amp_dtype, device)
    if compile_seconds:
        print(f"compile_seconds {compile_seconds:.3f}")

    warmup_steps, stable_steps, decay_steps = resolve_training_schedule(args)
    effective_tokens = count_increment(args)
    first_group = opt.param_groups[0]
    lr_peak = float(first_group.get("lr_peak", first_group["lr"]))
    beta = float(first_group.get("beta", math.nan))
    derf_beta = next(iter(derf_opts.values())).beta if derf_opts else math.nan
    kv_decoder_beta = kv_decoder_opt.beta if kv_decoder_opt is not None else math.nan
    state_half_life = first_group.get("state_half_life", math.nan)
    io_weights = optimizer_io_label(raw_model)
    params_info = parameter_summary(raw_model)
    kv_info = kv_cache_summary(raw_model)
    deepnorm_alpha = raw_model.cfg.resolved_deepnorm_alpha

    group_text = format_optimizer_schedule(opt)
    print(
        "schedule "
        f"warmup_steps={warmup_steps} stable_steps={stable_steps} decay_steps={decay_steps} "
        f"count_increment={effective_tokens} "
        f"lr_peak={lr_peak:.6f} "
        f"state_half_life={state_half_life:.3g} beta={beta:.6f} "
        f"optimizer={OPTIMIZER_NAME} update={args.hyperball_update} "
        f"norm={args.norm_type} derf_lr={args.derf_lr:.6f} "
        f"derf_beta={derf_beta:.6f} derf_shape={args.train_derf_shape} "
        f"kv_cache={kv_info['type']} kv_dim={kv_info['cache_dim']}/"
        f"{kv_info['original_cache_dim']} kv_ratio={kv_info['cache_ratio']:.3f} "
        f"kv_key_rank={kv_info['key_rank']} kv_value_rank={kv_info['value_rank']} "
        f"kv_decoder_lr={args.kv_decoder_lr:.6f} "
        f"kv_decoder_beta={kv_decoder_beta:.6f} "
        f"params={params_info['total']} trainable_params={params_info['trainable']} "
        f"dropout={args.dropout:.3f} "
        f"resid_scale={args.resid_scale:.6f} "
        f"block_type={args.block_type} "
        f"deepnorm_alpha={deepnorm_alpha:.6g} "
        f"deepnorm_branch_scale={args.deepnorm_branch_scale:.6g} "
        f"deepnorm_calibrate={args.deepnorm_calibrate_branches} "
        f"lns={args.lns} "
        f"attn={args.attn_type} "
        f"seed={args.seed} data_seed={resolve_data_seed(args)} "
        f"eval_seed={resolve_eval_seed(args)} fixed_eval={args.fixed_eval_batches} "
        f"hidden_ulmo={args.hidden_ulmo} "
        f"io_weights={io_weights} embed_ulmo={args.embed_ulmo} out_ulmo={args.out_ulmo} "
        f"qkv=fused spi_iteration={args.spi_iteration} "
        f"groups={group_text}"
    )
    metrics.write(
        "config",
        args=vars(args),
        schedule={
            "warmup_steps": warmup_steps,
            "stable_steps": stable_steps,
            "decay_steps": decay_steps,
            "count_increment": effective_tokens,
            "seed": args.seed,
            "data_seed": resolve_data_seed(args),
            "eval_seed": resolve_eval_seed(args),
            "compile_seed": resolve_compile_seed(args),
            "fixed_eval_batches": args.fixed_eval_batches,
        },
        optimizer={
            "name": OPTIMIZER_NAME,
            "lr_peak": lr_peak,
            "state_half_life": state_half_life,
            "beta": beta,
            "derf_lr_peak": args.derf_lr,
            "derf_state_half_life": args.derf_state_half_life,
            "derf_beta": derf_beta,
            "derf_groups": list(derf_opts),
            "kv_decoder_lr_peak": args.kv_decoder_lr,
            "kv_decoder_beta": kv_decoder_beta,
            "update_rule": args.hyperball_update,
            "groups": group_text,
        },
        model={
            "params": params_info,
            "dropout": args.dropout,
            "resid_scale": args.resid_scale,
            "block_type": args.block_type,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_branch_scale": args.deepnorm_branch_scale,
            "deepnorm_calibrate_branches": args.deepnorm_calibrate_branches,
            "deepnorm_calibration": deepnorm_calibration,
            "lns": args.lns,
            "norm_type": args.norm_type,
            "derf_alpha": args.derf_alpha,
            "derf_shift": args.derf_shift,
            "train_derf_shape": args.train_derf_shape,
            "attn_type": args.attn_type,
            "kv_cache": kv_info,
            "hidden_ulmo": args.hidden_ulmo,
            "io_weights": io_weights,
            "embed_ulmo": args.embed_ulmo,
            "out_ulmo": args.out_ulmo,
            "qkv": "fused",
            "spi_iteration": args.spi_iteration,
        },
    )
    if args.track_line_probe and args.grad_accum != 1:
        print("line_probe_disabled_requires_grad_accum_1")
    if line_curve_scales:
        print("line_curve_scales " + ",".join(f"{x:g}" for x in line_curve_scales))

    total_opt_steps = 0
    best_val = float("inf")
    max_val = float("-inf")
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    initial_val = None
    diverged = False
    diverge_reason = ""
    train_start = sync_now(device)
    eval_seconds = 0.0
    step_stat_accum = {}

    for step in range(args.max_iters):
        current_lrs = apply_scheduled_lr(
            opt,
            step,
            args.max_iters,
            warmup_steps,
            decay_steps,
            args.schedule_floor,
        )
        lr = current_lrs.get("hidden", next(iter(current_lrs.values())))
        derf_lr = 0.0
        if derf_opts:
            derf_lr = schedule_at_step(
                step,
                args.max_iters,
                args.derf_lr,
                args.schedule_floor * args.derf_lr,
                warmup_steps,
                decay_steps,
            )
            for derf_group_opt in derf_opts.values():
                derf_group_opt.lr = derf_lr
        kv_decoder_lr = 0.0
        if kv_decoder_opt is not None:
            kv_decoder_lr = schedule_at_step(
                step,
                args.max_iters,
                kv_decoder_opt.lr_peak,
                args.schedule_floor * kv_decoder_opt.lr_peak,
                warmup_steps,
                decay_steps,
            )
            kv_decoder_opt.lr = kv_decoder_lr

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            eval_start = sync_now(device)
            train_loss = float(last_train_loss)
            val_loss, logit_stats = estimate_val_metrics(
                model,
                source,
                args.eval_iters,
                amp_dtype,
                args.track_logit_stats,
                eval_batches["val"] if eval_batches is not None else None,
            )
            last_val_loss = val_loss
            opt_stats = (
                consume_step_stats(step_stat_accum) if args.track_step_stats else {}
            )
            weight_rms = optimizer_rms_state(opt)
            derf_stats = derf_state(raw_model)
            norm_affine_stats = rmsnorm_affine_state(raw_model)
            kv_decoder_stats = kv_decoder_state(raw_model)

            if not math.isfinite(val_loss):
                diverged, diverge_reason = True, "nonfinite_eval_loss"
            else:
                prev_best = best_val
                if initial_val is None:
                    initial_val = val_loss
                best_val = min(best_val, val_loss)
                max_val = max(max_val, val_loss)
                if step > 0 and val_loss > initial_val * args.diverge_mult:
                    diverged = True
                    diverge_reason = (
                        f"val_loss_exceeded_{args.diverge_mult:.2f}x_initial"
                    )
                if not args.no_save and val_loss < prev_best:
                    save_checkpoint(Path(args.out_path), raw_model, dataset)
                if not args.no_save:
                    save_eval_checkpoint(
                        Path(args.out_path), step, val_loss, raw_model, dataset, args
                    )
                    if step == args.max_iters - 1:
                        path = Path(args.out_path)
                        save_checkpoint(
                            path.with_name(f"{path.stem}_final{path.suffix}"),
                            raw_model,
                            dataset,
                        )

            now = sync_now(device)
            eval_seconds += now - eval_start
            elapsed = max(now - train_start, 1e-9)
            train_elapsed = max(elapsed - eval_seconds, 1e-9)
            total_tokens = total_opt_steps * effective_tokens
            mem_text = cuda_memory_text(device)
            opt_text = step_stats_text(opt_stats)
            rms_text = rms_state_text(weight_rms)
            derf_text = derf_state_text(derf_stats)
            norm_affine_text = rmsnorm_affine_state_text(norm_affine_stats)
            kv_decoder_text = kv_decoder_state_text(kv_decoder_stats)
            logit_text = (
                " | logits "
                f"std={logit_stats['logit_std']:.3f},"
                f"H={logit_stats['softmax_entropy']:.3f},"
                f"pmax={logit_stats['softmax_max_prob']:.3f}"
                if logit_stats
                else ""
            )
            print(
                f"step {step:5d} | lr {lr:.6f} beta {beta:.6f} "
                f"derf_lr {derf_lr:.6f} kv_decoder_lr {kv_decoder_lr:.6f} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"best_val {best_val:.4f} | train_seconds {elapsed:.3f} | "
                f"tok/s {total_tokens / elapsed:.0f} "
                f"train_tok/s {total_tokens / train_elapsed:.0f}"
                f"{mem_text}{logit_text}{rms_text}{derf_text}{norm_affine_text}"
                f"{kv_decoder_text}{opt_text}"
            )
            memory = cuda_memory_stats(device)
            metrics.write(
                "eval",
                step=step,
                total_opt_steps=total_opt_steps,
                lr=lr,
                derf_lr=derf_lr,
                kv_decoder_lr=kv_decoder_lr,
                beta=beta,
                lrs=current_lrs,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val=best_val,
                max_val=max_val,
                train_seconds=elapsed,
                train_compute_seconds=train_elapsed,
                eval_seconds=eval_seconds,
                tokens_per_second=total_tokens / elapsed,
                train_tokens_per_second=total_tokens / train_elapsed,
                cuda_memory=memory,
                logit_stats=logit_stats,
                weight_rms=weight_rms,
                derf=derf_stats,
                norm_affine=norm_affine_stats,
                kv_decoder=kv_decoder_stats,
                step_stats=opt_stats,
            )
            if (
                args.max_cuda_reserved_gb > 0
                and memory
                and memory["reserved_gb"] > args.max_cuda_reserved_gb
            ):
                diverged = True
                diverge_reason = (
                    f"cuda_reserved_{memory['reserved_gb']:.2f}G_exceeded_"
                    f"{args.max_cuda_reserved_gb:.2f}G"
                )
        if diverged:
            print(f"diverged {diverge_reason}")
            break

        line_active = line_probe_active(args, step)
        line_batch = None
        line_rng_before = None
        line_loss_before = None
        opt.zero_grad(set_to_none=True)
        if derf_opts:
            zero_derf_optimizers(derf_opts, set_to_none=True)
        if kv_decoder_opt is not None:
            kv_decoder_opt.zero_grad(set_to_none=True)
        if conv_probe is not None:
            conv_probe.start_step(step)
        train_loss = None
        for micro_step in range(args.grad_accum):
            batch = source.get("train")
            if line_active and micro_step == 0:
                line_batch = batch
                line_rng_before = capture_rng(device)
            with amp_ctx(amp_dtype):
                _, loss = model(*batch)
                loss = loss / args.grad_accum
            loss_value = loss.detach()
            if line_active and micro_step == 0:
                line_loss_before = float(loss_value)
            train_loss = loss_value if train_loss is None else train_loss + loss_value
            loss.backward()
        if diverged:
            print(f"diverged {diverge_reason}")
            break
        last_train_loss = train_loss if train_loss is not None else float("nan")

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        if conv_probe is not None:
            conv_text = conv_probe.capture(step, current_lrs)
            if conv_text:
                print(conv_text)
                metrics.write(
                    "convergence",
                    step=step,
                    total_opt_steps=total_opt_steps,
                    lrs=current_lrs,
                    groups=conv_probe.summary,
                )
        line_params_before = (
            capture_params(raw_model.parameters())
            if line_active and line_curve_scales
            else None
        )
        stat_snapshot = (
            capture_step_stats(opt) if args.track_step_stats or line_active else None
        )
        opt.step()
        if derf_opts:
            step_derf_optimizers(derf_opts)
        if kv_decoder_opt is not None:
            kv_decoder_opt.step()
        line_stats = {}
        if stat_snapshot is not None:
            if args.track_step_stats:
                accumulate_step_stats(step_stat_accum, stat_snapshot)
            if line_active:
                line_stat_accum = {}
                accumulate_step_stats(line_stat_accum, stat_snapshot)
                line_stats = consume_step_stats(line_stat_accum)
        total_opt_steps = step + 1
        if line_active:
            line_record = run_line_probe(
                model,
                step,
                line_batch,
                line_rng_before,
                line_loss_before,
                line_params_before,
                line_curve_scales,
                line_stats,
                amp_dtype,
                device,
            )
            if line_record:
                metrics.write(
                    "line_probe",
                    step=step,
                    total_opt_steps=total_opt_steps,
                    **line_record,
                )

    if not (args.skip_sample or diverged):
        prompt = args.prompt or "\n"
        x = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
        texts = generate_texts(
            raw_model,
            x,
            dataset.decode,
            args.sample_count,
            args.sample_tokens,
            args.temperature,
            args.top_k,
        )
        if not write_sample_report(args, texts):
            print_samples(texts)

    for group in opt.param_groups:
        stats = getattr(group.get("ulmo"), "stats", None)
        if stats:
            print(
                f"{group.get('name', 'group')}_ulmo_stats "
                + " ".join(f"{k}={v}" for k, v in stats.items())
            )

    last_train_loss = float(last_train_loss)
    result = {
        "best_val": best_val,
        "final_train": last_train_loss,
        "final_val": last_val_loss,
        "compile_seconds": compile_seconds,
        "initial_val": float("nan") if initial_val is None else initial_val,
        "max_val": max_val,
        "diverged": diverged,
        "diverge_reason": diverge_reason,
        "warmup_steps": warmup_steps,
        "stable_steps": stable_steps,
        "decay_steps": decay_steps,
    }
    metrics.write("final", **result)
    metrics.close()
    return result


@torch.inference_mode()
def sample(args):
    device, _ = configure_runtime(args)
    model, stoi, itos = load_checkpoint(Path(args.out_path), device)
    prompt = args.prompt or "\n"
    bad = [c for c in prompt if c not in stoi]
    if bad:
        raise ValueError(f"prompt contains unseen chars: {bad}")
    x = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    texts = generate_texts(
        model,
        x,
        lambda ids: "".join(itos[int(i)] for i in ids),
        args.sample_count,
        args.sample_tokens,
        args.temperature,
        args.top_k,
    )

    if write_sample_report(args, texts):
        return

    print_samples(texts)


def generate_texts(
    model,
    x: torch.Tensor,
    decode,
    sample_count: int,
    sample_tokens: int,
    temperature: float,
    top_k: int,
) -> list[str]:
    texts = []
    for _ in range(sample_count):
        y = model.generate(
            x,
            max_new_tokens=sample_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        texts.append(decode(y[0].tolist()))
    return texts


def write_sample_report(args, texts: list[str]) -> bool:
    if not args.sample_out:
        return False
    path = Path(args.sample_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sample_report(args, texts), encoding="utf-8")
    print(f"wrote_samples {path}")
    return True


def print_samples(texts: list[str]) -> None:
    for i, text in enumerate(texts, start=1):
        if len(texts) > 1:
            print(f"\n--- sample {i} ---\n")
        elif i == 1:
            print("\n--- sample ---\n")
        print(text)


def sample_report(args, texts: list[str]) -> str:
    prompt = args.prompt or "\\n"
    lines = [
        "# Sample Report",
        "",
        f"- checkpoint: `{args.out_path}`",
        f"- seed: `{args.seed}`",
        f"- prompt: `{prompt}`",
        f"- sample_tokens: `{args.sample_tokens}`",
        f"- temperature: `{args.temperature}`",
        f"- top_k: `{args.top_k}`",
        f"- sample_count: `{len(texts)}`",
        "",
    ]
    for i, text in enumerate(texts, start=1):
        lines.extend([f"## Sample {i}", "", "```text", text, "```", ""])
    return "\n".join(lines)


@torch.inference_mode()
def evaluate(args):
    device, amp_dtype = configure_runtime(args)
    dataset = load_dataset(args)
    model, _, _ = load_checkpoint(Path(args.out_path), device)
    source = BatchSource(
        dataset.train,
        dataset.val,
        model.cfg.block_size,
        args.batch_size,
        device,
        train_seed=resolve_data_seed(args),
        val_seed=resolve_eval_seed(args),
    )
    eval_batches = fixed_eval_batches(args, source)
    losses = estimate_loss(model, source, args.eval_iters, amp_dtype, fixed_batches=eval_batches)
    print(
        f"eval_iters {args.eval_iters} | batch_size {args.batch_size} | "
        f"fixed_eval {args.fixed_eval_batches} | eval_seed {resolve_eval_seed(args)} | "
        f"train {losses['train']:.4f} | val {losses['val']:.4f}"
        f"{cuda_memory_text(device)}"
    )


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "sample", "eval"], default="train")
    p.add_argument("--data-path", default="data/tiny_shakespeare.txt")
    p.add_argument("--out-path", default="out/hyperball_shakespeare.pt")
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="seed for the independent training-batch RNG; defaults to --seed",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="seed for fixed evaluation batches; defaults to --seed + 1",
    )
    p.add_argument(
        "--compile-seed",
        type=int,
        default=None,
        help="seed for the compile warmup batch; defaults to --seed + 2",
    )
    p.add_argument(
        "--fixed-eval-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse the same sampled train/val eval batches at every evaluation",
    )
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="request deterministic kernels and disable TF32/benchmark autotuning",
    )
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64, help="microbatch size")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument(
        "--resid-scale",
        type=float,
        default=1.0,
        help="Pre-LN residual branch multiplier; DeepNorm uses its own residual alpha",
    )
    p.add_argument(
        "--block-type",
        choices=["preln", "deepnorm"],
        default="preln",
        help="Transformer block topology",
    )
    p.add_argument(
        "--deepnorm-alpha",
        type=float,
        default=0.0,
        help="DeepNorm residual multiplier; <=0 uses decoder-only default (2*n_layer)^(1/4)",
    )
    p.add_argument(
        "--deepnorm-branch-scale",
        type=float,
        default=1.0,
        help="fixed scalar multiplier on DeepNorm attention/MLP residual branches",
    )
    p.add_argument(
        "--deepnorm-calibrate-branches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "set fixed DeepNorm branch scales at init so branch/(alpha*x) "
            "matches 1/sqrt(2*n_layer)"
        ),
    )
    p.add_argument(
        "--lns",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply LayerNorm Scaling in Pre-LN blocks by scaling norm outputs by 1/sqrt(layer)",
    )
    p.add_argument(
        "--kv-cache",
        choices=["full", "equivariant-lowrank"],
        default="full",
        help="KV-cache architecture used by attention",
    )
    p.add_argument(
        "--kv-key-rank",
        type=int,
        default=3,
        help="complex head-mixing rank per RoPE frequency for low-rank KV",
    )
    p.add_argument(
        "--kv-value-rank",
        type=int,
        default=192,
        help="shared real value-cache rank for low-rank KV",
    )
    p.add_argument(
        "--kv-decoder-lr",
        type=float,
        default=0.001,
        help="peak normalized-SGD learning rate for low-rank KV decoder tensors",
    )
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument(
        "--attn-type",
        choices=["softmax", "linear", "erf"],
        default="softmax",
        help=(
            "attention kernel; linear is a normalized ELU+1 feature-kernel "
            "reference, erf uses normalized 1+erf(score) weights"
        ),
    )
    p.add_argument(
        "--norm-type",
        choices=["rmsnorm", "rmsnorm-affine", "derf"],
        default="rmsnorm",
        help="activation transform used at pre-attn, pre-MLP, and final norm sites",
    )
    p.add_argument(
        "--derf-alpha",
        type=float,
        default=0.5,
        help="Derf input scale alpha initialization",
    )
    p.add_argument(
        "--derf-shift",
        type=float,
        default=0.0,
        help="Derf horizontal shift initialization",
    )
    p.add_argument(
        "--derf-lr",
        type=float,
        default=0.001,
        help="peak normalized-SGD learning rate for Derf shape and small norm affine groups",
    )
    p.add_argument(
        "--derf-state-half-life",
        type=float,
        default=DEFAULT_STATE_HALF_LIFE,
        help="momentum half-life for Derf normalized-SGD updates",
    )
    p.add_argument(
        "--train-derf-shape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train Derf alpha/shift; gamma/beta remain trainable when Derf is active",
    )
    p.add_argument(
        "--tie-weights",
        action="store_true",
        help="share input embedding and output head weights",
    )

    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--diverge-mult", type=float, default=2.0)

    p.add_argument(
        "--warmup-iters", type=int, default=100, help="if >=0, overrides warmup-frac"
    )
    p.add_argument("--warmup-frac", type=float, default=0.0)
    p.add_argument(
        "--decay-iters", type=int, default=-1, help="if >=0, overrides decay-frac"
    )
    p.add_argument("--decay-frac", type=float, default=0.15)

    p.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=(
            "optimizer step size for all groups; retract uses it before "
            "retraction, slerp uses it as an angular step"
        ),
    )
    for group in GROUP_NAMES:
        p.add_argument(f"--lr-{group}", type=float, default=None)
    p.add_argument(
        "--state-half-life",
        type=float,
        default=DEFAULT_STATE_HALF_LIFE,
        help="momentum-state retention half-life in processed tokens",
    )
    for group in GROUP_NAMES:
        p.add_argument(f"--state-half-life-{group}", type=float, default=None)
    p.add_argument(
        "--schedule-floor",
        type=float,
        default=0.0,
        help="WSD schedule floor for the learning-rate ratio (0 = no movement at decay end)",
    )
    p.add_argument(
        "--hyperball-update",
        choices=["slerp", "retract"],
        default=DEFAULT_HYPERBALL_UPDATE,
        help="fixed-radius update rule; slerp is the tangent geodesic variant",
    )
    p.add_argument(
        "--target-rms",
        dest="target_rms",
        type=float,
        default=None,
        help=(
            "initialization RMS target for all optimizer groups; "
            "defaults are embed=0.70, hidden=0.051, out=0.022"
        ),
    )
    for group in GROUP_NAMES:
        p.add_argument(f"--target-rms-{group}", type=float, default=None)
    p.add_argument(
        "--out-rms-rule",
        choices=["fixed", "fan-in"],
        default="fixed",
        help=(
            "default output-head radius when --target-rms-out is omitted; "
            "fan-in uses 1/sqrt(d_model)"
        ),
    )
    p.add_argument(
        "--hidden-ulmo",
        choices=["streaming-svd", "gram-ns"],
        default="gram-ns",
        help="hidden-matrix ULMO",
    )
    p.add_argument(
        "--embed-ulmo",
        choices=["colnorm", "sign", "rownorm"],
        default="colnorm",
        help="embedding-table ULMO; tied weights force Sign",
    )
    p.add_argument(
        "--out-ulmo",
        choices=["sign", "colnorm", "rownorm"],
        default="sign",
        help="output-head ULMO",
    )
    p.add_argument("--pe-steps", type=int, default=5, help="Gram-NS coefficient steps")
    p.add_argument("--spi-steps", type=int, default=1)
    p.add_argument("--spi-ridge", type=float, default=1e-3)
    p.add_argument(
        "--spi-iteration",
        choices=["scqr2", "norm-power"],
        default="norm-power",
        help="streaming-SVD subspace iteration path",
    )
    p.add_argument("--spi-refresh-interval", type=int, default=100)
    p.add_argument("--spi-refresh-threshold", type=float, default=0.10)

    p.add_argument("--prompt", default="To be, or not to be")
    p.add_argument("--sample-tokens", type=int, default=400)
    p.add_argument("--sample-count", type=int, default=1)
    p.add_argument("--sample-out", default="")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--skip-sample", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument(
        "--save-interval",
        type=int,
        default=400,
        help="save eval checkpoints every N steps in addition to best/final",
    )
    p.add_argument(
        "--max-cuda-reserved-gb",
        type=float,
        default=0.0,
        help="abort if process CUDA reserved memory exceeds this limit",
    )
    p.add_argument(
        "--track-step-stats",
        action="store_true",
        help="accumulate optimizer group stats and print them on eval lines",
    )
    p.add_argument(
        "--track-logit-stats",
        action="store_true",
        help="log cheap validation-batch softmax/logit statistics on eval lines",
    )
    p.add_argument(
        "--metrics-jsonl",
        default="",
        help="append structured config/eval/convergence/final records to this JSONL path",
    )
    p.add_argument(
        "--run-name",
        default="",
        help="optional run label included in structured metrics records",
    )
    p.add_argument(
        "--track-convergence-stats",
        action="store_true",
        help="probe smoothness and spectral-ratio stats during training",
    )
    p.add_argument(
        "--track-line-probe",
        action="store_true",
        help="estimate same-batch learning-rate aggressiveness with one extra forward",
    )
    p.add_argument(
        "--line-probe-interval",
        type=int,
        default=100,
        help="optimizer-step interval for same-batch line probes",
    )
    p.add_argument(
        "--line-curve-scales",
        default="",
        help="comma-separated update multipliers for expensive same-batch line curves",
    )
    p.add_argument(
        "--convergence-interval",
        type=int,
        default=50,
        help="optimizer-step interval for convergence probes",
    )
    p.add_argument(
        "--convergence-action-scale",
        dest="convergence_action_scale",
        type=float,
        default=0.5,
        help="target normalized action scale for L1-derived learning-rate reports",
    )
    p.add_argument(
        "--convergence-probe",
        choices=["representative", "all"],
        default="representative",
        help="which parameters to include in convergence probes",
    )
    p.add_argument(
        "--convergence-support-steps",
        type=int,
        default=7,
        help="Gram-NS polar-support steps for spectral dual stats",
    )

    return p


def main():
    args = make_parser().parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "sample":
        sample(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
