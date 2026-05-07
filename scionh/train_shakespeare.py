import math
from dataclasses import dataclass, field
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
from scionh.training.cli import make_parser
from scionh.training.device import (
    cuda_memory_stats,
    cuda_memory_text,
    sync_now,
)
from scionh.training.evaluation import (
    amp_ctx,
    estimate_loss,
    estimate_val_metrics,
)
from scionh.training.metrics import MetricsLogger
from scionh.training.runtime import (
    build_model,
    configure_runtime,
    fixed_eval_batches,
    load_dataset,
    resolve_compile_seed,
    resolve_data_seed,
    resolve_eval_seed,
)


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
    compile_kwargs = {}
    if args.compile_mode != "default":
        compile_kwargs["mode"] = args.compile_mode
    model = torch.compile(model, **compile_kwargs)
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


@dataclass
class TrainingComponents:
    metrics: MetricsLogger
    dataset: object
    raw_model: GPT
    source: BatchSource
    eval_batches: dict | None
    opt: object
    derf_opts: dict
    kv_decoder_opt: object | None
    conv_probe: ConvergenceProbe | None
    model: object
    compile_seconds: float
    deepnorm_calibration: dict


@dataclass
class TrainingSchedule:
    warmup_steps: int
    stable_steps: int
    decay_steps: int
    effective_tokens: int


@dataclass
class TrainingMetadata:
    lr_peak: float
    beta: float
    derf_beta: float
    kv_decoder_beta: float
    state_half_life: float
    io_weights: str
    params_info: dict
    kv_info: dict
    deepnorm_alpha: float
    group_text: str


@dataclass
class TrainingProgress:
    train_start: float
    total_opt_steps: int = 0
    best_val: float = field(default_factory=lambda: float("inf"))
    max_val: float = field(default_factory=lambda: float("-inf"))
    last_train_loss: float = field(default_factory=lambda: float("nan"))
    last_val_loss: float = field(default_factory=lambda: float("nan"))
    initial_val: float | None = None
    diverged: bool = False
    diverge_reason: str = ""
    eval_seconds: float = 0.0
    step_stat_accum: dict = field(default_factory=dict)

    def mark_diverged(self, reason: str) -> None:
        self.diverged = True
        self.diverge_reason = reason


@dataclass
class StepRates:
    all_lrs: dict
    lr: float
    derf_lr: float
    kv_decoder_lr: float


@dataclass
class EvalTiming:
    elapsed: float
    train_elapsed: float
    total_tokens: int


@dataclass
class LineProbeContext:
    batch: tuple[torch.Tensor, torch.Tensor] | None = None
    rng_before: object | None = None
    loss_before: float | None = None
    params_before: list[tuple[torch.Tensor, torch.Tensor]] | None = None


def build_training_components(
    args, device: torch.device, amp_dtype: torch.dtype | None
) -> TrainingComponents:
    metrics = MetricsLogger(args.metrics_jsonl, args.run_name)
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
    deepnorm_calibration = maybe_calibrate_deepnorm(raw_model, source, args)
    derf_opts = build_derf_optimizers(raw_model, args)
    kv_decoder_opt = build_kv_decoder_optimizer(raw_model, args)
    conv_probe = build_convergence_probe(raw_model, opt, args)
    model, compile_seconds = maybe_compile(raw_model, source, args, amp_dtype, device)
    if compile_seconds:
        print(f"compile_seconds {compile_seconds:.3f}")
    return TrainingComponents(
        metrics=metrics,
        dataset=dataset,
        raw_model=raw_model,
        source=source,
        eval_batches=eval_batches,
        opt=opt,
        derf_opts=derf_opts,
        kv_decoder_opt=kv_decoder_opt,
        conv_probe=conv_probe,
        model=model,
        compile_seconds=compile_seconds,
        deepnorm_calibration=deepnorm_calibration,
    )


def maybe_calibrate_deepnorm(raw_model: GPT, source: BatchSource, args) -> dict:
    if not args.deepnorm_calibrate_branches:
        return {}
    idx, _targets = source.seeded_batch("train", resolve_data_seed(args))
    calibration = calibrate_deepnorm_branches(raw_model, idx)
    text = deepnorm_calibration_text(calibration)
    if text:
        print(text)
    return calibration


def build_convergence_probe(raw_model: GPT, opt, args) -> ConvergenceProbe | None:
    if not args.track_convergence_stats:
        return None
    conv_probe = ConvergenceProbe(raw_model, opt, args)
    conv_probe.register_hooks(raw_model)
    if args.compile:
        print("compile_disabled_for_convergence_stats")
        args.compile = False
    return conv_probe


def make_training_schedule(args) -> TrainingSchedule:
    warmup_steps, stable_steps, decay_steps = resolve_training_schedule(args)
    return TrainingSchedule(
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        effective_tokens=count_increment(args),
    )


def collect_training_metadata(
    raw_model: GPT, opt, derf_opts: dict, kv_decoder_opt
) -> TrainingMetadata:
    first_group = opt.param_groups[0]
    return TrainingMetadata(
        lr_peak=float(first_group.get("lr_peak", first_group["lr"])),
        beta=float(first_group.get("beta", math.nan)),
        derf_beta=next(iter(derf_opts.values())).beta if derf_opts else math.nan,
        kv_decoder_beta=kv_decoder_opt.beta if kv_decoder_opt is not None else math.nan,
        state_half_life=float(first_group.get("state_half_life", math.nan)),
        io_weights=optimizer_io_label(raw_model),
        params_info=parameter_summary(raw_model),
        kv_info=kv_cache_summary(raw_model),
        deepnorm_alpha=raw_model.cfg.resolved_deepnorm_alpha,
        group_text=format_optimizer_schedule(opt),
    )


def print_training_config(
    args, schedule: TrainingSchedule, metadata: TrainingMetadata
) -> None:
    kv_info = metadata.kv_info
    params_info = metadata.params_info
    print(
        "schedule "
        f"warmup_steps={schedule.warmup_steps} "
        f"stable_steps={schedule.stable_steps} decay_steps={schedule.decay_steps} "
        f"count_increment={schedule.effective_tokens} "
        f"lr_peak={metadata.lr_peak:.6f} "
        f"state_half_life={metadata.state_half_life:.3g} "
        f"beta={metadata.beta:.6f} "
        f"optimizer={OPTIMIZER_NAME} update={args.hyperball_update} "
        f"compile_mode={args.compile_mode} "
        f"norm={args.norm_type} derf_lr={args.derf_lr:.6f} "
        f"derf_beta={metadata.derf_beta:.6f} derf_shape={args.train_derf_shape} "
        f"kv_cache={kv_info['type']} kv_dim={kv_info['cache_dim']}/"
        f"{kv_info['original_cache_dim']} kv_ratio={kv_info['cache_ratio']:.3f} "
        f"kv_key_rank={kv_info['key_rank']} kv_value_rank={kv_info['value_rank']} "
        f"kv_decoder_lr={args.kv_decoder_lr:.6f} "
        f"kv_decoder_beta={metadata.kv_decoder_beta:.6f} "
        f"params={params_info['total']} trainable_params={params_info['trainable']} "
        f"dropout={args.dropout:.3f} "
        f"resid_scale={args.resid_scale:.6f} "
        f"block_type={args.block_type} "
        f"deepnorm_alpha={metadata.deepnorm_alpha:.6g} "
        f"deepnorm_branch_scale={args.deepnorm_branch_scale:.6g} "
        f"deepnorm_calibrate={args.deepnorm_calibrate_branches} "
        f"lns={args.lns} "
        f"attn={args.attn_type} "
        f"seed={args.seed} data_seed={resolve_data_seed(args)} "
        f"eval_seed={resolve_eval_seed(args)} fixed_eval={args.fixed_eval_batches} "
        f"hidden_ulmo={args.hidden_ulmo} "
        f"io_weights={metadata.io_weights} embed_ulmo={args.embed_ulmo} "
        f"out_ulmo={args.out_ulmo} "
        f"qkv=fused spi_iteration={args.spi_iteration} "
        f"groups={metadata.group_text}"
    )


def write_config_metrics(
    args,
    metrics: MetricsLogger,
    schedule: TrainingSchedule,
    metadata: TrainingMetadata,
    deepnorm_calibration: dict,
    derf_opts: dict,
) -> None:
    metrics.write(
        "config",
        args=vars(args),
        schedule={
            "warmup_steps": schedule.warmup_steps,
            "stable_steps": schedule.stable_steps,
            "decay_steps": schedule.decay_steps,
            "count_increment": schedule.effective_tokens,
            "seed": args.seed,
            "data_seed": resolve_data_seed(args),
            "eval_seed": resolve_eval_seed(args),
            "compile_seed": resolve_compile_seed(args),
            "fixed_eval_batches": args.fixed_eval_batches,
        },
        optimizer={
            "name": OPTIMIZER_NAME,
            "lr_peak": metadata.lr_peak,
            "state_half_life": metadata.state_half_life,
            "beta": metadata.beta,
            "derf_lr_peak": args.derf_lr,
            "derf_state_half_life": args.derf_state_half_life,
            "derf_beta": metadata.derf_beta,
            "derf_groups": list(derf_opts),
            "kv_decoder_lr_peak": args.kv_decoder_lr,
            "kv_decoder_beta": metadata.kv_decoder_beta,
            "update_rule": args.hyperball_update,
            "groups": metadata.group_text,
        },
        model={
            "params": metadata.params_info,
            "dropout": args.dropout,
            "resid_scale": args.resid_scale,
            "block_type": args.block_type,
            "deepnorm_alpha": metadata.deepnorm_alpha,
            "deepnorm_branch_scale": args.deepnorm_branch_scale,
            "deepnorm_calibrate_branches": args.deepnorm_calibrate_branches,
            "deepnorm_calibration": deepnorm_calibration,
            "lns": args.lns,
            "norm_type": args.norm_type,
            "derf_alpha": args.derf_alpha,
            "derf_shift": args.derf_shift,
            "train_derf_shape": args.train_derf_shape,
            "attn_type": args.attn_type,
            "kv_cache": metadata.kv_info,
            "hidden_ulmo": args.hidden_ulmo,
            "io_weights": metadata.io_weights,
            "embed_ulmo": args.embed_ulmo,
            "out_ulmo": args.out_ulmo,
            "qkv": "fused",
            "spi_iteration": args.spi_iteration,
        },
    )


def warn_training_options(args, line_curve_scales: list[float]) -> None:
    if args.track_line_probe and args.grad_accum != 1:
        print("line_probe_disabled_requires_grad_accum_1")
    if line_curve_scales:
        print("line_curve_scales " + ",".join(f"{x:g}" for x in line_curve_scales))


def schedule_step_rates(
    args,
    opt,
    derf_opts: dict,
    kv_decoder_opt,
    step: int,
    schedule: TrainingSchedule,
) -> StepRates:
    current_lrs = apply_scheduled_lr(
        opt,
        step,
        args.max_iters,
        schedule.warmup_steps,
        schedule.decay_steps,
        args.schedule_floor,
    )
    derf_lr = schedule_derf_lr(args, derf_opts, step, schedule)
    kv_decoder_lr = schedule_kv_decoder_lr(args, kv_decoder_opt, step, schedule)
    return StepRates(
        all_lrs=current_lrs,
        lr=current_lrs.get("hidden", next(iter(current_lrs.values()))),
        derf_lr=derf_lr,
        kv_decoder_lr=kv_decoder_lr,
    )


def schedule_derf_lr(
    args, derf_opts: dict, step: int, schedule: TrainingSchedule
) -> float:
    if not derf_opts:
        return 0.0
    lr = schedule_at_step(
        step,
        args.max_iters,
        args.derf_lr,
        args.schedule_floor * args.derf_lr,
        schedule.warmup_steps,
        schedule.decay_steps,
    )
    for derf_group_opt in derf_opts.values():
        derf_group_opt.lr = lr
    return lr


def schedule_kv_decoder_lr(args, kv_decoder_opt, step: int, schedule: TrainingSchedule) -> float:
    if kv_decoder_opt is None:
        return 0.0
    lr = schedule_at_step(
        step,
        args.max_iters,
        kv_decoder_opt.lr_peak,
        args.schedule_floor * kv_decoder_opt.lr_peak,
        schedule.warmup_steps,
        schedule.decay_steps,
    )
    kv_decoder_opt.lr = lr
    return lr


def should_evaluate(args, step: int) -> bool:
    return step % args.eval_interval == 0 or step == args.max_iters - 1


def run_eval_step(
    args,
    components: TrainingComponents,
    schedule: TrainingSchedule,
    metadata: TrainingMetadata,
    progress: TrainingProgress,
    rates: StepRates,
    step: int,
    amp_dtype: torch.dtype | None,
    device: torch.device,
) -> None:
    eval_start = sync_now(device)
    train_loss = float(progress.last_train_loss)
    val_loss, logit_stats = estimate_val_metrics(
        components.model,
        components.source,
        args.eval_iters,
        amp_dtype,
        args.track_logit_stats,
        components.eval_batches["val"] if components.eval_batches is not None else None,
    )
    progress.last_val_loss = val_loss
    eval_stats = collect_eval_stats(args, components, progress)
    update_eval_tracking(progress, args, step, val_loss, components)

    now = sync_now(device)
    progress.eval_seconds += now - eval_start
    timing = current_eval_timing(progress, schedule, now)
    print_eval_status(
        step,
        train_loss,
        val_loss,
        timing,
        rates,
        metadata,
        eval_stats,
        logit_stats,
        device,
        progress,
    )
    memory = cuda_memory_stats(device)
    write_eval_metrics(
        components.metrics,
        step,
        train_loss,
        val_loss,
        timing,
        rates,
        metadata,
        progress,
        memory,
        eval_stats,
        logit_stats,
    )
    enforce_cuda_memory_limit(progress, args, memory)


def collect_eval_stats(
    args, components: TrainingComponents, progress: TrainingProgress
) -> dict:
    return {
        "opt": consume_step_stats(progress.step_stat_accum)
        if args.track_step_stats
        else {},
        "weight_rms": optimizer_rms_state(components.opt),
        "derf": derf_state(components.raw_model),
        "norm_affine": rmsnorm_affine_state(components.raw_model),
        "kv_decoder": kv_decoder_state(components.raw_model),
    }


def update_eval_tracking(
    progress: TrainingProgress,
    args,
    step: int,
    val_loss: float,
    components: TrainingComponents,
) -> None:
    if not math.isfinite(val_loss):
        progress.mark_diverged("nonfinite_eval_loss")
        return

    prev_best = progress.best_val
    if progress.initial_val is None:
        progress.initial_val = val_loss
    progress.best_val = min(progress.best_val, val_loss)
    progress.max_val = max(progress.max_val, val_loss)

    if step > 0 and val_loss > progress.initial_val * args.diverge_mult:
        progress.mark_diverged(f"val_loss_exceeded_{args.diverge_mult:.2f}x_initial")

    save_eval_artifacts(args, step, val_loss, prev_best, components)


def save_eval_artifacts(
    args,
    step: int,
    val_loss: float,
    prev_best: float,
    components: TrainingComponents,
) -> None:
    if args.no_save:
        return
    out_path = Path(args.out_path)
    if val_loss < prev_best:
        save_checkpoint(out_path, components.raw_model, components.dataset)
    save_eval_checkpoint(
        out_path, step, val_loss, components.raw_model, components.dataset, args
    )
    if step == args.max_iters - 1:
        save_checkpoint(
            out_path.with_name(f"{out_path.stem}_final{out_path.suffix}"),
            components.raw_model,
            components.dataset,
        )


def current_eval_timing(
    progress: TrainingProgress, schedule: TrainingSchedule, now: float
) -> EvalTiming:
    elapsed = max(now - progress.train_start, 1e-9)
    train_elapsed = max(elapsed - progress.eval_seconds, 1e-9)
    total_tokens = progress.total_opt_steps * schedule.effective_tokens
    return EvalTiming(elapsed, train_elapsed, total_tokens)


def print_eval_status(
    step: int,
    train_loss: float,
    val_loss: float,
    timing: EvalTiming,
    rates: StepRates,
    metadata: TrainingMetadata,
    eval_stats: dict,
    logit_stats: dict,
    device: torch.device,
    progress: TrainingProgress,
) -> None:
    print(
        f"step {step:5d} | lr {rates.lr:.6f} beta {metadata.beta:.6f} "
        f"derf_lr {rates.derf_lr:.6f} "
        f"kv_decoder_lr {rates.kv_decoder_lr:.6f} | "
        f"train {train_loss:.4f} | val {val_loss:.4f} | "
        f"best_val {progress.best_val:.4f} | train_seconds {timing.elapsed:.3f} | "
        f"tok/s {timing.total_tokens / timing.elapsed:.0f} "
        f"train_tok/s {timing.total_tokens / timing.train_elapsed:.0f}"
        f"{cuda_memory_text(device)}{logit_stats_text(logit_stats)}"
        f"{rms_state_text(eval_stats['weight_rms'])}"
        f"{derf_state_text(eval_stats['derf'])}"
        f"{rmsnorm_affine_state_text(eval_stats['norm_affine'])}"
        f"{kv_decoder_state_text(eval_stats['kv_decoder'])}"
        f"{step_stats_text(eval_stats['opt'])}"
    )


def logit_stats_text(logit_stats: dict) -> str:
    if not logit_stats:
        return ""
    return (
        " | logits "
        f"std={logit_stats['logit_std']:.3f},"
        f"H={logit_stats['softmax_entropy']:.3f},"
        f"pmax={logit_stats['softmax_max_prob']:.3f}"
    )


def write_eval_metrics(
    metrics: MetricsLogger,
    step: int,
    train_loss: float,
    val_loss: float,
    timing: EvalTiming,
    rates: StepRates,
    metadata: TrainingMetadata,
    progress: TrainingProgress,
    memory: dict,
    eval_stats: dict,
    logit_stats: dict,
) -> None:
    metrics.write(
        "eval",
        step=step,
        total_opt_steps=progress.total_opt_steps,
        lr=rates.lr,
        derf_lr=rates.derf_lr,
        kv_decoder_lr=rates.kv_decoder_lr,
        beta=metadata.beta,
        lrs=rates.all_lrs,
        train_loss=train_loss,
        val_loss=val_loss,
        best_val=progress.best_val,
        max_val=progress.max_val,
        train_seconds=timing.elapsed,
        train_compute_seconds=timing.train_elapsed,
        eval_seconds=progress.eval_seconds,
        tokens_per_second=timing.total_tokens / timing.elapsed,
        train_tokens_per_second=timing.total_tokens / timing.train_elapsed,
        cuda_memory=memory,
        logit_stats=logit_stats,
        weight_rms=eval_stats["weight_rms"],
        derf=eval_stats["derf"],
        norm_affine=eval_stats["norm_affine"],
        kv_decoder=eval_stats["kv_decoder"],
        step_stats=eval_stats["opt"],
    )


def enforce_cuda_memory_limit(
    progress: TrainingProgress, args, memory: dict[str, float]
) -> None:
    if (
        args.max_cuda_reserved_gb > 0
        and memory
        and memory["reserved_gb"] > args.max_cuda_reserved_gb
    ):
        progress.mark_diverged(
            f"cuda_reserved_{memory['reserved_gb']:.2f}G_exceeded_"
            f"{args.max_cuda_reserved_gb:.2f}G"
        )


def run_training_step(
    args,
    components: TrainingComponents,
    progress: TrainingProgress,
    rates: StepRates,
    line_curve_scales: list[float],
    step: int,
    amp_dtype: torch.dtype | None,
    device: torch.device,
) -> None:
    line_active = line_probe_active(args, step)
    line_context = LineProbeContext()
    zero_training_optimizers(components)
    if components.conv_probe is not None:
        components.conv_probe.start_step(step)

    train_loss = accumulate_microbatches(
        components.model,
        components.source,
        args,
        amp_dtype,
        device,
        line_active,
        line_context,
    )
    progress.last_train_loss = (
        float(train_loss) if train_loss is not None else float("nan")
    )
    apply_gradient_clipping(args, components.raw_model)
    capture_convergence(components, progress, rates, step)

    if line_active and line_curve_scales:
        line_context.params_before = capture_params(components.raw_model.parameters())

    stat_snapshot = capture_training_step_stats(args, components.opt, line_active)
    step_training_optimizers(components)
    line_stats = consume_training_step_stats(
        args, progress.step_stat_accum, stat_snapshot, line_active
    )
    progress.total_opt_steps = step + 1
    record_line_probe(
        args,
        components,
        progress,
        line_context,
        line_curve_scales,
        line_stats,
        step,
        amp_dtype,
        device,
        line_active,
    )


def zero_training_optimizers(components: TrainingComponents) -> None:
    components.opt.zero_grad(set_to_none=True)
    if components.derf_opts:
        zero_derf_optimizers(components.derf_opts, set_to_none=True)
    if components.kv_decoder_opt is not None:
        components.kv_decoder_opt.zero_grad(set_to_none=True)


def accumulate_microbatches(
    model,
    source: BatchSource,
    args,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    line_active: bool,
    line_context: LineProbeContext,
) -> torch.Tensor | None:
    train_loss = None
    for micro_step in range(args.grad_accum):
        batch = source.get("train")
        if line_active and micro_step == 0:
            line_context.batch = batch
            line_context.rng_before = capture_rng(device)
        with amp_ctx(amp_dtype):
            _, loss = model(*batch)
            loss = loss / args.grad_accum
        loss_value = loss.detach()
        if line_active and micro_step == 0:
            line_context.loss_before = float(loss_value)
        train_loss = loss_value if train_loss is None else train_loss + loss_value
        loss.backward()
    return train_loss


def apply_gradient_clipping(args, raw_model: GPT) -> None:
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)


def capture_convergence(
    components: TrainingComponents,
    progress: TrainingProgress,
    rates: StepRates,
    step: int,
) -> None:
    if components.conv_probe is None:
        return
    conv_text = components.conv_probe.capture(step, rates.all_lrs)
    if not conv_text:
        return
    print(conv_text)
    components.metrics.write(
        "convergence",
        step=step,
        total_opt_steps=progress.total_opt_steps,
        lrs=rates.all_lrs,
        groups=components.conv_probe.summary,
    )


def capture_training_step_stats(args, opt, line_active: bool):
    if args.track_step_stats or line_active:
        return capture_step_stats(opt)
    return None


def step_training_optimizers(components: TrainingComponents) -> None:
    components.opt.step()
    if components.derf_opts:
        step_derf_optimizers(components.derf_opts)
    if components.kv_decoder_opt is not None:
        components.kv_decoder_opt.step()


def consume_training_step_stats(
    args, step_stat_accum: dict, stat_snapshot, line_active: bool
) -> dict:
    if stat_snapshot is None:
        return {}
    if args.track_step_stats:
        accumulate_step_stats(step_stat_accum, stat_snapshot)
    if not line_active:
        return {}
    line_stat_accum = {}
    accumulate_step_stats(line_stat_accum, stat_snapshot)
    return consume_step_stats(line_stat_accum)


def record_line_probe(
    args,
    components: TrainingComponents,
    progress: TrainingProgress,
    line_context: LineProbeContext,
    line_curve_scales: list[float],
    line_stats: dict,
    step: int,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    line_active: bool,
) -> None:
    if not line_active:
        return
    line_record = run_line_probe(
        components.model,
        step,
        line_context.batch,
        line_context.rng_before,
        line_context.loss_before,
        line_context.params_before,
        line_curve_scales,
        line_stats,
        amp_dtype,
        device,
    )
    if not line_record:
        return
    components.metrics.write(
        "line_probe",
        step=step,
        total_opt_steps=progress.total_opt_steps,
        **line_record,
    )


def maybe_generate_samples(
    args,
    raw_model: GPT,
    dataset,
    device: torch.device,
    diverged: bool,
) -> None:
    if args.skip_sample or diverged:
        return
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


def print_ulmo_stats(opt) -> None:
    for group in opt.param_groups:
        stats = getattr(group.get("ulmo"), "stats", None)
        if stats:
            print(
                f"{group.get('name', 'group')}_ulmo_stats "
                + " ".join(f"{k}={v}" for k, v in stats.items())
            )


def final_training_result(
    progress: TrainingProgress,
    schedule: TrainingSchedule,
    compile_seconds: float,
) -> dict:
    return {
        "best_val": progress.best_val,
        "final_train": progress.last_train_loss,
        "final_val": progress.last_val_loss,
        "compile_seconds": compile_seconds,
        "initial_val": float("nan")
        if progress.initial_val is None
        else progress.initial_val,
        "max_val": progress.max_val,
        "diverged": progress.diverged,
        "diverge_reason": progress.diverge_reason,
        "warmup_steps": schedule.warmup_steps,
        "stable_steps": schedule.stable_steps,
        "decay_steps": schedule.decay_steps,
    }


def train(args):
    args.hyperball_update = resolve_hyperball_update(args)
    device, amp_dtype = configure_runtime(args)
    line_curve_scales = parse_line_scales(args.line_curve_scales)
    if line_curve_scales:
        args.track_line_probe = True

    components = build_training_components(args, device, amp_dtype)
    schedule = make_training_schedule(args)
    metadata = collect_training_metadata(
        components.raw_model,
        components.opt,
        components.derf_opts,
        components.kv_decoder_opt,
    )
    print_training_config(args, schedule, metadata)
    write_config_metrics(
        args,
        components.metrics,
        schedule,
        metadata,
        components.deepnorm_calibration,
        components.derf_opts,
    )
    warn_training_options(args, line_curve_scales)

    progress = TrainingProgress(train_start=sync_now(device))
    for step in range(args.max_iters):
        rates = schedule_step_rates(
            args,
            components.opt,
            components.derf_opts,
            components.kv_decoder_opt,
            step,
            schedule,
        )
        if should_evaluate(args, step):
            run_eval_step(
                args,
                components,
                schedule,
                metadata,
                progress,
                rates,
                step,
                amp_dtype,
                device,
            )
        if progress.diverged:
            print(f"diverged {progress.diverge_reason}")
            break
        run_training_step(
            args,
            components,
            progress,
            rates,
            line_curve_scales,
            step,
            amp_dtype,
            device,
        )

    maybe_generate_samples(
        args,
        components.raw_model,
        components.dataset,
        device,
        progress.diverged,
    )
    print_ulmo_stats(components.opt)
    result = final_training_result(progress, schedule, components.compile_seconds)
    components.metrics.write("final", **result)
    components.metrics.close()
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
