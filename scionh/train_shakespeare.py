import math
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
        f"compile_mode={args.compile_mode} "
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
