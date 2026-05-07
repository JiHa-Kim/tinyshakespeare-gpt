import argparse
import json
from pathlib import Path

import torch

from scionh.optim.setup import resolve_hyperball_update
from scionh.probes.line import parse_line_scales
from scionh.train_shakespeare import (
    LineProbeContext,
    TrainingProgress,
    accumulate_microbatches,
    apply_gradient_clipping,
    build_training_components,
    capture_training_step_stats,
    consume_training_step_stats,
    make_training_schedule,
    run_training_step,
    schedule_step_rates,
    step_training_optimizers,
    zero_training_optimizers,
)
from scionh.training.cli import make_parser
from scionh.training.device import cuda_memory_stats, sync_now
from scionh.training.metrics import jsonable
from scionh.training.runtime import configure_runtime


def make_perf_parser() -> argparse.ArgumentParser:
    parser = make_parser()
    parser.description = "Benchmark training-step performance without eval or sampling."
    parser.set_defaults(mode="train", no_save=True, skip_sample=True, compile=False)
    parser.add_argument("--perf-warmup", type=int, default=6)
    parser.add_argument("--perf-steps", type=int, default=10)
    parser.add_argument("--perf-json", default="")
    parser.add_argument("--perf-profile", action="store_true")
    parser.add_argument("--perf-profile-rows", type=int, default=20)
    parser.add_argument("--perf-record-shapes", action="store_true")
    return parser


def _timed_step(args, components, progress, schedule, amp_dtype, device) -> dict:
    times = {}

    t0 = sync_now(device)
    rates = schedule_step_rates(
        args,
        components.opt,
        components.derf_opts,
        components.kv_decoder_opt,
        progress.total_opt_steps,
        schedule,
    )
    t1 = sync_now(device)

    zero_training_optimizers(components)
    t2 = sync_now(device)

    train_loss = accumulate_microbatches(
        components.model,
        components.source,
        args,
        amp_dtype,
        device,
        False,
        LineProbeContext(),
    )
    progress.last_train_loss = float(train_loss) if train_loss is not None else float("nan")
    t3 = sync_now(device)

    apply_gradient_clipping(args, components.raw_model)
    t4 = sync_now(device)

    stat_snapshot = capture_training_step_stats(args, components.opt, False)
    consume_training_step_stats(args, progress.step_stat_accum, stat_snapshot, False)
    t5 = sync_now(device)

    step_training_optimizers(components)
    progress.total_opt_steps += 1
    t6 = sync_now(device)

    times["schedule"] = t1 - t0
    times["zero_grad"] = t2 - t1
    times["fwd_bwd"] = t3 - t2
    times["grad_clip"] = t4 - t3
    times["step_stats"] = t5 - t4
    times["opt_step"] = t6 - t5
    times["total"] = t6 - t0
    times["loss"] = progress.last_train_loss
    times["lr"] = rates.lr
    return times


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _summarize(rows: list[dict]) -> dict[str, float]:
    fields = (
        "schedule",
        "zero_grad",
        "fwd_bwd",
        "grad_clip",
        "step_stats",
        "opt_step",
        "total",
    )
    return {f"{field}_ms": 1000.0 * _mean([row[field] for row in rows]) for field in fields}


def _print_summary(summary: dict, args, device: torch.device, rows: list[dict]) -> None:
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    total_seconds = summary["total_ms"] / 1000.0
    tok_s = tokens_per_step / max(total_seconds, 1e-12)
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device)
    )
    print(
        "perf "
        f"device={device_name!r} "
        f"steps={len(rows)} warmup={args.perf_warmup} "
        f"tokens_per_step={tokens_per_step} "
        f"total_ms={summary['total_ms']:.3f} "
        f"fwd_bwd_ms={summary['fwd_bwd_ms']:.3f} "
        f"opt_step_ms={summary['opt_step_ms']:.3f} "
        f"tok_s={tok_s:.0f}"
    )
    print(
        "perf_segments "
        f"schedule_ms={summary['schedule_ms']:.3f} "
        f"zero_grad_ms={summary['zero_grad_ms']:.3f} "
        f"grad_clip_ms={summary['grad_clip_ms']:.3f} "
        f"step_stats_ms={summary['step_stats_ms']:.3f}"
    )


def _write_json(path: str, payload: dict) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    print(f"wrote_perf_json {out_path}")


def _profile_steps(args, components, progress, schedule, amp_dtype, device) -> str:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.perf_record_shapes,
    ) as prof:
        for _ in range(args.perf_steps):
            run_training_step(
                args,
                components,
                progress,
                schedule_step_rates(
                    args,
                    components.opt,
                    components.derf_opts,
                    components.kv_decoder_opt,
                    progress.total_opt_steps,
                    schedule,
                ),
                parse_line_scales(args.line_curve_scales),
                progress.total_opt_steps,
                amp_dtype,
                device,
            )
    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=args.perf_profile_rows)
    print(table)
    return table


def main() -> None:
    args = make_perf_parser().parse_args()
    args.hyperball_update = resolve_hyperball_update(args)
    device, amp_dtype = configure_runtime(args)
    components = build_training_components(args, device, amp_dtype)
    schedule = make_training_schedule(args)
    progress = TrainingProgress(train_start=sync_now(device))

    for step in range(args.perf_warmup):
        rates = schedule_step_rates(
            args,
            components.opt,
            components.derf_opts,
            components.kv_decoder_opt,
            step,
            schedule,
        )
        run_training_step(
            args,
            components,
            progress,
            rates,
            parse_line_scales(args.line_curve_scales),
            step,
            amp_dtype,
            device,
        )

    rows = [_timed_step(args, components, progress, schedule, amp_dtype, device) for _ in range(args.perf_steps)]
    summary = _summarize(rows)
    _print_summary(summary, args, device, rows)

    profile_table = ""
    if args.perf_profile:
        profile_table = _profile_steps(args, components, progress, schedule, amp_dtype, device)

    payload = {
        "summary": summary,
        "rows": rows,
        "profile": profile_table,
        "cuda_memory": cuda_memory_stats(device),
        "config": {
            "device": str(device),
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "grad_accum": args.grad_accum,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "d_model": args.d_model,
            "hidden_ulmo": args.hidden_ulmo,
            "pe_steps": args.pe_steps,
            "gns_cudagraph": args.gns_cudagraph,
            "hyperball_update": args.hyperball_update,
            "compile": args.compile,
        },
    }
    _write_json(args.perf_json, payload)


if __name__ == "__main__":
    main()
