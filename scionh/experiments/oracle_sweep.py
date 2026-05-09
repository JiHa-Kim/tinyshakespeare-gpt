import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Arm:
    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class PlannedRun:
    arm: Arm
    seed: int
    run_name: str
    metrics_path: Path
    cmd: list[str]


SCREENING_ARMS = (
    Arm("hidden_gram_ns", ("--hidden-ulmo", "gram-ns")),
    Arm("hidden_swan", ("--hidden-ulmo", "swan")),
    Arm("hidden_streaming_svd", ("--hidden-ulmo", "streaming-svd")),
    Arm("hidden_frobenius", ("--hidden-ulmo", "frobenius")),
    Arm("hidden_colnorm", ("--hidden-ulmo", "colnorm")),
    Arm("hidden_rownorm", ("--hidden-ulmo", "rownorm")),
    Arm("hidden_sign", ("--hidden-ulmo", "sign")),
)

BLOCKWISE_ARMS = (
    Arm("hidden_gram_ns", ("--hidden-ulmo", "gram-ns")),
    Arm("hidden_svd_exact", ("--hidden-ulmo", "svd")),
    Arm(
        "hidden_block_rows2",
        (
            "--hidden-ulmo",
            "blockwise-gram-ns",
            "--block-ulmo-axis",
            "rows",
            "--block-ulmo-parts",
            "2",
        ),
    ),
    Arm(
        "hidden_block_cols2",
        (
            "--hidden-ulmo",
            "blockwise-gram-ns",
            "--block-ulmo-axis",
            "cols",
            "--block-ulmo-parts",
            "2",
        ),
    ),
    Arm(
        "hidden_block_rows4",
        (
            "--hidden-ulmo",
            "blockwise-gram-ns",
            "--block-ulmo-axis",
            "rows",
            "--block-ulmo-parts",
            "4",
        ),
    ),
    Arm(
        "hidden_block_cols4",
        (
            "--hidden-ulmo",
            "blockwise-gram-ns",
            "--block-ulmo-axis",
            "cols",
            "--block-ulmo-parts",
            "4",
        ),
    ),
)

LAYER_POLICY_ARMS = (
    Arm(
        "policy_gns",
        ("--embed-ulmo", "colnorm", "--hidden-ulmo", "gram-ns", "--out-ulmo", "sign"),
    ),
    Arm(
        "policy_streaming_svd",
        (
            "--embed-ulmo",
            "colnorm",
            "--hidden-ulmo",
            "streaming-svd",
            "--out-ulmo",
            "sign",
        ),
    ),
    Arm(
        "policy_frobenius",
        (
            "--embed-ulmo",
            "frobenius",
            "--hidden-ulmo",
            "frobenius",
            "--out-ulmo",
            "frobenius",
        ),
    ),
)

EDGE_ARMS = (
    Arm("edge_default", ()),
    Arm("embed_frobenius", ("--embed-ulmo", "frobenius")),
    Arm("out_frobenius", ("--out-ulmo", "frobenius")),
    Arm("io_frobenius", ("--embed-ulmo", "frobenius", "--out-ulmo", "frobenius")),
    Arm("out_colnorm", ("--out-ulmo", "colnorm")),
)


def unique_arms(arms: tuple[Arm, ...]) -> tuple[Arm, ...]:
    out = {}
    for arm in arms:
        out.setdefault(arm.name, arm)
    return tuple(out.values())


def arms_for_matrix(name: str) -> tuple[Arm, ...]:
    if name == "screening":
        return SCREENING_ARMS
    if name == "blockwise":
        return BLOCKWISE_ARMS
    if name == "layer-policy":
        return LAYER_POLICY_ARMS
    if name == "edge":
        return EDGE_ARMS
    if name == "all":
        return unique_arms(
            SCREENING_ARMS
            + BLOCKWISE_ARMS
            + LAYER_POLICY_ARMS
            + EDGE_ARMS
        )
    raise ValueError(f"unknown matrix: {name}")


def parse_seeds(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def parse_names(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def select_arms(matrix: str, names: str) -> tuple[Arm, ...]:
    arms = arms_for_matrix(matrix)
    selected = parse_names(names)
    if not selected:
        return arms
    by_name = {arm.name: arm for arm in arms}
    missing = sorted(selected - set(by_name))
    if missing:
        valid = ", ".join(sorted(by_name))
        raise ValueError(f"unknown arm(s): {', '.join(missing)}; valid arms: {valid}")
    return tuple(arm for arm in arms if arm.name in selected)


def metrics_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            return any(
                '"event":"final"' in line or '"event": "final"' in line
                for line in f
            )
    except OSError:
        return False


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        choices=[
            "screening",
            "blockwise",
            "layer-policy",
            "edge",
            "all",
        ],
        default="screening",
    )
    parser.add_argument(
        "--arms",
        default="",
        help="comma-separated arm names to run from the selected matrix",
    )
    parser.add_argument("--seeds", default="1337,1338,1339")
    parser.add_argument("--metrics-dir", default="_local/oracle_lab/runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--convergence-interval", type=int, default=50)
    parser.add_argument("--convergence-support-steps", type=int, default=7)
    parser.add_argument(
        "--convergence-probe",
        choices=["representative", "all"],
        default="all",
    )
    parser.add_argument(
        "--line-curves",
        action="store_true",
        help="enable same-batch line probes and a small loss curve",
    )
    parser.add_argument("--line-probe-interval", type=int, default=100)
    parser.add_argument("--line-curve-scales", default="0,0.5,1,1.5")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--track-step-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="capture optimizer update statistics; disables clean throughput comparisons",
    )
    parser.add_argument(
        "--track-convergence-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="capture convergence probes; train_shakespeare disables model compile for this",
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute commands; otherwise print the planned commands",
    )
    parser.add_argument(
        "--skip-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip metrics files that already contain a final record",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="delete an existing metrics file for a selected run before executing",
    )
    return parser


def plan_run(args, arm: Arm, seed: int, extra: list[str]) -> PlannedRun:
    run_name = f"{args.matrix}_seed{seed}_{arm.name}"
    metrics_path = Path(args.metrics_dir) / f"{run_name}.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "scionh.train_shakespeare",
        "--mode",
        "train",
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--data-seed",
        str(seed),
        "--eval-seed",
        str(seed + 10_000),
        "--max-iters",
        str(args.max_iters),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-iters",
        str(args.eval_iters),
        "--metrics-jsonl",
        str(metrics_path),
        "--run-name",
        run_name,
    ]
    if args.track_step_stats:
        cmd.append("--track-step-stats")
    if args.track_convergence_stats:
        cmd.extend(
            [
                "--track-convergence-stats",
                "--convergence-probe",
                args.convergence_probe,
                "--convergence-interval",
                str(args.convergence_interval),
                "--convergence-support-steps",
                str(args.convergence_support_steps),
            ]
        )
    if not args.compile:
        cmd.append("--no-compile")
    if not args.save:
        cmd.extend(["--no-save", "--skip-sample"])
    if args.line_curves:
        cmd.extend(
            [
                "--line-probe-interval",
                str(args.line_probe_interval),
                "--line-curve-scales",
                args.line_curve_scales,
            ]
        )
    cmd.extend(arm.args)
    cmd.extend(extra)
    return PlannedRun(arm, seed, run_name, metrics_path, cmd)


def format_command(cmd: list[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def main() -> int:
    parser = make_parser()
    args, extra = parser.parse_known_args()
    plans = [
        plan_run(args, arm, seed, extra)
        for seed in parse_seeds(args.seeds)
        for arm in select_arms(args.matrix, args.arms)
    ]

    if not args.run:
        for plan in plans:
            status = ""
            if plan.metrics_path.exists():
                status = (
                    " # complete"
                    if metrics_complete(plan.metrics_path)
                    else " # existing"
                )
            print(format_command(plan.cmd) + status)
        return 0

    for plan in plans:
        exists = plan.metrics_path.exists()
        complete = metrics_complete(plan.metrics_path)
        if exists and args.skip_complete and complete and not args.force_rerun:
            print(f"skip_complete {plan.run_name}", flush=True)
            continue
        if exists and not args.force_rerun:
            print(
                f"skip_existing {plan.run_name} "
                f"(use --force-rerun to replace {plan.metrics_path})",
                flush=True,
            )
            continue
        if exists and args.force_rerun:
            plan.metrics_path.unlink()
        print(format_command(plan.cmd), flush=True)
        subprocess.run(plan.cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
