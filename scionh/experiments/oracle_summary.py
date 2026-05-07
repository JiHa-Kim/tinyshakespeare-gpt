import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RUN_RE = re.compile(r"^(?P<matrix>.+)_seed(?P<seed>\d+)_(?P<arm>.+)$")


@dataclass
class RunSummary:
    run_name: str
    path: Path
    matrix: str = ""
    seed: int | None = None
    arm: str = ""
    best_val: float | None = None
    final_val: float | None = None
    last_val: float | None = None
    initial_val: float | None = None
    diverged: bool = False
    diverge_reason: str = ""
    complete: bool = False
    evals: int = 0
    train_tps: list[float] = field(default_factory=list)
    wall_tps: list[float] = field(default_factory=list)
    hidden_rspec_early: list[float] = field(default_factory=list)
    hidden_act_early: list[float] = field(default_factory=list)
    hidden_l1_early: list[float] = field(default_factory=list)
    hidden_gdual_early: list[float] = field(default_factory=list)
    line_ratio: list[float] = field(default_factory=list)
    line_curvature: list[float] = field(default_factory=list)
    hidden_ulmo: str = ""
    embed_ulmo: str = ""
    out_ulmo: str = ""
    block_axis: str = ""
    block_parts: int | None = None
    hidden_ulmo_calls: int | None = None
    hidden_ulmo_active_calls: int | None = None


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def median(values: list[float]) -> float | None:
    values = sorted(x for x in values if math.isfinite(x))
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def mean(values: list[float]) -> float | None:
    values = [x for x in values if math.isfinite(x)]
    if not values:
        return None
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float | None:
    values = [x for x in values if math.isfinite(x)]
    if len(values) < 2:
        return None
    center = sum(values) / len(values)
    variance = sum((x - center) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def standard_error(values: list[float]) -> float | None:
    std = sample_std(values)
    if std is None:
        return None
    return std / math.sqrt(len(values))


def t95_critical(n: int) -> float | None:
    if n < 2:
        return None
    # Two-sided 95% Student-t critical values by degrees of freedom.
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    return table.get(n - 1, 1.96)


def ci95(values: list[float]) -> tuple[float | None, float | None]:
    center = mean(values)
    se = standard_error(values)
    critical = t95_critical(len(values))
    if center is None or se is None or critical is None:
        return None, None
    half_width = critical * se
    return center - half_width, center + half_width


def parse_run_name(run_name: str) -> tuple[str, int | None, str]:
    match = RUN_RE.match(run_name)
    if match is None:
        return "", None, run_name
    return match.group("matrix"), int(match.group("seed")), match.group("arm")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL record") from exc
    return records


def apply_config(summary: RunSummary, record: dict) -> None:
    args = record.get("args") or {}
    model = record.get("model") or {}
    summary.hidden_ulmo = str(
        model.get("hidden_ulmo") or args.get("hidden_ulmo") or ""
    )
    summary.embed_ulmo = str(model.get("embed_ulmo") or args.get("embed_ulmo") or "")
    summary.out_ulmo = str(model.get("out_ulmo") or args.get("out_ulmo") or "")
    summary.block_axis = str(
        model.get("block_ulmo_axis") or args.get("block_ulmo_axis") or ""
    )
    parts = model.get("block_ulmo_parts", args.get("block_ulmo_parts"))
    if isinstance(parts, int):
        summary.block_parts = parts


def append_finite(values: list[float], value: Any) -> None:
    x = finite(value)
    if x is not None:
        values.append(x)


def append_positive(values: list[float], value: Any) -> None:
    x = finite(value)
    if x is not None and x > 0.0:
        values.append(x)


def apply_eval(summary: RunSummary, record: dict) -> None:
    summary.evals += 1
    val = finite(record.get("val_loss"))
    if val is not None:
        summary.last_val = val
        summary.best_val = (
            val if summary.best_val is None else min(summary.best_val, val)
        )
    best = finite(record.get("best_val"))
    if best is not None:
        summary.best_val = (
            best if summary.best_val is None else min(summary.best_val, best)
        )
    append_positive(summary.train_tps, record.get("train_tokens_per_second"))
    append_positive(summary.wall_tps, record.get("tokens_per_second"))


def apply_convergence(summary: RunSummary, record: dict, early_steps: int) -> None:
    step = record.get("step")
    if not isinstance(step, int) or step > early_steps:
        return
    hidden = (record.get("groups") or {}).get("hidden") or {}
    append_finite(summary.hidden_rspec_early, hidden.get("spec_ratio"))
    append_finite(summary.hidden_act_early, hidden.get("action_eff"))
    append_finite(summary.hidden_l1_early, hidden.get("l1"))
    append_finite(summary.hidden_gdual_early, hidden.get("gdual"))


def apply_line_probe(summary: RunSummary, record: dict, early_steps: int) -> None:
    step = record.get("step")
    if not isinstance(step, int) or step > early_steps:
        return
    probe = record.get("probe") or {}
    append_finite(summary.line_ratio, probe.get("ratio"))
    append_finite(summary.line_curvature, probe.get("curvature"))


def apply_final(summary: RunSummary, record: dict) -> None:
    summary.complete = True
    best = finite(record.get("best_val"))
    if best is not None:
        summary.best_val = (
            best if summary.best_val is None else min(summary.best_val, best)
        )
    summary.final_val = finite(record.get("final_val"))
    summary.initial_val = finite(record.get("initial_val"))
    summary.diverged = bool(record.get("diverged", False))
    summary.diverge_reason = str(record.get("diverge_reason") or "")
    ulmo_stats = record.get("ulmo_stats") or {}
    hidden = ulmo_stats.get("hidden") or {}
    calls = hidden.get("calls")
    active_calls = hidden.get("active_calls")
    if isinstance(calls, int):
        summary.hidden_ulmo_calls = calls
    if isinstance(active_calls, int):
        summary.hidden_ulmo_active_calls = active_calls


def summarize_file(path: Path, early_steps: int) -> RunSummary:
    records = load_jsonl(path)
    run_name = path.stem
    for record in records:
        if record.get("run_name"):
            run_name = str(record["run_name"])
            break
    matrix, seed, arm = parse_run_name(run_name)
    summary = RunSummary(
        run_name=run_name,
        path=path,
        matrix=matrix,
        seed=seed,
        arm=arm,
    )

    for record in records:
        event = record.get("event")
        if event == "config":
            apply_config(summary, record)
        elif event == "eval":
            apply_eval(summary, record)
        elif event == "convergence":
            apply_convergence(summary, record, early_steps)
        elif event == "line_probe":
            apply_line_probe(summary, record, early_steps)
        elif event == "final":
            apply_final(summary, record)
    return summary


def discover(paths: list[Path], pattern: str) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob(pattern)))
        elif path.is_file():
            files.append(path)
    return files


def run_row(summary: RunSummary, baseline: RunSummary | None) -> dict[str, Any]:
    tps = median(summary.train_tps)
    best = summary.best_val
    baseline_best = baseline.best_val if baseline else None
    baseline_tps = median(baseline.train_tps) if baseline else None
    delta_best = (
        best - baseline_best
        if best is not None and baseline_best is not None
        else None
    )
    delta_tps_pct = (
        100.0 * (tps / baseline_tps - 1.0)
        if tps is not None and baseline_tps not in (None, 0.0)
        else None
    )
    return {
        "matrix": summary.matrix,
        "seed": summary.seed,
        "arm": summary.arm,
        "best_val": best,
        "delta_best": delta_best,
        "final_val": summary.final_val,
        "last_val": summary.last_val,
        "train_tps": tps,
        "delta_tps_pct": delta_tps_pct,
        "early_rspec_hidden": median(summary.hidden_rspec_early),
        "early_act_hidden": median(summary.hidden_act_early),
        "early_l1_hidden": median(summary.hidden_l1_early),
        "early_gdual_hidden": median(summary.hidden_gdual_early),
        "line_ratio": median(summary.line_ratio),
        "line_curvature": median(summary.line_curvature),
        "evals": summary.evals,
        "complete": summary.complete,
        "diverged": summary.diverged,
        "diverge_reason": summary.diverge_reason,
        "hidden_ulmo": summary.hidden_ulmo,
        "embed_ulmo": summary.embed_ulmo,
        "out_ulmo": summary.out_ulmo,
        "block_axis": summary.block_axis,
        "block_parts": summary.block_parts,
        "hidden_ulmo_calls": summary.hidden_ulmo_calls,
        "hidden_ulmo_active_calls": summary.hidden_ulmo_active_calls,
        "run_name": summary.run_name,
        "path": str(summary.path),
    }


def baseline_by_seed(
    summaries: list[RunSummary], baseline_arm: str
) -> dict[tuple[str, int | None], RunSummary]:
    baselines = {}
    for summary in summaries:
        if summary.arm == baseline_arm:
            baselines[(summary.matrix, summary.seed)] = summary
    return baselines


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["matrix"]), str(row["arm"])), []).append(row)

    out = []
    for (matrix, arm), items in sorted(grouped.items()):
        paired_best = numeric_values(items, "delta_best")
        paired_tps = numeric_values(items, "delta_tps_pct")
        best_ci_low, best_ci_high = ci95(paired_best)
        tps_ci_low, tps_ci_high = ci95(paired_tps)
        out.append(
            {
                "matrix": matrix,
                "arm": arm,
                "runs": len(items),
                "complete": sum(1 for item in items if item["complete"]),
                "incomplete": sum(1 for item in items if not item["complete"]),
                "diverged": sum(1 for item in items if item["diverged"]),
                "paired": len(paired_best),
                "val_wins": sum(1 for x in paired_best if x < 0.0),
                "speed_wins": sum(1 for x in paired_tps if x > 0.0),
                "best_val_mean": mean_values(items, "best_val"),
                "best_val_median": median_values(items, "best_val"),
                "delta_best_mean": mean(paired_best),
                "delta_best_median": median(paired_best),
                "delta_best_sem": standard_error(paired_best),
                "delta_best_ci95_low": best_ci_low,
                "delta_best_ci95_high": best_ci_high,
                "train_tps_median": median_values(items, "train_tps"),
                "delta_tps_pct_mean": mean(paired_tps),
                "delta_tps_pct_median": median(paired_tps),
                "delta_tps_pct_sem": standard_error(paired_tps),
                "delta_tps_pct_ci95_low": tps_ci_low,
                "delta_tps_pct_ci95_high": tps_ci_high,
                "early_rspec_hidden_median": median_values(
                    items, "early_rspec_hidden"
                ),
                "early_act_hidden_median": median_values(items, "early_act_hidden"),
                "line_ratio_median": median_values(items, "line_ratio"),
            }
        )
    return out


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if finite(row.get(key)) is not None]


def mean_values(rows: list[dict[str, Any]], key: str) -> float | None:
    return mean(numeric_values(rows, key))


def median_values(rows: list[dict[str, Any]], key: str) -> float | None:
    return median(numeric_values(rows, key))


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
            return f"{value:.3e}"
        return f"{value:.5g}"
    return str(value)


def print_markdown(rows: list[dict[str, Any]], columns: list[str]) -> None:
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row.get(col)) for col in columns) + " |")


def print_csv(rows: list[dict[str, Any]], columns: list[str]) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=columns,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col) for col in columns})


def output_rows(rows: list[dict[str, Any]], columns: list[str], fmt: str) -> None:
    if fmt == "csv":
        print_csv(rows, columns)
        return
    if fmt == "jsonl":
        for row in rows:
            print(json.dumps(row, separators=(",", ":")))
        return
    print_markdown(rows, columns)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("_local/oracle_lab/runs")],
    )
    parser.add_argument("--glob", default="*.jsonl")
    parser.add_argument("--early-steps", type=int, default=200)
    parser.add_argument("--baseline-arm", default="hidden_gram_ns")
    parser.add_argument("--per-run", action="store_true")
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "jsonl"],
        default="markdown",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    files = discover(args.paths, args.glob)
    summaries = [summarize_file(path, args.early_steps) for path in files]
    summaries = [
        summary
        for summary in summaries
        if summary.evals or summary.best_val is not None
    ]
    baselines = baseline_by_seed(summaries, args.baseline_arm)
    rows = [
        run_row(summary, baselines.get((summary.matrix, summary.seed)))
        for summary in summaries
    ]

    if args.per_run:
        columns = [
            "matrix",
            "seed",
            "arm",
            "best_val",
            "delta_best",
            "train_tps",
            "delta_tps_pct",
            "early_rspec_hidden",
            "early_act_hidden",
            "line_ratio",
            "complete",
            "diverged",
            "run_name",
        ]
        output_rows(rows, columns, args.format)
        return 0

    aggregate = aggregate_rows(rows)
    columns = [
        "matrix",
        "arm",
        "runs",
        "complete",
        "incomplete",
        "diverged",
        "paired",
        "val_wins",
        "speed_wins",
        "best_val_mean",
        "delta_best_mean",
        "delta_best_sem",
        "delta_best_ci95_low",
        "delta_best_ci95_high",
        "train_tps_median",
        "delta_tps_pct_median",
        "early_rspec_hidden_median",
        "early_act_hidden_median",
        "line_ratio_median",
    ]
    output_rows(aggregate, columns, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
