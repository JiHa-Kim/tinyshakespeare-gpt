import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUN_RE = re.compile(r"^(?P<matrix>.+)_seed(?P<seed>\d+)_(?P<arm>.+)$")


@dataclass(frozen=True)
class EvalPoint:
    step: int
    opt_steps: int
    x: float
    val_loss: float
    train_loss: float | None = None


@dataclass(frozen=True)
class CurveFit:
    a: float
    b: float
    c: float
    fit_rmse: float
    fit_mae: float
    fit_points: int


@dataclass(frozen=True)
class PredictionAudit:
    run_name: str
    path: Path
    matrix: str
    seed: int | None
    arm: str
    points: int
    fit_points: int
    fit_start_x: float
    fit_end_x: float
    holdout_points: int
    a: float
    b: float
    c: float
    fit_rmse: float
    fit_mae: float
    holdout_rmse: float | None
    holdout_mae: float | None
    holdout_max_abs: float | None
    final_actual: float | None
    final_pred: float | None
    final_error: float | None
    final_abs_error: float | None
    final_pct_of_progress: float | None
    optimizer: str
    hidden_ulmo: str
    schedule: str


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def parse_run_name(run_name: str) -> tuple[str, int | None, str]:
    match = RUN_RE.match(run_name)
    if match is None:
        return "", None, run_name
    return match.group("matrix"), int(match.group("seed")), match.group("arm")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def discover(paths: list[Path], pattern: str) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob(pattern)))
        elif path.is_file():
            files.append(path)
    return files


def metadata(records: list[dict[str, Any]]) -> tuple[str, str, str]:
    optimizer = ""
    hidden_ulmo = ""
    schedule = ""
    for record in records:
        if record.get("event") != "config":
            continue
        opt = record.get("optimizer") or {}
        model = record.get("model") or {}
        sched = record.get("schedule") or {}
        args = record.get("args") or {}
        optimizer = str(opt.get("name") or "")
        hidden_ulmo = str(model.get("hidden_ulmo") or args.get("hidden_ulmo") or "")
        warmup = sched.get("warmup_steps")
        stable = sched.get("stable_steps")
        decay = sched.get("decay_steps")
        if isinstance(warmup, int) and isinstance(stable, int) and isinstance(decay, int):
            schedule = f"warmup={warmup},stable={stable},decay={decay}"
        break
    return optimizer, hidden_ulmo, schedule


def eval_points(records: list[dict[str, Any]], x_axis: str) -> list[EvalPoint]:
    count_increment = 1
    for record in records:
        if record.get("event") != "config":
            continue
        schedule = record.get("schedule") or {}
        raw = schedule.get("count_increment")
        if isinstance(raw, int) and raw > 0:
            count_increment = raw
        break

    points = []
    for record in records:
        if record.get("event") != "eval":
            continue
        val = finite(record.get("val_loss"))
        step = record.get("step")
        opt_steps = record.get("total_opt_steps")
        if val is None or not isinstance(step, int) or not isinstance(opt_steps, int):
            continue
        x = float(opt_steps if x_axis == "steps" else opt_steps * count_increment)
        points.append(
            EvalPoint(
                step=step,
                opt_steps=opt_steps,
                x=x,
                val_loss=val,
                train_loss=finite(record.get("train_loss")),
            )
        )
    return sorted(points, key=lambda point: point.x)


def predict(fit: CurveFit, x: float) -> float:
    return fit.a / math.sqrt(x + fit.b) + fit.c


def error_stats(actual: list[float], predicted: list[float]) -> tuple[float, float, float]:
    errors = [p - y for y, p in zip(actual, predicted, strict=True)]
    if not errors:
        return float("nan"), float("nan"), float("nan")
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    max_abs = max(abs(e) for e in errors)
    return rmse, mae, max_abs


def linear_fit_for_b(xs: list[float], ys: list[float], b: float) -> CurveFit | None:
    try:
        us = [1.0 / math.sqrt(x + b) for x in xs]
    except (ValueError, ZeroDivisionError):
        return None
    if not all(math.isfinite(u) for u in us):
        return None

    mean_u = sum(us) / len(us)
    mean_y = sum(ys) / len(ys)
    var_u = sum((u - mean_u) ** 2 for u in us)
    if var_u <= 0.0:
        return None

    cov = sum((u - mean_u) * (y - mean_y) for u, y in zip(us, ys, strict=True))
    a = cov / var_u
    c = mean_y - a * mean_u
    preds = [a * u + c for u in us]
    rmse, mae, _ = error_stats(ys, preds)
    return CurveFit(a=a, b=b, c=c, fit_rmse=rmse, fit_mae=mae, fit_points=len(xs))


def positive_candidates(span: float, grid_size: int) -> list[float]:
    span = max(span, 1.0)
    values = {0.0}
    lo = span * 1e-6
    hi = span * 100.0
    if grid_size <= 1:
        values.add(span)
    else:
        log_lo = math.log(lo)
        log_hi = math.log(hi)
        for i in range(grid_size):
            t = i / (grid_size - 1)
            values.add(math.exp((1.0 - t) * log_lo + t * log_hi))
    return sorted(values)


def b_candidates(xs: list[float], grid_size: int, allow_negative: bool) -> list[float]:
    min_x = min(xs)
    max_x = max(xs)
    span = max(max_x - min_x, max_x, 1.0)
    values = set(positive_candidates(span, grid_size))
    if allow_negative and min_x > 0.0:
        # Stay away from the singularity at x + b = 0.
        for i in range(1, grid_size + 1):
            ratio = 0.99 * i / grid_size
            values.add(-min_x * ratio)
    lower = -min_x + 1e-12 * span
    return sorted(b for b in values if b > lower)


def fit_inverse_sqrt(
    xs: list[float],
    ys: list[float],
    grid_size: int,
    allow_negative_b: bool,
) -> CurveFit:
    candidates = b_candidates(xs, grid_size, allow_negative_b)
    fits = [fit for b in candidates if (fit := linear_fit_for_b(xs, ys, b))]
    if not fits:
        raise ValueError("could not fit inverse-square-root curve")

    best = min(fits, key=lambda fit: fit.fit_rmse)
    best_index = fits.index(best)
    if 0 < best_index < len(fits) - 1:
        refined = golden_refine(xs, ys, fits[best_index - 1].b, fits[best_index + 1].b)
        if refined is not None and refined.fit_rmse < best.fit_rmse:
            best = refined
    return best


def golden_refine(
    xs: list[float],
    ys: list[float],
    lo: float,
    hi: float,
    iterations: int = 80,
) -> CurveFit | None:
    if not lo < hi:
        return None
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - phi * (hi - lo)
    d = lo + phi * (hi - lo)
    fit_c = linear_fit_for_b(xs, ys, c)
    fit_d = linear_fit_for_b(xs, ys, d)
    if fit_c is None or fit_d is None:
        return None

    for _ in range(iterations):
        if fit_c.fit_rmse <= fit_d.fit_rmse:
            hi = d
            d = c
            fit_d = fit_c
            c = hi - phi * (hi - lo)
            fit_c = linear_fit_for_b(xs, ys, c)
            if fit_c is None:
                return fit_d
        else:
            lo = c
            c = d
            fit_c = fit_d
            d = lo + phi * (hi - lo)
            fit_d = linear_fit_for_b(xs, ys, d)
            if fit_d is None:
                return fit_c
    return min((fit_c, fit_d), key=lambda fit: fit.fit_rmse)


def fit_window(
    points: list[EvalPoint],
    start_frac: float,
    end_frac: float,
    min_fit_points: int,
    start_x: float | None = None,
    end_x: float | None = None,
) -> tuple[list[EvalPoint], list[EvalPoint]]:
    if not points:
        return [], []
    min_seen = min(point.x for point in points)
    max_seen = max(point.x for point in points)
    span = max_seen - min_seen
    fit_start = min_seen + start_frac * span if start_x is None else start_x
    fit_end = min_seen + end_frac * span if end_x is None else end_x
    fit = [point for point in points if fit_start <= point.x <= fit_end]

    if len(fit) < min_fit_points:
        ordered = [point for point in points if point.x >= fit_start]
        fit = ordered[:min_fit_points]
        if not fit:
            return [], []
        fit_end = fit[-1].x

    holdout = [point for point in points if point.x > fit_end]
    return fit, holdout


def audit_file(
    path: Path,
    x_axis: str,
    fit_start_frac: float,
    fit_end_frac: float,
    min_fit_points: int,
    grid_size: int,
    allow_negative_b: bool,
    fit_start_x: float | None = None,
    fit_end_x: float | None = None,
) -> PredictionAudit | None:
    records = load_jsonl(path)
    points = eval_points(records, x_axis)
    fit_points, holdout = fit_window(
        points,
        fit_start_frac,
        fit_end_frac,
        min_fit_points,
        fit_start_x,
        fit_end_x,
    )
    if len(fit_points) < min_fit_points or not holdout:
        return None

    xs = [point.x for point in fit_points]
    ys = [point.val_loss for point in fit_points]
    fit = fit_inverse_sqrt(xs, ys, grid_size, allow_negative_b)

    holdout_actual = [point.val_loss for point in holdout]
    holdout_pred = [predict(fit, point.x) for point in holdout]
    holdout_rmse, holdout_mae, holdout_max_abs = error_stats(
        holdout_actual, holdout_pred
    )

    final = holdout[-1]
    final_pred = predict(fit, final.x)
    final_error = final_pred - final.val_loss
    initial = points[0].val_loss if points else None
    progress = None
    if initial is not None:
        total_progress = initial - final.val_loss
        if abs(total_progress) > 1e-12:
            progress = final_error / total_progress

    run_name = path.stem
    for record in records:
        raw_name = record.get("run_name")
        if raw_name:
            run_name = str(raw_name)
            break
    matrix, seed, arm = parse_run_name(run_name)
    optimizer, hidden_ulmo, schedule = metadata(records)

    return PredictionAudit(
        run_name=run_name,
        path=path,
        matrix=matrix,
        seed=seed,
        arm=arm,
        points=len(points),
        fit_points=len(fit_points),
        fit_start_x=fit_points[0].x,
        fit_end_x=fit_points[-1].x,
        holdout_points=len(holdout),
        a=fit.a,
        b=fit.b,
        c=fit.c,
        fit_rmse=fit.fit_rmse,
        fit_mae=fit.fit_mae,
        holdout_rmse=holdout_rmse,
        holdout_mae=holdout_mae,
        holdout_max_abs=holdout_max_abs,
        final_actual=final.val_loss,
        final_pred=final_pred,
        final_error=final_error,
        final_abs_error=abs(final_error),
        final_pct_of_progress=progress,
        optimizer=optimizer,
        hidden_ulmo=hidden_ulmo,
        schedule=schedule,
    )


def row(audit: PredictionAudit) -> dict[str, Any]:
    return {
        "matrix": audit.matrix,
        "seed": audit.seed,
        "arm": audit.arm,
        "run_name": audit.run_name,
        "optimizer": audit.optimizer,
        "hidden_ulmo": audit.hidden_ulmo,
        "schedule": audit.schedule,
        "points": audit.points,
        "fit_start_x": audit.fit_start_x,
        "fit_end_x": audit.fit_end_x,
        "fit_points": audit.fit_points,
        "holdout_points": audit.holdout_points,
        "fit_rmse": audit.fit_rmse,
        "holdout_rmse": audit.holdout_rmse,
        "holdout_mae": audit.holdout_mae,
        "holdout_max_abs": audit.holdout_max_abs,
        "final_actual": audit.final_actual,
        "final_pred": audit.final_pred,
        "final_error": audit.final_error,
        "final_abs_error": audit.final_abs_error,
        "final_pct_of_progress": audit.final_pct_of_progress,
        "a": audit.a,
        "b": audit.b,
        "c": audit.c,
        "path": str(audit.path),
    }


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


def output_rows(rows: list[dict[str, Any]], columns: list[str], fmt: str) -> None:
    if fmt == "jsonl":
        for item in rows:
            print(json.dumps(item, separators=(",", ":")))
        return
    if fmt == "csv":
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for item in rows:
            writer.writerow({column: item.get(column) for column in columns})
        return

    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for item in rows:
        print(
            "| "
            + " | ".join(format_value(item.get(column)) for column in columns)
            + " |"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("_local/oracle_lab/runs")],
    )
    parser.add_argument("--glob", default="*.jsonl")
    parser.add_argument("--x-axis", choices=["steps", "tokens"], default="steps")
    parser.add_argument("--fit-start-frac", type=float, default=0.05)
    parser.add_argument("--fit-end-frac", type=float, default=0.15)
    parser.add_argument(
        "--fit-start-x",
        type=float,
        default=None,
        help="absolute fit-window start on the chosen x-axis; overrides --fit-start-frac",
    )
    parser.add_argument(
        "--fit-end-x",
        type=float,
        default=None,
        help="absolute fit-window end on the chosen x-axis; overrides --fit-end-frac",
    )
    parser.add_argument("--min-fit-points", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument(
        "--allow-negative-b",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allow the denominator offset b to be negative if fit points remain valid",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "jsonl"],
        default="markdown",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.fit_start_x is None and args.fit_end_x is None and not (
        0.0 <= args.fit_start_frac < args.fit_end_frac <= 1.0
    ):
        raise ValueError("fit fractions must satisfy 0 <= start < end <= 1")
    if (args.fit_start_x is None) != (args.fit_end_x is None):
        raise ValueError("--fit-start-x and --fit-end-x must be provided together")
    if args.fit_start_x is not None and not args.fit_start_x < args.fit_end_x:
        raise ValueError("absolute fit x-window must satisfy start < end")
    if args.min_fit_points < 3:
        raise ValueError("--min-fit-points must be at least 3")

    audits = []
    for path in discover(args.paths, args.glob):
        audit = audit_file(
            path,
            args.x_axis,
            args.fit_start_frac,
            args.fit_end_frac,
            args.min_fit_points,
            args.grid_size,
            args.allow_negative_b,
            args.fit_start_x,
            args.fit_end_x,
        )
        if audit is not None:
            audits.append(audit)

    rows = [row(audit) for audit in audits]
    columns = [
        "run_name",
        "optimizer",
        "hidden_ulmo",
        "points",
        "fit_start_x",
        "fit_end_x",
        "holdout_points",
        "fit_rmse",
        "holdout_rmse",
        "holdout_mae",
        "holdout_max_abs",
        "final_actual",
        "final_pred",
        "final_error",
        "final_pct_of_progress",
        "c",
    ]
    output_rows(rows, columns, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
