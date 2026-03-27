import argparse
import copy
import csv
from pathlib import Path

from train_shakespeare import make_parser, train


def exp2_grid(exp2_min: float, exp2_max: float, step: float):
    if step <= 0:
        raise ValueError('step must be > 0')
    vals = []
    x = exp2_min
    while x <= exp2_max + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals


def dedupe_sorted(xs):
    out = []
    seen = set()
    for x in sorted(xs):
        k = round(x, 10)
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def has_flag(rest, *names):
    return any(name in rest for name in names)


def apply_proxy_defaults(base, rest, args):
    if args.no_proxy:
        return
    if not has_flag(rest, '--n-layer'):
        base.n_layer = args.proxy_n_layer
    if not has_flag(rest, '--n-head'):
        base.n_head = args.proxy_n_head
    if not has_flag(rest, '--d-model'):
        base.d_model = args.proxy_d_model
    if not has_flag(rest, '--max-iters'):
        base.max_iters = args.proxy_max_iters
    if not has_flag(rest, '--eval-interval'):
        base.eval_interval = args.proxy_eval_interval
    if not has_flag(rest, '--eval-iters'):
        base.eval_iters = args.proxy_eval_iters


def run_sweep(base, exp2s, stage_name: str, keep_checkpoints: bool):
    rows = []
    for i, exp2_lr in enumerate(exp2s):
        lr = 2.0 ** exp2_lr
        run = copy.deepcopy(base)
        run.lr = lr
        run.seed = base.seed + i
        out_path = Path(base.out_path)
        run.out_path = str(out_path.with_name(f'{out_path.stem}_{stage_name}_{i}_2p{exp2_lr:+.2f}{out_path.suffix}'))
        print(f'=== {stage_name} | 2**{exp2_lr:.2f} = {lr:.3e} ({i + 1}/{len(exp2s)}) ===')
        metrics = train(run)
        row = {
            'stage': stage_name,
            'exp2_lr': exp2_lr,
            'lr': lr,
            'best_val': metrics['best_val'],
            'final_val': metrics['final_val'],
            'final_train': metrics['final_train'],
            'compile_seconds': metrics['compile_seconds'],
            'n_layer': run.n_layer,
            'n_head': run.n_head,
            'd_model': run.d_model,
            'max_iters': run.max_iters,
            'out_path': run.out_path,
        }
        rows.append(row)
        print(
            f'>>> {stage_name} | 2**{exp2_lr:.2f} = {lr:.3e} '
            f'| best_val {metrics["best_val"]:.4f} | final_val {metrics["final_val"]:.4f}'
        )
        if not keep_checkpoints:
            Path(run.out_path).unlink(missing_ok=True)
    rows.sort(key=lambda x: x['best_val'])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp2-min', type=float, default=-14.0)
    p.add_argument('--exp2-max', type=float, default=-8.0)
    p.add_argument('--coarse-step', type=float, default=1.0)
    p.add_argument('--fine-step', type=float, default=0.25)
    p.add_argument('--fine-radius', type=float, default=1.0)
    p.add_argument('--csv-path', default='out/lr_sweep.csv')
    p.add_argument('--keep-checkpoints', action='store_true')
    p.add_argument('--no-proxy', action='store_true')
    p.add_argument('--proxy-n-layer', type=int, default=4)
    p.add_argument('--proxy-n-head', type=int, default=4)
    p.add_argument('--proxy-d-model', type=int, default=256)
    p.add_argument('--proxy-max-iters', type=int, default=600)
    p.add_argument('--proxy-eval-interval', type=int, default=50)
    p.add_argument('--proxy-eval-iters', type=int, default=20)
    args, rest = p.parse_known_args()

    base = make_parser().parse_args(rest)
    apply_proxy_defaults(base, rest, args)

    coarse_exp2s = exp2_grid(args.exp2_min, args.exp2_max, args.coarse_step)
    coarse_rows = run_sweep(base, coarse_exp2s, 'coarse', args.keep_checkpoints)
    best_exp2 = coarse_rows[0]['exp2_lr']

    fine_exp2s = exp2_grid(best_exp2 - args.fine_radius, best_exp2 + args.fine_radius, args.fine_step)
    fine_exp2s = [x for x in dedupe_sorted(fine_exp2s) if round(x, 10) not in {round(y, 10) for y in coarse_exp2s}]
    fine_rows = run_sweep(base, fine_exp2s, 'fine', args.keep_checkpoints) if fine_exp2s else []

    rows = coarse_rows + fine_rows
    rows.sort(key=lambda x: x['best_val'])

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print('\n=== top runs ===')
    for row in rows[:min(5, len(rows))]:
        print(
            f'{row["stage"]:>6s} | 2**{row["exp2_lr"]:.2f} = {row["lr"]:.3e} '
            f'| best_val {row["best_val"]:.4f} | final_val {row["final_val"]:.4f}'
        )
    print(f'best coarse exponent: {best_exp2:.2f}')
    print(f'csv_path {csv_path}')


if __name__ == '__main__':
    main()
