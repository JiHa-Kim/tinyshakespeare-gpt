import argparse
import copy
import csv
import math
from pathlib import Path

from scion import scion_transfer_lr
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


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float('nan')
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


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
    if not has_flag(rest, '--compile', '--no-compile'):
        base.compile = False


def token_budget(cfg):
    return cfg.max_iters * cfg.batch_size * cfg.grad_accum * cfg.block_size


def transfer_lrs(proxy_lr: float, proxy_cfg, target_cfg, alpha: float):
    mT = token_budget(target_cfg) / max(token_budget(proxy_cfg), 1)
    mL = target_cfg.n_layer / max(proxy_cfg.n_layer, 1)
    per_group = scion_transfer_lr(proxy_lr, mT=mT, mL=mL, alpha=alpha)
    return mT, mL, per_group


def stable_trial(metrics) -> bool:
    return (not metrics['diverged']) and math.isfinite(metrics['final_val'])


def aggregate_trials(rows):
    finals = [r['final_val'] for r in rows if math.isfinite(r['final_val'])]
    bests = [r['best_val'] for r in rows if math.isfinite(r['best_val'])]
    initials = [r['initial_val'] for r in rows if math.isfinite(r['initial_val'])]
    stable_count = sum(1 for r in rows if r['stable'])
    summary = dict(rows[0])
    summary.update(
        {
            'trial_count': len(rows),
            'stable_trials': stable_count,
            'stability_rate': stable_count / len(rows),
            'median_final_val': median(finals),
            'median_best_val': median(bests),
            'median_initial_val': median(initials),
            'all_diverge_reasons': '|'.join(sorted({r['diverge_reason'] for r in rows if r['diverge_reason']})),
        }
    )
    return summary


def run_candidate(base, exp2_lr, stage_name, keep_checkpoints, prenorm, proxy_cfg, target_cfg, alpha, args):
    lr = 2.0**exp2_lr
    mT, mL, per_group = transfer_lrs(lr, proxy_cfg, target_cfg, alpha)
    out_path = Path(base.out_path)
    raw_rows = []

    for seed_idx in range(args.num_seeds):
        run = copy.deepcopy(base)
        run.lr = lr
        run.prenorm = prenorm
        run.seed = base.seed + seed_idx * args.seed_stride
        run.skip_sample = True
        run.no_save = not keep_checkpoints
        run.out_path = str(
            out_path.with_name(
                f'{out_path.stem}_{prenorm}_{stage_name}_{seed_idx}_2p{exp2_lr:+.2f}{out_path.suffix}'
            )
        )
        print(
            f'=== {prenorm} | {stage_name} | seed {run.seed} | 2**{exp2_lr:.2f} = {lr:.3e} '
            f'({seed_idx + 1}/{args.num_seeds}) ==='
        )
        metrics = train(run)
        raw_rows.append(
            {
                'prenorm': prenorm,
                'stage': stage_name,
                'exp2_lr': exp2_lr,
                'lr': lr,
                'stable': stable_trial(metrics),
                'diverged': metrics['diverged'],
                'diverge_reason': metrics['diverge_reason'],
                'initial_val': metrics['initial_val'],
                'best_val': metrics['best_val'],
                'final_val': metrics['final_val'],
                'final_train': metrics['final_train'],
                'max_val': metrics['max_val'],
                'compile_seconds': metrics['compile_seconds'],
                'warmup_steps': metrics['warmup_steps'],
                'stable_steps': metrics['stable_steps'],
                'decay_steps': metrics['decay_steps'],
                'proxy_n_layer': proxy_cfg.n_layer,
                'proxy_n_head': proxy_cfg.n_head,
                'proxy_d_model': proxy_cfg.d_model,
                'proxy_max_iters': proxy_cfg.max_iters,
                'target_n_layer': target_cfg.n_layer,
                'target_n_head': target_cfg.n_head,
                'target_d_model': target_cfg.d_model,
                'target_max_iters': target_cfg.max_iters,
                'rho_embed': getattr(target_cfg, 'rho_embed', float('nan')),
                'rho_hidden': getattr(target_cfg, 'rho_hidden', float('nan')),
                'rho_out': getattr(target_cfg, 'rho_out', float('nan')),
                'mT': mT,
                'mL': mL,
                'target_embed_lr': per_group['embed'],
                'target_hidden_lr': per_group['hidden'],
                'target_out_lr': per_group['out'],
                'out_path': run.out_path,
                'seed': run.seed,
            }
        )

    summary = aggregate_trials(raw_rows)
    status = f"stable {summary['stable_trials']}/{summary['trial_count']}"
    print(
        f">>> {prenorm} | {stage_name} | 2**{exp2_lr:.2f} = {lr:.3e} | {status} | "
        f"median_final_val {summary['median_final_val']:.4f} | median_best_val {summary['median_best_val']:.4f} | "
        f"target_hidden_lr {summary['target_hidden_lr']:.3e}"
    )
    return raw_rows, summary


def shortlist_by_quality(rows, final_val_tol: float):
    best = min(r['median_final_val'] for r in rows if math.isfinite(r['median_final_val']))
    cutoff = best * (1.0 + final_val_tol)
    return [r for r in rows if math.isfinite(r['median_final_val']) and r['median_final_val'] <= cutoff]



def choose_recommendation(rows, stable_threshold: float, final_val_tol: float):
    eligible = [r for r in rows if r['stability_rate'] >= stable_threshold and math.isfinite(r['median_final_val'])]
    if eligible:
        shortlist = shortlist_by_quality(eligible, final_val_tol)
        shortlist.sort(key=lambda r: (-r['exp2_lr'], r['median_final_val']))
        return shortlist[0], 'largest-stable-within-final-val-tol'
    finite = [r for r in rows if math.isfinite(r['median_final_val'])]
    finite.sort(key=lambda r: (r['median_final_val'], r['exp2_lr']))
    return (finite[0] if finite else rows[0]), 'fallback-best-median-final-val'



def sweep_prenorm(base, target_cfg, args, coarse_exp2s, keep_checkpoints, prenorm):
    proxy_cfg = copy.deepcopy(base)
    raw_rows = []
    coarse_summaries = []
    for exp2_lr in coarse_exp2s:
        raw, summary = run_candidate(
            proxy_cfg,
            exp2_lr,
            'coarse',
            keep_checkpoints,
            prenorm,
            proxy_cfg,
            target_cfg,
            args.alpha_transfer,
            args,
        )
        raw_rows.extend(raw)
        coarse_summaries.append(summary)

    coarse_rec, _ = choose_recommendation(coarse_summaries, args.stable_threshold, args.final_val_tol)
    center = coarse_rec['exp2_lr']
    fine_exp2s = exp2_grid(center - args.fine_radius, center + args.fine_radius, args.fine_step)
    fine_exp2s = [
        x
        for x in dedupe_sorted(fine_exp2s)
        if round(x, 10) not in {round(y, 10) for y in coarse_exp2s}
    ]

    fine_summaries = []
    for exp2_lr in fine_exp2s:
        raw, summary = run_candidate(
            proxy_cfg,
            exp2_lr,
            'fine',
            keep_checkpoints,
            prenorm,
            proxy_cfg,
            target_cfg,
            args.alpha_transfer,
            args,
        )
        raw_rows.extend(raw)
        fine_summaries.append(summary)

    summaries = coarse_summaries + fine_summaries
    rec, mode = choose_recommendation(summaries, args.stable_threshold, args.final_val_tol)
    return raw_rows, summaries, rec, center, mode



def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



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
    p.add_argument('--alpha-transfer', type=float, default=0.5)
    p.add_argument('--prenorm', choices=['rmsnorm', 'rmsball', 'both'], default='both')
    p.add_argument('--num-seeds', type=int, default=1)
    p.add_argument('--seed-stride', type=int, default=1000)
    p.add_argument('--stable-threshold', type=float, default=1.0)
    p.add_argument('--final-val-tol', type=float, default=0.03)
    args, rest = p.parse_known_args()

    if args.num_seeds <= 0:
        raise ValueError('--num-seeds must be > 0')
    if not (0.0 <= args.stable_threshold <= 1.0):
        raise ValueError('--stable-threshold must lie in [0, 1]')
    if args.final_val_tol < 0.0:
        raise ValueError('--final-val-tol must be >= 0')

    target_cfg = make_parser().parse_args(rest)
    base = copy.deepcopy(target_cfg)
    apply_proxy_defaults(base, rest, args)

    coarse_exp2s = exp2_grid(args.exp2_min, args.exp2_max, args.coarse_step)
    all_raw_rows = []
    all_summaries = []
    recommendations = []
    prenorms = ['rmsnorm', 'rmsball'] if args.prenorm == 'both' else [args.prenorm]
    for prenorm in prenorms:
        raw_rows, summaries, rec, center, mode = sweep_prenorm(
            base, target_cfg, args, coarse_exp2s, args.keep_checkpoints, prenorm
        )
        all_raw_rows.extend(raw_rows)
        all_summaries.extend(summaries)
        recommendations.append((prenorm, rec, center, mode))

    csv_path = Path(args.csv_path)
    summary_path = csv_path.with_name(f'{csv_path.stem}_summary{csv_path.suffix}')
    write_csv(csv_path, all_raw_rows)
    write_csv(summary_path, all_summaries)

    print('\n=== recommendations ===')
    for prenorm, rec, center, mode in recommendations:
        print(
            f"{prenorm:>7s} | center 2**{center:.2f} | proxy_lr 2**{rec['exp2_lr']:.2f} = {rec['lr']:.3e} "
            f"| {mode} | stability {rec['stable_trials']}/{rec['trial_count']} | "
            f"median_final_val {rec['median_final_val']:.4f} | target_hidden_lr {rec['target_hidden_lr']:.3e} "
            f"| target_embed_lr {rec['target_embed_lr']:.3e} | target_out_lr {rec['target_out_lr']:.3e}"
        )
    print(f'csv_path {csv_path}')
    print(f'summary_csv_path {summary_path}')


if __name__ == '__main__':
    main()
