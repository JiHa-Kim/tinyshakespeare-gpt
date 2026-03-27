# Minimal Scion LR-tuning repo

A small reference repo for vanilla Scion on tiny Shakespeare, with a tuning path that is much closer to the Scion paper:

- `lionk_ccwd.py`: Lion-K core with corrected decoupled decay support
- `scion.py`: Scion specialization, LMOs, geometry-matched initialization, and LR transfer helper
- `gpt.py` + `train_shakespeare.py`: a minimal GPT training loop
- `tune_lr.py`: proxy-model LR sweep with optional multi-seed aggregation

## What changed

The important fixes are:

1. `scion_transfer_lr(...)` is implemented and used by the sweep script.
2. Vanilla `scion` is the default everywhere.
3. Initialization now matches the optimizer geometry instead of using generic orthogonal or unit-norm init.
   - embeddings: ColNorm with transpose semantics and radius scaling
   - hidden matrices: spectral init with the same shape-dependent scaling used by the spectral LMO
   - output head: RowNorm with radius scaling
4. The stable-decay scheduler is now exact:
   - no accidental one-step decay when `decay_frac = 0`
   - the last decay step reaches `min_lr` exactly
5. LR tuning is more robust:
   - same seed across LR candidates by default
   - optional multi-seed confirmation with aggregated median metrics
   - recommendation uses the largest stable LR within a small final-loss tolerance, instead of blindly taking the largest stable point

## Scion-first tuning policy

Start with plain Scion:

- `--optimizer scion`
- `--warmup-frac 0.0`
- `--min-lr 0.0`
- `--phi 0.0`
- `--grad-clip 0.0`

Keep the radii fixed while tuning LR:

- `--rho-embed 1`
- `--rho-hidden 3`
- `--rho-out 10`

Tune `--prenorm rmsnorm` and `--prenorm rmsball` separately.

## Default schedule

The default schedule is stable-decay, which is just WSD with zero warmup:

- `warmup_frac = 0.0`
- `decay_frac = 0.285`
- `min_lr = 0.0`

That mirrors the Scion paper's nanoGPT setup more closely than the previous 20 percent tail.

## Install

```bash
pip install torch
```

## Train a single run

```bash
python train_shakespeare.py \
  --mode train \
  --optimizer scion \
  --prenorm rmsnorm \
  --batch-size 16 --grad-accum 4 --block-size 256 \
  --n-layer 6 --n-head 6 --d-model 384 \
  --lr 1e-3 \
  --warmup-frac 0.0 --decay-frac 0.285 --min-lr 0.0 \
  --beta2 0.95 --phi 0.0 \
  --rho-embed 1 --rho-hidden 3 --rho-out 10 \
  --no-compile
```

For the RMS-ball ablation:

```bash
python train_shakespeare.py --mode train --optimizer scion --prenorm rmsball --no-compile
```

## LR sweep on a proxy model

The sweep works in base-2 LR space. It does a coarse sweep, then a fine sweep around the best coarse center.

Recommended first pass:

```bash
python tune_lr.py \
  --prenorm both \
  --exp2-min -14 --exp2-max -8 \
  --coarse-step 1.0 \
  --fine-radius 1.0 --fine-step 0.25 \
  --num-seeds 1 \
  --batch-size 16 --grad-accum 4 --block-size 256 \
  --warmup-frac 0.0 --decay-frac 0.285 --min-lr 0.0 \
  --optimizer scion \
  --rho-embed 1 --rho-hidden 3 --rho-out 10
```

For a more robust confirmation pass:

```bash
python tune_lr.py \
  --prenorm both \
  --exp2-min -14 --exp2-max -8 \
  --coarse-step 1.0 \
  --fine-radius 1.0 --fine-step 0.25 \
  --num-seeds 3 --seed-stride 1000 \
  --stable-threshold 1.0 \
  --final-val-tol 0.03 \
  --batch-size 16 --grad-accum 4 --block-size 256 \
  --warmup-frac 0.0 --decay-frac 0.285 --min-lr 0.0 \
  --optimizer scion \
  --rho-embed 1 --rho-hidden 3 --rho-out 10
```

This writes:

- `out/lr_sweep.csv`: one row per trial
- `out/lr_sweep_summary.csv`: one row per LR candidate after aggregation

## How the recommendation is chosen

The sweep now recommends:

1. candidates whose stability rate is at least `--stable-threshold`
2. among those, candidates whose median final validation loss is within `--final-val-tol` of the best stable median final validation loss
3. the largest LR in that shortlist

That keeps the Scion/WSD bias toward large stable LRs, but avoids picking a clearly worse point just because it is barely stable.

## Proxy-to-target LR transfer

The script also prints transferred target LRs using:

- token multiplier `mT`
- depth multiplier `mL`
- transfer exponent `alpha`

Because the trainer currently uses a single global LR, the practical starting point for the full model is the transferred hidden LR.

## File guide

### `scion.py`

- `ColNormLMO`, `RowNormLMO`, `SpectralLMO`
- `init_colnorm_`, `init_rownorm_`, `init_spectral_`
- `scion_transfer_lr(...)`
- `Scion`, `ScionC`

### `train_shakespeare.py`

- geometry-matched Scion initialization
- exact stable-decay schedule
- grouped Scion optimizer construction
- training, checkpointing, and sampling

### `tune_lr.py`

- coarse + fine base-2 LR sweep
- proxy defaults
- optional multi-seed aggregation
- CSV export for both raw trials and aggregated summaries

## Current default recommendation

Until you have evidence otherwise, use:

- `optimizer = scion`
- `warmup = 0`
- `decay_frac = 0.285`
- fixed radii
- LR tuned on a proxy model
- transferred hidden LR as the first full-model LR
