# Minimal Lion-K / ScionC repo

A tiny reference implementation with four pieces:

- `lionk_ccwd.py`: general Lion-K with corrected cautious weight decay and primal averaging
- `scionc.py`: ScionC specialization, LMOs, init helpers, and simple transfer helper
- `gpt.py` + `train_shakespeare.py`: modern minimal GPT for tiny Shakespeare
- `tune_lr.py`: minimal LR sweep script

Model choices:

- no bias
- gainless pre-norm blocks with an explicit ablation: `rmsnorm` or `rmsball`
- RoPE attention
- RMS-ball projection on `q` and `k`
- SwiGLU MLP
- untied embeddings / output head
- output head always uses row norm
- ScionC defaults to primal averaging OFF (`phi = 0.0`)
- ScionC defaults to no gradient clipping (`--grad-clip 0.0`)

## Tuning policy

Tune a single global learning rate first:

- `--lr`

Keep the Scion radii fixed at the paper-style GPT defaults:

- `--rho-hidden 50`
- `--rho-out 3000`

These radii live inside the LMOs and act like built-in per-group step scales, so fixing them and tuning only `lr` is the cleanest first pass.

## Schedule policy

Default schedule is warmup-stable-decay (`--schedule wsd`) with:

- `--warmup-frac 0.0`
- `--decay-frac 0.2`
- `--min-lr 0.0`

That means:

- no warmup by default for ScionC
- hold the peak LR constant for the first `80%` of training
- linearly decay over the last `20%`

## What is optimized here

This repo keeps the code minimal while pushing the hot path as far as possible without adding a lot of machinery:

- whole Shakespeare train/val tensors are moved to the target device once
- batches are drawn with vectorized indexing on-device
- `scaled_dot_product_attention` is used directly
- TF32 is enabled on CUDA
- autocast uses `bfloat16` when available
- optional `torch.compile`
- compile time is logged separately from actual training throughput
- gradient accumulation is built in via `--grad-accum`

## Install

```bash
pip install torch
```

## Train

```bash
python train_shakespeare.py --mode train --compile
```

Useful flags:

```bash
python train_shakespeare.py \
  --mode train \
  --n-layer 6 --n-head 6 --d-model 384 \
  --batch-size 16 --grad-accum 4 --block-size 256 \
  --lr 1e-3 \
  --schedule wsd --warmup-frac 0.0 --decay-frac 0.2 --min-lr 0.0 \
  --beta2 0.95 --phi 0.0 \
  --rho-hidden 50 --rho-out 3000 \
  --prenorm rmsnorm \
  --compile
```

This example uses a microbatch of `16` and gradient accumulation of `4`, so the effective batch is:

```text
16 * 4 = 64
```

RMS-ball pre-norm ablation:

```bash
python train_shakespeare.py --mode train --prenorm rmsball
```

To turn primal averaging on explicitly:

```bash
python train_shakespeare.py --mode train --phi 1.0
```

## LR sweep

The tuning script now uses a two-stage base-2 sweep:

1. coarse sweep over integer-ish `log2(lr)` values
2. fine sweep around the best coarse exponent

So the main tuning space is:

- `--exp2-min`
- `--exp2-max`
- `--coarse-step`
- `--fine-radius`
- `--fine-step`

Example:

```bash
python tune_lr.py \
  --exp2-min -14 --exp2-max -8 \
  --coarse-step 1.0 \
  --fine-radius 1.0 --fine-step 0.25 \
  --csv-path out/lr_sweep.csv \
  --mode train --max-iters 400 --eval-interval 50 --eval-iters 20 \
  --batch-size 16 --grad-accum 4 --block-size 256 \
  --schedule wsd --warmup-frac 0.0 --decay-frac 0.2 \
  --compile
```

This first tries:

```text
2**(-14), 2**(-13), ..., 2**(-8)
```

then refines near the best exponent, for example with quarter-step resolution:

```text
2**(e* - 1.0), 2**(e* - 0.75), ..., 2**(e* + 1.0)
```

The script writes a CSV and prints the top runs ranked by best validation loss.

## Logging

When `--compile` is enabled, the script logs:

- `compile_seconds`: one-time compile and first forward/backward warmup cost
- `train_seconds`: elapsed training time excluding compile
- `tok/s`: throughput excluding compile

## Sample

```bash
python train_shakespeare.py --mode sample --out-path out/scionc_shakespeare.pt --prompt "To be, or not to be"
```

## File guide

### `lionk_ccwd.py`

- `LionKCCWDPA`: general optimizer core
- `corrected_eta(...)`: corrected multiplicative decay helper

### `scionc.py`

- LMOs with explicit radii: `ColNormLMO`, `SpectralLMO`, `RowNormLMO`, `RMSLMO`
- init helpers: `init_colnorm_`, `init_rownorm_`, `init_semiorthogonal_`
- transfer helper: `scion_transfer_lr(...)`
- optimizer specialization: `ScionC`

### `gpt.py`

- GPT model
- gainless `Norm(kind='rmsnorm'|'rmsball')`
- tiny Shakespeare downloader
- on-device vectorized dataset batching

### `train_shakespeare.py`

- Scion init and parameter grouping
- one-global-lr tuning interface
- WSD / cosine / linear / constant schedules
- explicit radii flags for hidden/output LMOs
- gradient accumulation
- compile-time warmup and separate logging
- train loop
- checkpoint save/load
- sampling

### `tune_lr.py`

- geometric LR sweep
- CSV export
- ranking by best validation loss
