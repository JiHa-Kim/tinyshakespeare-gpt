# Minimal Lion-K / ScionC repo

A tiny reference implementation with three pieces:

- `lionk_ccwd.py`: general Lion-K with corrected cautious weight decay and primal averaging
- `scionc.py`: ScionC specialization, LMOs, init helpers, and simple transfer helper
- `gpt.py` + `train_shakespeare.py`: modern minimal GPT for tiny Shakespeare

Model choices:

- no bias
- gainless pre-norm blocks with an explicit ablation: `rmsnorm` or `rmsball`
- RoPE attention
- RMS-ball projection on `q` and `k`
- SwiGLU MLP
- untied embeddings / output head
- output head always uses row norm
- ScionC defaults to primal averaging OFF (`phi = 0.0`)

Files:

- `lionk_ccwd.py`: Lion-K core with optional corrected decoupled decay
- `scion.py`: Scion LMOs, geometry-matched initialization helpers, LR transfer helper, and workspace reuse in the spectral LMO
- `gpt.py` + `train_shakespeare.py`: a small GPT training loop for tiny Shakespeare, with val-only evals during training and best-or-final checkpointing

## Main policy

Default optimizer settings:

- optimizer: `scionc`
- warmup: `0`
- min LR: `0`

Tune separately for:

- `--prenorm rmsnorm`
- `--prenorm rmsball`

## Geometry-matched Scion init

Initialization matches the Scion optimizer geometry instead of using generic unscaled init:

- token embedding: column-normalized init on the transposed embedding matrix, with `rho_embed`
- hidden matrices: spectral / semi-orthogonal init with the same dimension-aware scaling used by the spectral LMO, with `rho_hidden`
- output head: row-normalized init with `rho_out`

## Exact single-run schedule

`train_shakespeare.py` now uses an exact schedule:

- warmup steps are explicit
- stable phase length is explicit
- if `decay_frac = 0`, there is no accidental one-step decay
- the last decay step reaches `min_lr` exactly

Default single-run decay fraction:

- `--decay-frac 0.285`

## Recommended commands

### Train a single run

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
