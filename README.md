# Minimal ScionC Repo

A compact ScionC sandbox organized by category:

- `scionc/optim/`: Hyperball optimizer and schedule parametrization helpers.
- `scionc/ulmos/`: ULMOs, Gram-NS, and streaming SVD helpers.
- `scionc/models/`: compact GPT model and tiny Shakespeare data utilities.
- `scionc/probes/`: convergence, line, and optimizer-step stats probes.
- `scionc/train_shakespeare.py`: training entrypoint.

## Active Recipe: Hyperball

Each controlled weight block lives on a fixed-radius RMS sphere.
Initialization sets each block radius. Group-specific learning rates set the
Euclidean step taken before retraction. A separate state half-life sets
momentum retention.

One optimizer update advances the count by
`batch_size * block_size * grad_accum` processed tokens. For one controlled
block:

```math
M \leftarrow \beta\,M + (1-\beta)\,G,
\qquad
V = \operatorname{ulmo}(M),
\qquad
\hat W' = \operatorname{rmsnorm}\!\left(
  \hat W + \eta_t\,\operatorname{rmsnorm}(V)
\right).
```

The sign is positive because local ULMOs return descent directions. This is
the usual Hyperball `W <- RMSNorm(W - lr RMSNorm(V))` update when `V`
denotes a gradient-like atom.

The active coordinates are:

- state half-life $h_\beta$ (processed tokens),
- per-update state retention $\beta = 2^{-\Delta\tau/h_\beta}$,
- scheduled pre-retraction learning rate $\eta_t$,
- frozen block radius $R = \|W_0\|_{\mathrm{rms}}$.

## Defaults

- optimizer: Hyperball
- hidden ULMO: Gram Newton-Schulz
- input/output ULMOs: untied ColNorm + Sign; tied Sign + Sign
- batch size: 64
- gradient accumulation: 1
- block size: 256
- initialization RMS: embedding 0.70, hidden 0.051, output 0.022
- state half-life: about 2.21e5 processed tokens
- peak pre-retraction learning rate: embedding about 3.68%, hidden about 2.56%,
  output about 0.35%
- WSD schedule: 100 warmup steps, stable phase, 15% decay (floor=0);
  the schedule scales the learning rate directly
- update rule: `retract` by default; `--hyperball-update slerp` runs the
  tangent-projected geodesic comparison

## Hidden ULMOs

`--hidden-ulmo gram-ns` is the default. It uses the Gram Newton-Schulz form with
the four-moment spectral upper-bound normalization that reuses the first `G @ G`
product already needed by the polynomial iteration.

`--hidden-ulmo streaming-svd` keeps a per-parameter cached right-singular basis
and applies one or more streaming subspace steps per optimizer update.

## Recommended Command

```bash
uv run python -m scionc.train_shakespeare \
  --mode train \
  --out-path out/hyperball_2k.pt \
  --sample-out out/hyperball_2k_samples.md \
  --batch-size 64 --grad-accum 1 --block-size 256 \
  --n-layer 6 --n-head 6 --d-model 384 \
  --schedule-floor 0 \
  --hyperball-update retract \
  --warmup-iters 100 --decay-frac 0.15 \
  --state-half-life 2.214e5 \
  --hidden-ulmo gram-ns \
  --embed-ulmo colnorm --out-ulmo sign \
  --temperature 0.8 --top-k 40 --sample-count 2
```

Evaluate the saved checkpoint with more batches:

```bash
uv run python -m scionc.train_shakespeare \
  --mode eval \
  --device cuda \
  --out-path out/hyperball_2k.pt \
  --eval-iters 200
```
