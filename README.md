# Tiny Shakespeare GPT

A compact Tiny Shakespeare GPT research sandbox for ScionH/Hyperball training
and Schedule-Free fixed-RMS variants.

- `scionh/optim/`: Hyperball, Schedule-Free Hyperball, and schedule helpers.
- `scionh/ulmos/`: ULMOs, Gram-NS, SWAN, and streaming-SVD helpers.
- `scionh/models/`: compact GPT model and architecture switches.
- `scionh/probes/`: focused diagnostics and training-step performance probes.
- `scionh/experiments/`: prediction audits and registered experiment helpers.
- `scionh/train_shakespeare.py`: training entrypoint.

## Active Recipe

Each controlled weight block lives on a fixed-radius RMS sphere.
Initialization sets each block radius. Group-specific learning rates set the
Euclidean step taken before retraction, and a separate state half-life sets
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

The sign is positive because local ULMOs return descent directions.

## Schedule-Free ScionH

`--schedule-free` switches to a Schedule-Free ScionH variant. It keeps the
paper's three-sequence structure and replaces AdamC's raw iterate update with
the ScionH/Hyperball retraction:

- `z`: raw iterate updated by the ScionH ULMO atom and Hyperball retraction.
- `x`: the Schedule-Free average of `z`; this is used for evaluation and
  checkpoint output.
- model weights during training: the Schedule-Free query `y`.

Use `--sf-geometry ambient` for the paper-style linear `x/y` recurrence with a
fixed-RMS raw `z`. Use `--sf-geometry geodesic` for the fixed-RMS manifold
variant where `x` and `y` are spherical interpolants and all three sequences
stay on the RMS sphere. See `docs/schedulefree_scionh.md` for the derivation.

The inverse-square-root prediction audit fits
`val_loss = a / sqrt(t + b) + c` on eval points and reports held-out error:

```bash
uv run python -m scionh.experiments.prediction_audit _local/prediction_probe
```

For short Tiny Shakespeare runs, fit after the early transient rather than
blindly trusting the paper's 5%-to-15% window:

```bash
uv run python -m scionh.experiments.prediction_audit \
  _local/prediction_probe/sf_scionh.jsonl \
  --fit-start-x 480 --fit-end-x 800
```

## Defaults

- optimizer: Hyperball, or Schedule-Free Hyperball with `--schedule-free`
- hidden ULMO: Gram Newton-Schulz
- input/output ULMOs: untied ColNorm + Sign; tied Sign + Sign
- batch size: 64
- gradient accumulation: 1
- block size: 256
- initialization RMS: embedding 0.70, hidden 0.051, output 0.022
- state half-life: about 2.21e5 processed tokens
- peak pre-retraction learning rate: embedding about 3.68%, hidden about 2.56%,
  output about 0.35%
- WSD schedule: 100 warmup steps, stable phase, 15% decay by default; schedule
  free runs default to no decay unless `--decay-iters` is explicit
- update rule: `retract` by default; `--hyperball-update slerp` runs the
  tangent-projected geodesic comparison

## Hidden ULMOs

`--hidden-ulmo gram-ns` is the default. It uses the Gram Newton-Schulz form with
the four-moment spectral upper-bound normalization that reuses the first `G @ G`
product already needed by the polynomial iteration.

`--hidden-ulmo swan` applies shape-aware GradNorm before the same Gram
Newton-Schulz whitening step: row GradNorm for wide matrices and col GradNorm
for tall matrices.

`--hidden-ulmo streaming-svd` keeps a per-parameter cached right-singular basis
and applies one or more streaming subspace steps per optimizer update.

Additional hidden-matrix arms remain available for controlled geometry checks:
`frobenius`, `colnorm`, `rownorm`, `sign`, `svd`, `blockwise-gram-ns`, and
`blockwise-svd`.

## Commands

Schedule-Free fixed-RMS ScionH:

```bash
uv run python -m scionh.train_shakespeare \
  --mode train --schedule-free --sf-geometry geodesic \
  --compile --compile-mode reduce-overhead \
  --warmup-iters 0 --sf-c-warmup-iters 0 --decay-iters 0 \
  --sf-beta 0.9 --sf-r 0 \
  --metrics-jsonl _local/prediction_probe/sf_scionh_geodesic.jsonl \
  --no-save --skip-sample
```

Plain Hyperball baseline:

```bash
uv run python -m scionh.train_shakespeare \
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

Evaluate a saved checkpoint:

```bash
uv run python -m scionh.train_shakespeare \
  --mode eval \
  --device cuda \
  --out-path out/hyperball_2k.pt \
  --eval-iters 200
```
