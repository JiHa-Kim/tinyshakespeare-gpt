# Minimal ScionC Repo

A compact ScionC sandbox organized by category:

- `scionc/optim/`: RMS-Sphere optimizer and schedule parametrization helpers.
- `scionc/ulmos/`: ULMOs, Gram-NS, and streaming SVD helpers.
- `scionc/models/`: compact GPT model and tiny Shakespeare data utilities.
- `scionc/probes/`: convergence, line, and optimizer-step stats probes.
- `scionc/train_shakespeare.py`: training entrypoint.

## Active Recipe: RMS-Sphere

Each controlled weight block lives on a fixed-radius RMS sphere.
Initialization sets each block radius. A single direction-retention
half-life sets the angular movement. Weight decay is the exact radial
Lagrange correction, not a manually scheduled penalty.

One optimizer update advances the count by
`batch_size * block_size * grad_accum` processed tokens. For one controlled
block:

```math
M \leftarrow q\,M + (1-q)\,G,
\qquad
V = \operatorname{ulmo}(M),
\qquad
\hat W' = q\,\hat W + \sqrt{1-q^2}\,U,
```

where $U$ is the unit tangent direction obtained by projecting the ULMO
atom $V$ onto the tangent space and normalizing.

The active coordinates are:

- direction half-life $h$ (processed tokens),
- per-update retention $q = 2^{-\Delta\tau/h}$,
- frozen block radius $R = \|W_0\|_{\mathrm{rms}}$.

## Defaults

- optimizer: RMS-Sphere
- hidden ULMO: Gram Newton-Schulz
- input/output ULMOs: untied ColNorm + Sign; tied Sign + Sign
- batch size: 64
- gradient accumulation: 1
- block size: 256
- initialization RMS: embedding 0.70, hidden 0.051, output 0.022
- direction half-life: about 2.21e5 processed tokens
- WSD schedule: 100 warmup steps, stable phase, 15% decay (floor=0)

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
  --out-path out/rms_sphere_2k.pt \
  --sample-out out/rms_sphere_2k_samples.md \
  --batch-size 64 --grad-accum 1 --block-size 256 \
  --n-layer 6 --n-head 6 --d-model 384 \
  --direction-half-life 2.214e5 \
  --schedule-floor 0 \
  --warmup-iters 100 --decay-frac 0.15 \
  --hidden-ulmo gram-ns \
  --embed-ulmo colnorm --out-ulmo sign \
  --temperature 0.8 --top-k 40 --sample-count 2
```

Evaluate the saved checkpoint with more batches:

```bash
uv run python -m scionc.train_shakespeare \
  --mode eval \
  --device cuda \
  --out-path out/rms_sphere_2k.pt \
  --eval-iters 200
```
