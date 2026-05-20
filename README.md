# Tiny Shakespeare GPT

A compact Tiny Shakespeare GPT research sandbox organized by category:

- `scionh/optim/`: Hyperball, SODA-Hyperball, and schedule parametrization helpers.
- `scionh/ulmos/`: ULMOs, Gram-NS, and streaming SVD helpers.
- `scionh/models/`: compact GPT model and tiny Shakespeare data utilities.
- `scionh/probes/`: convergence, line, and optimizer-step stats probes.
- `scionh/experiments/`: command generators for registered experiment matrices.
- `scionh/train_shakespeare.py`: training entrypoint.

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

### SODA-Hyperball

SODA-Hyperball is the default optimizer. It adds the SODA Algorithm 1
initialization anchor from Pethick et al. (arXiv:2605.11172) before the
Hyperball fixed-radius retraction:

```math
W_{k+1} \leftarrow
\left(1 - \frac{1}{k+2}\right)W_k
+ \frac{1}{k+2}W_0
+ \operatorname{BaseUpdate}(G_k).
```

The SODA correction is not multiplied by the learning-rate schedule; scheduled
movement remains inside the ULMO base update. The final Hyperball retraction
preserves each block's fixed RMS radius, so the anchor acts through the
tangent-visible component rather than as a second norm controller.
`--soda-groups` accepts
`auto`, `all`, `none`, or a comma-separated subset of `embed,hidden,out`.
The default `auto` applies SODA to non-output groups: `embed,hidden` for
untied weights, and `hidden` for tied weights. This follows the last-layer
weight-decay note: norm control on the unembedding can bound logits when the
final hidden state is normalized. The Frobenius-normalization note supports
using global radius control most directly on hidden matrices whose outputs are
scale-invariant through RMSNorm. Use `--no-soda` for the plain Hyperball
baseline.

### Schedule-Free ScionH

`--schedule-free` switches to a Schedule-Free ScionH variant and disables SODA.
It keeps the paper's three-sequence structure and replaces AdamC's raw iterate
update with the ScionH/Hyperball retraction:

- `z`: raw iterate updated by the ScionH ULMO atom and Hyperball retraction.
- `x`: the unprojected ambient Schedule-Free average of `z`.
- model weights: `beta*x + (1-beta)*z` for training, `x` for
  evaluation/checkpoints.

Use `--sf-c-warmup-iters` for the paper's averaging warm-start, `--sf-r 1` for
long-run weighting, and optionally `--sf-polyak` for a ScionH-native Polyak
step scalar based on the tangent atom-gradient support pairing.
Use `--sf-geometry geodesic` for the fixed-RMS manifold variant where `x` and
`y` are spherical interpolants instead of ambient linear averages. See
`docs/schedulefree_scionh.md` for the derivation.

## Defaults

- optimizer: SODA-Hyperball
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
- SODA-Hyperball: on by default; disable with `--no-soda`

## Hidden ULMOs

`--hidden-ulmo gram-ns` is the default. It uses the Gram Newton-Schulz form with
the four-moment spectral upper-bound normalization that reuses the first `G @ G`
product already needed by the polynomial iteration.

`--hidden-ulmo swan` applies shape-aware GradNorm before the same Gram
Newton-Schulz whitening step: row GradNorm for wide matrices and col GradNorm
for tall matrices.

`--hidden-ulmo streaming-svd` keeps a per-parameter cached right-singular basis
and applies one or more streaming subspace steps per optimizer update.

Additional hidden-matrix oracle arms are available for controlled geometry
experiments:

- `frobenius`: Euclidean/Frobenius steepest descent direction.
- `colnorm`, `rownorm`, `sign`: non-spectral norm-ball LMOs.
- `svd`: exact polar-factor spectral oracle for small correctness baselines.
- `blockwise-gram-ns`, `blockwise-svd`: row/column-partitioned spectral oracles
  for shardwise/blockwise comparisons (`--block-ulmo-axis`, `--block-ulmo-parts`).

To generate a registered command matrix with convergence stats and per-run
JSONL metrics:

```bash
uv run python -m scionh.experiments.oracle_sweep \
  --matrix screening \
  --arms hidden_gram_ns,hidden_frobenius \
  --seeds 1337,1338,1339 \
  --device cuda
```

Omit `--arms` for the full matrix. Add `--run` to execute the generated
commands. Existing completed JSONL runs are skipped by default; use
`--force-rerun` only when replacing a run intentionally.

Use `--matrix spectral-shape` for the streaming-SVD oracle family that sweeps
the power/Schatten continuum, alignment-targeted power responses,
effective-rank targets, and capped stable-rank targets.

Summarize completed JSONL runs:

```bash
uv run python -m scionh.experiments.oracle_summary _local/oracle_lab/runs
```

The aggregate summary reports paired validation deltas against
`hidden_gram_ns`, paired win counts, median throughput, and small-sample 95%
confidence intervals for the validation-loss delta. In the current screens,
`hidden_streaming_svd` is the strongest replacement candidate for Gram-NS:
it improves validation in the small model and in a wider `d_model=128` check
while remaining cheaper. `hidden_rownorm` is still the cheap non-spectral
candidate, but its validation win appears width-sensitive.

Audit inverse-square-root validation-loss extrapolations, matching the
Schedule-Free+ paper's 5%-to-15% fit window:

```bash
uv run python -m scionh.experiments.prediction_audit _local/oracle_lab/runs
```

The audit fits `val_loss = a / sqrt(t + b) + c` on early eval points and
reports held-out RMSE plus final-loss error. Use this before trusting early
loss-curve predictions for ScionH variants, especially when the run still uses
WSD decay rather than a schedule-free averaged iterate.
For Schedule-Free warm-starts, prefer absolute audit windows with
`--fit-start-x` and `--fit-end-x` so the fit starts after `--sf-c-warmup-iters`.

For the Schedule-Free ScionH variant, fit only after LR warmup and
`--sf-c-warmup-iters` are outside the fit window:

```bash
uv run python -m scionh.train_shakespeare \
  --mode train --schedule-free --no-soda \
  --warmup-iters 100 --sf-c-warmup-iters 200 \
  --decay-iters 0 --sf-beta 0.9 --sf-r 0 \
  --metrics-jsonl _local/prediction_probe/sf_scionh.jsonl \
  --no-save --skip-sample
```

## Recommended SODA-Hyperball Command

```bash
uv run python -m scionh.train_shakespeare \
  --mode train \
  --out-path out/soda_hyperball_2k.pt \
  --sample-out out/soda_hyperball_2k_samples.md \
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

To run the plain Hyperball baseline, add `--no-soda` to the same command.

Evaluate the saved checkpoint with more batches:

```bash
uv run python -m scionh.train_shakespeare \
  --mode eval \
  --device cuda \
  --out-path out/soda_hyperball_2k.pt \
  --eval-iters 200
```
