# RMS-Sphere Parametrization

This note documents the active optimizer coordinates used by
`scionc/train_shakespeare.py` after the switch to the RMS-Sphere formulation.

## Constraint

Each controlled weight block $W_b$ lives on a fixed-radius RMS sphere:

$$
\|W_b\|_{\mathrm{rms}} = R_b = \|W_{b,0}\|_{\mathrm{rms}}.
$$

The radius is inherited from initialization and never changes.

## Direction Clock

A single direction half-life $h$ (in processed-token units) sets the
per-update direction retention:

$$
q = 2^{-\Delta\tau / h}.
$$

The induced angular step and relative movement are:

$$
\theta = \arccos q, \qquad \varepsilon = \sqrt{1 - q^2}.
$$

Momentum retention is tied: $\beta = q$.

## Spherical Update

Let $\hat W = W / R$ with $\|\hat W\|_{\mathrm{rms}} = 1$. The ULMO atom
$V = \mathrm{ulmo}(M)$ is projected to the tangent space and normalized:

$$
D = V - \langle V, \hat W \rangle_{\mathrm{rms}} \hat W,
\qquad
U = D / \|D\|_{\mathrm{rms}}.
$$

The spherical update is:

$$
\hat W' = q \hat W + \varepsilon\, U.
$$

Radius preservation is automatic: $\|\hat W'\|_{\mathrm{rms}}^2 = q^2 + (1-q^2) = 1$.
A final normalization corrects floating-point roundoff.

## WSD Schedule

The WSD warmup/stable/decay schedule produces a scalar $s_t \in [0, 1]$.
This scales the halving exponent:

$$
q_t = q_{\mathrm{peak}}^{s_t}.
$$

At $s_t = 1$ (stable phase), $q_t = q_{\mathrm{peak}}$ (full angular movement).
At $s_t = 0$ (floor), $q_t = 1$ (no movement).

## Transfer

When batch size, block size, or gradient accumulation changes, keep the
half-life fixed in processed-token units and recompute $q$ from the new
count increment:

$$
\Delta\tau = \text{batch size} \cdot \text{block size} \cdot \text{grad accum},
\qquad
q = 2^{-\Delta\tau / h}.
$$

## CLI

Primary coordinates:

- `--direction-half-life`: global direction-retention half-life in processed tokens.
- `--direction-half-life-{embed,hidden,out}`: per-group overrides.
- `--schedule-floor`: WSD schedule floor for the halving exponent ratio (default 0).
- `--target-rms-{embed,hidden,out}`: initialization RMS targets. Defaults are
  `embed=0.70`, `hidden=0.051`, `out=0.022`. After init, R is frozen.
- `--warmup-iters`, `--decay-frac`, etc.: WSD schedule shape.
