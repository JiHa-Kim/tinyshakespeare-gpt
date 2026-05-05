# Hyperball Parametrization

This note documents the active optimizer coordinates used by
`scionc/train_shakespeare.py` after the switch to the Hyperball fixed-RMS
formulation.

## Constraint

Each controlled weight block $W_b$ lives on a fixed-radius RMS sphere:

$$
\|W_b\|_{\mathrm{rms}} = R_b = \|W_{b,0}\|_{\mathrm{rms}}.
$$

The radius is inherited from initialization and never changes.

## State Clock And Step Size

A state half-life $h_\beta$ sets the momentum EMA retention:

$$
\beta = 2^{-\Delta\tau / h_\beta}.
$$

The default `retract` update uses a group-specific learning rate $\eta$ as the
Euclidean pre-retraction step size. The defaults are calibrated from the old
optimizer's observed stable `update/param` RMS, not from radial shrink
retentions. For the default batch geometry, this gives peak learning rates of about
`embed=0.0368`, `hidden=0.0256`, and `out=0.0035`.

## Hyperball Update

Let $\hat W = W / R$ with $\|\hat W\|_{\mathrm{rms}} = 1$. The default update
is the fixed-radius Hyperball RMSNorm retraction:

$$
\hat W' = \operatorname{rmsnorm}(\hat W + \eta_t\,
    \operatorname{rmsnorm}(V)).
$$

The sign is positive because the local ULMO convention returns descent
directions; this is equivalent to the usual $W - \eta V$ form when $V$ denotes
a gradient-like atom. The retraction keeps $\|\hat W'\|_{\mathrm{rms}} = 1$,
so $W'=R\hat W'$ preserves the initialized RMS radius.

The first-order tangent action is:

$$
V_\perp = V - \langle V, \hat W \rangle_{\mathrm{rms}} \hat W.
$$

Large radial alignment between $V$ and $\hat W$ therefore reduces effective
tangent movement even though the raw step is $\eta_t$.

## Geodesic Slerp Ablation

`--hyperball-update slerp` runs the tangent-projected geodesic comparison. It
normalizes the tangent component of the atom:

$$
D = V - \langle V, \hat W \rangle_{\mathrm{rms}} \hat W,
\qquad
U = D / \|D\|_{\mathrm{rms}},
$$

then applies the exponential map on the fixed-RMS sphere:

$$
\hat W' = \cos(\eta_t)\hat W + \sin(\eta_t)U.
$$

Radius preservation is automatic because $\| \hat W' \|_{\mathrm{rms}} = 1$
up to floating-point roundoff. In this ablation, $\eta_t$ is the angular step
in radians.

## WSD Schedule

The WSD warmup/stable/decay schedule produces a scalar $s_t \in [0, 1]$.
This scales the learning rate directly:

$$
\eta_t = s_t \eta_{\mathrm{peak}}.
$$

At $s_t = 1$ (stable phase), $\eta_t = \eta_{\mathrm{peak}}$. At $s_t = 0$
(floor), $\eta_t = 0$ (no movement). State retention $\beta$ is not scheduled.

## Transfer

When batch size, block size, or gradient accumulation changes, keep the state
half-life fixed in processed-token units and recompute $\beta$ from the new
count increment:

$$
\Delta\tau = \text{batch size} \cdot \text{block size} \cdot \text{grad accum},
\qquad
\beta = 2^{-\Delta\tau / h_\beta}.
$$

The learning rate is an explicit optimizer-step coordinate. Change it directly
when changing the batch geometry. Under `--hyperball-update slerp`, the same
scheduled coordinate is interpreted as a geodesic angle in radians.

## CLI

Primary coordinates:

- `--state-half-life`: global momentum-state half-life in processed tokens.
- `--state-half-life-{embed,hidden,out}`: per-group state half-life overrides.
- `--lr`: global pre-retraction learning rate.
- `--lr-{embed,hidden,out}`: per-group learning-rate overrides.
- `--schedule-floor`: WSD schedule floor for the learning-rate ratio (default 0).
- `--hyperball-update {retract,slerp}`: Hyperball RMSNorm retraction update or
  tangent-projected geodesic comparison.
- `--target-rms-{embed,hidden,out}`: initialization RMS targets. Defaults are
  `embed=0.70`, `hidden=0.051`, `out=0.022`. After init, R is frozen.
- `--warmup-iters`, `--decay-frac`, etc.: WSD schedule shape.
