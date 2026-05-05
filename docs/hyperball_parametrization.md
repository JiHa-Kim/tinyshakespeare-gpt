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

## Clocks

A state half-life $h_\beta$ sets the momentum EMA retention. A separate
weight-direction half-life $h_w$ sets the per-update spherical movement:

$$
\beta = 2^{-\Delta\tau / h_\beta},
\qquad
q = 2^{-\Delta\tau / h_w}.
$$

The induced angular step and relative movement of the weight are:

$$
\theta = \arccos q,
\qquad
\varepsilon = \sqrt{1 - q^2}.
$$

The default weight-direction clocks are calibrated from the old optimizer's
observed stable `update/param` RMS, not from its radial shrink retentions.
For the default batch geometry, this gives peak relative RMS movements of
about `embed=0.0368`, `hidden=0.0256`, and `out=0.0035`.

## Hyperball Update

Let $\hat W = W / R$ with $\|\hat W\|_{\mathrm{rms}} = 1$. The default update
is the fixed-radius Hyperball RMSNorm retraction:

$$
\hat W' = \operatorname{rmsnorm}(\hat W + \varepsilon\,
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
tangent movement even though the raw step is $\varepsilon$.

## Slerp Ablation

`--hyperball-update slerp` runs the older tangent-projected comparison. It
normalizes:

$$
D = V - \langle V, \hat W \rangle_{\mathrm{rms}} \hat W,
\qquad
U = D / \|D\|_{\mathrm{rms}},
$$

then updates:

$$
\hat W' = q \hat W + \varepsilon U.
$$

Radius preservation is automatic because $\| \hat W' \|_{\mathrm{rms}}^2 =
q^2 + (1-q^2) = 1$ up to floating-point roundoff.

## WSD Schedule

The WSD warmup/stable/decay schedule produces a scalar $s_t \in [0, 1]$.
This scales the spherical movement directly:

$$
\varepsilon_t = s_t \varepsilon_{\mathrm{peak}}, \qquad
q_t = \sqrt{1 - \varepsilon_t^2}.
$$

At $s_t = 1$ (stable phase), $\varepsilon_t = \varepsilon_{\mathrm{peak}}$
(full angular movement). At $s_t = 0$ (floor), $q_t = 1$ (no movement).
State retention $\beta$ is not scheduled.

## Transfer

When batch size, block size, or gradient accumulation changes, keep both
half-lives fixed in processed-token units and recompute $\beta$ and $q$ from
the new count increment:

$$
\Delta\tau = \text{batch size} \cdot \text{block size} \cdot \text{grad accum},
\qquad
\beta = 2^{-\Delta\tau / h_\beta},
\qquad
q = 2^{-\Delta\tau / h_w}.
$$

## CLI

Primary coordinates:

- `--state-half-life`: global momentum-state half-life in processed tokens.
- `--state-half-life-{embed,hidden,out}`: per-group state half-life overrides.
- `--direction-half-life`: global weight-direction half-life in processed tokens.
- `--direction-half-life-{embed,hidden,out}`: per-group overrides.
- `--schedule-floor`: WSD schedule floor for the movement ratio (default 0).
- `--hyperball-update {retract,slerp}`: Hyperball RMSNorm retraction update or
  normalized spherical-lerp comparison.
- `--target-rms-{embed,hidden,out}`: initialization RMS targets. Defaults are
  `embed=0.70`, `hidden=0.051`, `out=0.022`. After init, R is frozen.
- `--warmup-iters`, `--decay-frac`, etc.: WSD schedule shape.
