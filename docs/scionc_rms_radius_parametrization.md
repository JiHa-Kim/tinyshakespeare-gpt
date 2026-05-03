# ScionC RMS-Radius Parametrization

This note documents the optimizer coordinates used by
`scionc/train_shakespeare.py`. The active coordinate system is:

- target stationary RMS radius `R_W` per optimizer group,
- momentum-state retention half-life,
- weight-retention half-life,
- nonnegative step-scale schedule.

The derived support radius `rho` is not user-tuned directly. It is recomputed
from the RMS objective at the active momentum and weight retentions.

## Update

For one weight block, let `w` be the parameter, `g` the gradient, `m` the
momentum state, and `v` the ULMO atom. One optimizer step is:

```math
m'=\beta m+(1-\beta)g,
```

```math
v=\operatorname{ulmo}(m'),
```

```math
w'=\zeta w+(1-\zeta)\rho v.
```

The retentions come from half-lives in processed-token units:

```math
\beta = 2^{-\Delta\tau/h_\beta},
\qquad
\zeta_0 = 2^{-\Delta\tau/h_\zeta},
\qquad
\zeta = \zeta_0^{s_t}.
```

Here `s_t >= 0` is the scheduled step scale. `s_t = 0` gives `zeta = 1` and
turns off the weight update.

## RMS Radius Match

The user-facing radius is the target stationary RMS radius:

```math
\mathbb{E}\|w\|^2 = R_W^2.
```

Under the EMA atom-correlation approximation `c_k ~= beta^k` and unit-scale
ULMO atoms:

```math
A_\zeta = \frac{1+\zeta\beta}{1-\zeta\beta}.
```

The derived support radius is:

```math
\rho =
R_W
\sqrt{
\frac{1+\zeta}{(1-\zeta)A_\zeta}
}.
```

The additive optimizer coefficient is:

```math
\eta = (1-\zeta)\rho.
```

So the PyTorch optimizer group stores:

- `rms_radius`: target `R_W`,
- `rho`: active derived support radius,
- `weight_retention`: active `zeta`,
- `lr`: active additive coefficient `eta`,
- `atom_correlation`: active `A_zeta`.

## Transfer

When batch size, block size, or gradient accumulation changes, keep the semantic
coordinates fixed:

```math
R_W,\quad h_\beta,\quad h_\zeta,\quad s_t.
```

Then recompute:

```math
\Delta\tau
=
\text{batch size}
\cdot
\text{block size}
\cdot
\text{gradient accumulation},
```

```math
\beta = 2^{-\Delta\tau/h_\beta},
\qquad
\zeta = 2^{-s_t\Delta\tau/h_\zeta},
\qquad
\rho =
R_W
\sqrt{
\frac{1+\zeta}{(1-\zeta)A_\zeta}
}.
```

This preserves the momentum timescale, weight-retention timescale, scheduled
step-scale coordinate, and target RMS radius in the chosen count units.

## CLI

Primary coordinates:

- `--rms-radius-*`: group target RMS radii.
- `--weight-retention-half-life-*`: group weight-retention half-lives.
- `--beta-half-life`: momentum-state retention half-life.
- `--log2-step-scale*`: base-2 log of the peak step scale.
- `--step-scale*`: linear peak step scale.
- `--min-step-scale*`: linear schedule floor.

The output schedule prints both `rw` and the derived `rho`, so runs can be
compared in RMS-radius space while still exposing the instantaneous support
radius used by the update.
