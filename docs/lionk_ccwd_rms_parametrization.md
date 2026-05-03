# LionK/CCWD RMS-Radius Parametrization

This note documents the lower-level `LionKCCWDPA` optimizer. The current
coordinates are:

- additive direction scale `gamma`, stored in the PyTorch `lr` field,
- momentum-state retention `beta2`,
- readout blend `beta1`,
- active-coordinate weight retention `zeta`, or
- target per-coordinate RMS radius `R_W` plus online CCWD energy statistics.

The optimizer uses only these retention/radius coordinates.

## Direction

For one tensor, with gradient `g`, momentum state `m`, and direction map
`D`, the Lion-K readout is:

```math
m'=\beta_2 m+(1-\beta_2)g,
```

```math
z=\beta_1 m' + (1-\beta_1)g
\qquad
\text{(Nesterov readout),}
```

```math
u=D(z).
```

The default direction map is `u=-sign(z)`.

For the non-Nesterov readout, replace `m'` in the readout by the previous
momentum state. The effective stored-state coefficient is therefore:

```math
b =
\begin{cases}
\beta_1\beta_2, & \text{Nesterov},\\
\beta_1, & \text{non-Nesterov}.
\end{cases}
```

## Weight Retention

The unmasked update is:

```math
w'=\zeta w+\gamma u.
```

Equivalently:

```math
w'=w-(1-\zeta)w+\gamma u.
```

Cautious weight decay applies the retention action only on coordinates aligned
with the update direction:

```math
P_i=\mathbf{1}_{\{w_i u_i>0\}},
```

```math
w'=w-(1-\zeta)(P\odot w)+\gamma u.
```

Here `zeta` is the active-coordinate retention. The additive scale is always
`gamma`; the implied support radius is `rho=gamma/(1-zeta)` when
`zeta < 1`.

## Empirical RMS Match

For CCWD, the production rule measures the one-step energy terms instead of
assuming them away. Write `d=1-zeta`:

```math
w'=w-d(P\odot w)+\gamma u.
```

The exact one-step energy identity is:

```math
\|w'\|^2
=
\|w\|^2
-d(2-d)\|P\odot w\|^2
+\gamma^2\|u\|^2
+2\gamma\langle w,u\rangle
-2\gamma d\langle P\odot w,u\rangle.
```

For a per-coordinate target `R_W`, the group target energy is
`R_W^2 * numel`. Equivalently, with:

```math
p_2=\frac{\|P\odot w\|^2}{\|w\|^2},
\qquad
h=\frac{\langle w,u\rangle}{\|w\|\|u\|},
\qquad
k=\frac{\langle P\odot w,u\rangle}{\|w\|\|u\|},
```

```math
r=\frac{\|w\|}{R_W\sqrt{\mathrm{numel}}},
\qquad
\alpha=\frac{\gamma\|u\|}{R_W\sqrt{\mathrm{numel}}},
```

the target equation is:

```math
p_2r^2d^2-(2p_2r^2+2\alpha kr)d+
(r^2-1+\alpha^2+2\alpha hr)=0.
```

`LionKCCWDPA` tracks EMAs of `p2`, `h`, and `k`, uses the current `r` and
`alpha`, and applies the smaller valid root in `[0, 1]` after
`energy_warmup` steps. Before warmup, or if the quadratic has no valid root,
it falls back to the stationary prior below. This keeps the default adaptive
to persistent gradients, output-layer cross terms, sign persistence, and mask
structure, while still requiring only one scalar solve per optimizer group.

## Stationary Prior

For a stationary segment, write:

```math
c_k=
\frac{\mathbb{E}\langle u_t,u_{t-k}\rangle}
{\mathbb{E}\|u_t\|^2},
\qquad
c_u^2=\mathbb{E}u_{t,i}^2,
```

and:

```math
A_a=1+2\sum_{k\ge1}a^k c_k.
```

For the unmasked update, the stationary per-coordinate RMS balance is:

```math
R_W^2=
\frac{\gamma^2 c_u^2 A_\zeta}{1-\zeta^2}.
```

So, given `gamma` and `R_W`, the active decay complement
`d=1-zeta` solves:

```math
d(2-d)=
\frac{\gamma^2 c_u^2}{R_W^2}A_{1-d}.
```

For CCWD, let `p` be the squared-norm fraction affected by the cautious mask.
The scalar-retention closure uses:

```math
a=1-pd,
```

and solves:

```math
p\,d(2-d)=
\frac{\gamma^2 c_u^2}{R_W^2}A_{1-pd}.
```

This finite-retention equation is the cold-start and ablation prior. It avoids
the small-step approximation and costs only a scalar solve per optimizer group.

## Correlation Priors

For the linear readout under independent gradients, define:

```math
\nu_0
=(1-b)^2+\frac{b^2(1-\beta_2)}{1+\beta_2},
```

```math
r_1
=
\frac{
b(1-\beta_2)(1+\beta_2-b)
}{
(1+\beta_2)\nu_0
}.
```

Then:

```math
c_k=r_1\beta_2^{k-1},
```

and the weighted correlation factor has the closed form:

```math
A_a
=
1+\frac{2ar_1}{1-a\beta_2}.
```

For the default sign direction, the Gaussian sign correction gives:

```math
c_k=
\frac{2}{\pi}\arcsin(r_1\beta_2^{k-1}).
```

The implementation sums this without a long lag loop by using the arcsine
power series:

```math
A_a
=
1+
\frac{4}{\pi}
\sum_{j\ge0}
\frac{\binom{2j}{j}}{4^j(2j+1)}
\frac{a\,r_1^{2j+1}}{1-a\beta_2^{2j+1}}.
```

This keeps the correction scalar, cheap, and stable.

## Optimizer Fields

Important group fields:

- `lr`: additive direction scale `gamma`.
- `rms_radius`: target per-coordinate `R_W`; if omitted, plain `weight_decay`
  is used.
- `weight_retention`: active `zeta`; if supplied by the caller it is fixed.
- `decay_rule`: `auto`, `energy`, or `stationary`.
- `energy_beta`: EMA retention for `p2`, `h`, and `k`.
- `energy_warmup`: number of measured steps before the empirical root is used.
- `energy_p2`, `energy_h`, `energy_k`, `energy_radius_ratio`,
  `energy_step_ratio`: live diagnostics.
