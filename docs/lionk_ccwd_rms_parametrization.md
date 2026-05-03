# LionK/CCWD RMS-Radius Parametrization

This note documents the lower-level `LionKCCWDPA` optimizer. The current
coordinates are:

- additive direction scale `gamma`, stored in the PyTorch `lr` field,
- momentum-state retention `beta2`,
- readout blend `beta1`,
- active-coordinate weight retention `zeta`,
- optional target per-coordinate RMS radius `R_W`.

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

## RMS Match

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

This is the finite-retention equation used by the code. It avoids the
small-step approximation and costs only a scalar solve per optimizer group.

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

## Practical Diagnostics

The group-level correction is intentionally cheap. For analysis runs, the
exact one-step CCWD energy equation is useful:

```math
\|w'\|^2
=
\|w\|^2
-d(2-d)\|P\odot w\|^2
+\gamma^2\|u\|^2
+2\gamma\langle w,u\rangle
-2\gamma d\langle P\odot w,u\rangle.
```

Setting this equal to the target energy gives a quadratic in `d`. This is the
right diagnostic when measuring whether a layer is currently over-retained or
over-decayed, but it requires live reductions over weights and directions, so
it is not the default optimizer path.
