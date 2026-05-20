# Schedule-Free ScionH

This note records the Schedule-Free adaptation used by
`ScheduleFreeHyperball`.

The Schedule-Free+ paper keeps three sequences:

```math
y_t = \beta_t x_t + (1-\beta_t)z_t,\qquad
z_{t+1}=z_t-\alpha_t G_t(y_t),\qquad
x_{t+1}=(1-c_{t+1})x_t+c_{t+1}z_{t+1}.
```

For ScionH, the base iterate `z` is constrained to a fixed RMS sphere
`S_R = {W : rms(W)=R}`. The default `--sf-geometry ambient` mode keeps the
paper's linear `x` and `y` recurrences. This is the closest AdamC-to-ScionH
translation because it changes only the base optimizer step.

The adaptation therefore changes only the base optimizer step: AdamC's update
for `z` is replaced by the ScionH/Hyperball retraction, while the paper's
linear `x` and `y` recurrences remain intact:

```math
z_{t+1}=\operatorname{Retr}_R(z_t+\eta_t A_t),
\qquad
x_{t+1}=(1-c_{t+1})x_t+c_{t+1}z_{t+1},
```

```math
y_{t+1}=\beta_t x_{t+1}+(1-\beta_t)z_{t+1}.
```

The model tensor stores `y`; gradients are evaluated at `y`. Validation and
checkpoint output use the paper's averaged sequence `x`. SODA is disabled for
this variant: SODA is another initialization-anchored averaging rule, while
Schedule-Free already defines the averaging sequence being tested.

## Fixed-RMS Geometry

The fixed-RMS variant is `--sf-geometry geodesic`. It keeps all three
Schedule-Free sequences on `S_R` by replacing Euclidean interpolation with the
spherical interpolant. For flattened tensors with target norm `rho=R sqrt(d)`,
define

```math
\operatorname{slerp}_R(a,b;\theta)
= \rho\,
\frac{
  \sin((1-\theta)\omega)\,a/\rho
  + \sin(\theta\omega)\,b/\rho
}{
  \sin \omega
},
\qquad
\omega=\arccos\left(\frac{\langle a,b\rangle}{\rho^2}\right).
```

Then the fixed-RMS Schedule-Free ScionH equations are

```math
z_{t+1}=\operatorname{Retr}_R(z_t+\eta_t A_t),
\qquad
x_{t+1}=\operatorname{slerp}_R(x_t,z_{t+1};c_{t+1}),
```

```math
y_{t+1}=\operatorname{slerp}_R(z_{t+1},x_{t+1};\beta_t).
```

This is not the paper's linear averaging theorem anymore; it is the natural
manifold analogue. For small angles, the geodesic update agrees to first order
with tangent-space averaging on the RMS sphere, while preserving the ScionH
radius exactly for `x`, `y`, and `z`.

## Averaging Weights

The implementation follows Algorithm 1's warm-start and weighted averaging:

```math
c_t=1 \quad (t \le C_{\text{warmup}}),
\qquad
w_t=t^r\gamma_{\max,t}^{p},
\qquad
c_t=\frac{w_t}{\sum_{i>C_{\text{warmup}}}w_i}.
```

For fixed group learning-rate ratios, the group LR scale cancels out of `c_t`.
The default `r=0` is the short-run setting; `r=1` is the paper's long-run
recommendation.

## ScionH Polyak Scalar

The paper's AdamC Polyak denominator uses an L1 approximation to Adam's
preconditioned gradient norm. ScionH does not use Adam's coordinatewise
preconditioner, so the corresponding denominator is the first-order decrease
predicted by the ScionH atom.

For a block with query weight `y`, raw iterate `z`, radius `R`, group base step
`\ell_g`, gradient `g`, and ScionH descent atom `u`, the retraction has local
first-order displacement

```math
\delta_g(\rho)
  = \rho\,\ell_g\,R\sqrt{d}\,
      \left[
      \frac{u}{\|u\|_F}
      - \left\langle\frac{u}{\|u\|_F},\frac{z}{R\sqrt d}\right\rangle
        \frac{z}{R\sqrt d}
      \right]
      + O(\rho^2).
```

The radial component of `u` is removed because the RMS projection in the
Hyperball retraction maps it away to first order. The predicted loss decrease
per unit global scalar `rho` is therefore

```math
D_t
  = \sum_g \ell_g R_g\sqrt{d_g}
      \max\left(0,-\left\langle g_g,
        \operatorname{Tan}_{z_g}\left(\frac{u_g}{\|u_g\|_F}\right)
      \right\rangle\right).
```

The numerator uses the same Taylor idea as the paper, written against the
actual query point. In ambient mode this reduces to the paper's
`\beta <g,z-x>` correction. In geodesic mode it is the embedded first-order
correction from `y` to `z`:

```math
N_t = f(y_t)-f_*+\langle g_t,z_t-y_t\rangle .
```

The optional `--sf-polyak` scale is

```math
\rho_t = \frac{\max(0,N_t)}{\operatorname{EMA}(D_t)}.
```

Then each group uses effective step `rho_t * ell_g`. `--sf-polyak-max-scale`
can cap this multiplier for early smoke tests; `0` leaves it uncapped.

## Prediction Audit

The inverse-square-root fit should be applied to the validation loss of `x`,
after the Schedule-Free warm-start transition is out of the fit window:

```bash
python -m scionh.experiments.prediction_audit _local/prediction_probe \
  --glob '*.jsonl' --fit-start-x 160 --fit-end-x 260
```

Short Tiny Shakespeare runs are not a faithful test of the paper's long-run
claim. They are useful only as implementation smoke tests; the prediction
question needs a no-decay Schedule-Free run where the fit window does not
straddle LR warmup or `c_t=1` warm-start.
