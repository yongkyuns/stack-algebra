# Walkthrough: a two-state Kalman filter

Estimate position and velocity on a line from ten position measurements. The
purpose is to learn fixed-size matrix construction, prediction, a scalar
correction, and an in-place vector update. This is a **linear** Kalman filter,
not an EKF, ESKF, or navigation implementation.

You need basic Rust references and an understanding of matrix multiplication.
Start with [the shared setup](tutorials.md#before-you-start) and
[Getting started](getting-started.md) if the matrix types are new to you.
[Open the example source](https://github.com/yongkyuns/stack-algebra/blob/main/examples/kalman_1d.rs)
or stay on this page: the excerpts and [complete listing](#complete-runnable-example)
are included from the actual example when the guide is built. Excerpts explain
parts of one program; they are not separate programs to concatenate.

From the repository root:

```sh
cargo run --example kalman_1d --no-default-features
```

The host executable uses `std` to print. `--no-default-features` keeps the
library in its `no_std` configuration; it does not make the host executable a
bare-metal program. Use Cargo locally rather than the browser playground.

## 1. Choose the state, units, and storage

The state is a column vector, `state = [position, velocity]^T`. Position is in
metres and velocity in metres per second. Its uncertainty is a covariance
matrix:

```text
state: Matrix<2, 1, f32>       covariance: Matrix<2, 2, f32>
       [position]                         [P_pp  P_pv]
       [velocity]                         [P_vp  P_vv]
```

`P_pp` is position variance in m^2, `P_vv` is velocity variance in (m/s)^2,
and the off-diagonal entries are position/velocity cross-covariances in m^2/s.
The dimensions and `f32` precision are part of the types. Indexing is
zero-based: `state[(0, 0)]` is position and `state[(1, 0)]` is velocity.

The example starts at time zero with zero estimates and unit variances:

```rust,noplayground
{{#include ../examples/kalman_1d.rs:63:65}}
```

`Matrix::zeros` initializes both state entries, and `Matrix::eye` creates the
identity covariance, so the initial cross-covariances are zero. See the
[`Matrix` API](api/stack_algebra/struct.Matrix.html) for the type and operations.
These values are assumptions for the demonstration, not tuning advice.

The data and parameters are fixed:

```rust,noplayground
{{#include ../examples/kalman_1d.rs:15:18}}
```

`DT` is one second. `ACCELERATION_VARIANCE` is a variance, **not a standard
deviation**: 0.04 (m/s^2)^2 corresponds to 0.2 m/s^2 standard deviation.
Likewise, `MEASUREMENT_VARIANCE = 0.25` m^2 corresponds to 0.5 m standard
deviation. The position samples illustrate motion near 1 m/s; they are not a
random simulation.

## 2. Predict one step forward

The mean model is constant velocity: `p_next = p + dt*v`, `v_next = v`.
Unmodelled acceleration is represented by a zero-mean random value held
constant over each interval, independent between intervals. Its effect on
position and velocity is described by `G`:

```text
F = [[1, dt], [0, 1]]
G = [dt^2 / 2, dt]^T
Q = G G^T * acceleration_variance

state_minus = F * state
P_minus     = F * P * F^T + Q
```

This is a **discrete interval-constant acceleration model**, not a continuous
white-noise spectral density. Measurement noise is independent between
samples and independent of process noise.

Here is the entire prediction helper:

```rust,noplayground
{{#include ../examples/kalman_1d.rs:22:35}}
```

[`from_rows`](api/stack_algebra/struct.Matrix.html#method.from_rows) lets us
write the matrices in the same arrangement as the equations, even though the
library stores dense values column-major. [`transpose`](api/stack_algebra/struct.Matrix.html#method.transpose)
and `*` express the small products directly. Multiplying a 2x1 vector by its
1x2 transpose forms the 2x2 outer product used for `Q`.

The `&mut` arguments let the helper replace the caller's state and covariance.
The assignments through `*state` and `*covariance` write the new values back.
The right-hand sides use the previous values before that assignment. These
readable 2x2 expressions are intentional, not a claim of optimized filter
propagation or zero temporary storage.

**Checkpoint after the first prediction**, before observing position:

```text
state_minus = [0, 0]^T
Q           = [[0.01, 0.02], [0.02, 0.04]]
P_minus     = [[2.01, 1.02], [1.02, 1.04]]
```

The mean remains zero because the initial velocity is zero. Uncertainty grows,
and the off-diagonal entries become nonzero: uncertain velocity affects the
predicted position.

## 3. Correct using a scalar position measurement

We observe position only, so `H = [1, 0]`. Let `z` be the measured position,
`R` its noise variance, and `innovation = z - predicted_position`. The gain
is a two-entry vector, while the innovation variance is just one scalar:

```text
S = P_minus[0, 0] + R
K = [P_minus[0, 0], P_minus[1, 0]]^T / S
state_plus = state_minus + K * innovation
```

There is no matrix inverse or Cholesky solve in this scalar observation model.
The corresponding portion of `update_position` is:

```rust,noplayground
{{#include ../examples/kalman_1d.rs:43:51}}
```

[`axpy_in_place`](api/stack_algebra/struct.Matrix.html#method.axpy_in_place)
adds a scaled vector to an existing one. Here the receiver is `state`, the
scale is the scalar innovation, and the vector is `gain`. Both gain entries
are read from the predicted covariance before that covariance is changed.

**Checkpoint for the first measurement, `z = 1.2` m:**

```text
innovation = 1.2 - 0 = 1.2
S          = 2.01 + 0.25 = 2.26
K          = [0.889381, 0.451327]^T
state_plus = [1.067257, 0.541593]^T
```

The velocity estimate changes even though velocity was not measured directly.
That correction comes from the position/velocity cross-covariance established
by prediction. These checkpoints are rounded calculations using the written
decimal inputs; `f32` arithmetic may differ in the last displayed digits.

## 4. Update the covariance with the same gain

Updating only the state would leave the uncertainty inconsistent with the new
measurement. The example uses the Joseph covariance expression:

```text
J = I - K H
P_plus = J P_minus J^T + R K K^T
```

`residual_map` below is `J`, not the scalar innovation. Since `H = [1, 0]`,
its four entries are easy to construct explicitly:

```rust,noplayground
{{#include ../examples/kalman_1d.rs:53:58}}
```

Both terms are covariance contributions. This avoids teaching a subtractive
covariance shortcut as unconditionally safe in finite precision, but it does
not eliminate rounding error or prove positive semidefiniteness for arbitrary
inputs. The tests check symmetry and covariance validity with tolerances.

**Checkpoint after the first correction:**

```text
P_plus ~= [[0.222345, 0.112832],
           [0.112832, 0.579646]]
```

Position variance is smaller than the predicted 2.01 m^2. Velocity uncertainty
also decreases because the measurement informs the correlated velocity state.

## 5. Repeat in the right order and read the output

Each sample describes the next time step. Predict first, then correct using
that step's position. In particular, the first sample is at `t = 1 s`, not
`t = 0`:

```rust,noplayground
{{#include ../examples/kalman_1d.rs:67:78}}
```

The full output is:

```text
time_s measured_m position_m velocity_m_s
1 1.200 1.067 0.542
2 1.800 1.763 0.647
3 3.100 2.921 0.922
4 3.900 3.881 0.940
5 5.200 5.057 1.042
6 5.900 5.980 0.990
7 7.100 7.047 1.024
8 8.000 8.029 1.006
9 8.800 8.897 0.945
10 10.100 9.994 1.012
```

The filtered position need not equal the current measurement. The update
combines the prediction with the new observation according to their modeled
uncertainties. The velocity estimate approaches the roughly 1 m/s motion in
the sample sequence; this small example is not an accuracy benchmark.

## 6. Try a change, then check your understanding

In a local experiment, increase `MEASUREMENT_VARIANCE` from `0.25` to `1.0`.
For the **same first predicted covariance**, the larger denominator makes both
gain entries smaller, so the first observation produces a smaller correction.
Later gains also depend on the covariances produced by earlier steps.

Alternatively, set `ACCELERATION_VARIANCE` to zero. The model then adds no new
acceleration uncertainty through `Q`. This does not make the measurement noise
zero, and it is not generally a better model for changing velocity.

Restore the original constants before comparing with the printed output or
running the unchanged reference tests. Changing `DT` also changes the sample
times; it is not a harmless performance switch.

The example assumes finite inputs, valid covariance, nonnegative acceleration
variance, and positive measurement variance. Its helpers do not validate all
of those assumptions. Do not treat them as a production filter API for
arbitrary sensor input.

## 7. Run the focused correctness checks

```sh
cargo test --no-default-features --test kalman_1d
cargo test --features std --test kalman_1d
cargo test --release --no-default-features --test kalman_1d
```

The [five integration tests](https://github.com/yongkyuns/stack-algebra/blob/main/tests/kalman_1d.rs)
import the actual example helpers. They check a hand-calculated prediction at
a nonunit time step, a correlated-prior scalar correction, zero innovation,
every demo sample against an independent scalar `f64` reference, and 1,000
repeated updates. The reference uses expanded scalar equations instead of
repeating the example's matrix expressions. These are teaching-code checks,
not navigation qualification.

## Complete runnable example

This listing is included directly from `examples/kalman_1d.rs`, not maintained
as a second implementation. Run it with the Cargo command above. `pub(crate)`
exposes helpers only within the example/test crate, and `#[cfg(not(test))]`
keeps the printing entry point out when the integration tests import the file.
Neither creates a public `stack-algebra` filter API.

```rust,noplayground
{{#include ../examples/kalman_1d.rs}}
```

Continue with [fitting a line from a borrowed buffer](tutorial-mapped-least-squares.md)
to learn mapped storage and QR, or return to the [tutorial overview](tutorials.md).
