# Fit a curve online with recursive least squares

An embedded calibration task often receives one measurement at a time. It
should not need to keep every previous sample or rebuild a growing design
matrix before producing an estimate. This example learns a quadratic curve
from a stream using **recursive least squares (RLS)** with fixed-size storage.

Run the complete host example from your repository checkout:

```sh
cargo run --example recursive_least_squares --no-default-features
```

The final estimate below is produced by that Rust program. The observations,
intermediate coefficients, and smooth curve points are all exported from
execution, rather than calculated by the documentation renderer.

![Final recursive quadratic estimate, the received observations, and the separate synthetic reference.](generated/rls/rls-after-40.svg)

## 1. Same curved model, different data flow

The model is still `y = a*x² + b*x + c`. It is curved in `x`, but linear in the
three unknown coefficients. The [batch companion](tutorial-mapped-least-squares.md)
constructs all design rows and solves with column-pivoted QR. Here a new
`(x, y)` pair updates a compact summary of the observations received so far.

Both examples use the **same 40 synthetic pairs**. This driver visits source
row `(17 * step) % 40`, for zero-based `step`, instead of increasing `x`
monotonically. The fixed permutation covers different parts of the input
range early; it does not change the values or select observations based on
their errors. With no forgetting, the final objective is independent of order,
although intermediate estimates and floating-point rounding can differ.

The known generating curve is only used to construct and illustrate the
synthetic dataset. Its coefficients are not passed into the estimator.
This is a calibration scenario, not a simulation of a moving object's trajectory.

For numerical scaling, define `u = x / 3` and fit

```text
y = A*u² + B*u + C
phi = [u², u, 1]
theta = [A, B, C]ᵀ

a = A / 9, b = B / 3, c = C
```

The observations span `u` from `-1` to `1`. Figures and CSV coefficient columns
use the original `x` coordinates: `a`, `b`, and `c`. Scaling is part of the
model definition, including its prior, not a reason to mix these two sets of
coefficients.

## 2. What persists between measurements

The estimator is implemented in `examples/support/quadratic_rls.rs`, separate
from the host driver and its reporting storage:

```rust,noplayground
{{#include ../examples/support/quadratic_rls.rs:state}}
```

This is **square-root information RLS**, implemented by updating a QR factor.
`r` is a `3 × 3` upper-triangular factor; `transformed_rhs` is a `3 × 1` vector.
They summarize the least-squares problem, and the current coefficients are
obtained with the library's borrowed upper-triangular solve. There is no
per-update matrix inverse or growing design matrix.

The two matrices contain **12 scalars**. The complete object also stores the
input scale, square root of the forgetting factor, and accepted-sample count.
The executable prints `size_of` for both `f32` and `f64`; these are sizes for
the execution target, not measured MCU stack peaks. Updates also need bounded
local workspace, including a candidate state used to make rejection atomic.

The helper uses neither `std` nor `alloc`. The desktop driver deliberately
retains history and four estimator snapshots for documentation. Those vectors,
the synthetic input fixture, and plotting work are **not estimator state** and
are not required by an on-device stream.

## 3. Initialize a prior, then update one observation at a time

The reference run uses these settings:

```rust,noplayground
{{#include ../examples/recursive_least_squares.rs:configuration}}
```

```rust,noplayground
{{#include ../examples/recursive_least_squares.rs:initialize}}
```

The initial mean is zero. `PRIOR_PRECISION = 0.01` specifies a weak quadratic
penalty in the **normalized coefficient basis**. It is not a measured sensor
noise variance. The data terms have unit weight. `FORGETTING = 1` keeps all
accepted observations equally weighted, appropriate for this fixed curve.

Each call to `update(x, y)` first predicts using the previous coefficients,
records the innovation, and appends the new row to the compact QR problem.
Three Givens rotations eliminate that row, leaving another `3 × 3` triangular
factor. These orthogonal row operations also transform the right-hand side.
The estimator solves the resulting triangular system and commits the new
state only after the numerical checks succeed.

For `p` coefficients, row updates and triangular solves take `O(p²)` arithmetic
and fixed `O(p²)` storage. Here `p = 3`; neither grows with the stream length.
Square-root organization avoids the subtractive inverse-information update,
but is not a guarantee against overflow, poor scaling, or insufficient data.

<details>
<summary>The actual update implementation</summary>

```rust,noplayground
{{#include ../examples/support/quadratic_rls.rs:update}}
```

`InvalidConfiguration`, `NonFiniteInput`, `NumericalBreakdown`, and
`SampleCountOverflow` are example-local typed errors. A rejected update leaves
the entire prior estimator state unchanged, including its sample count.

</details>

### The exact objective matters

With normalized coefficients `theta`, initial mean `theta₀`, prior precision
`delta`, and forgetting factor `lambda`, the update minimizes

```text
J_k(theta) = delta * lambda^k * ||theta - theta₀||²
           + sum(i = 1..k) lambda^(k-i) * (y_i - phi_i*theta)²
```

Initialization uses `R₀ = sqrt(delta)*I` and `z₀ = R₀*theta₀`. Before appending
a new row, both retained factors are scaled by `sqrt(lambda)`. With
`lambda = 1`, the prior remains fixed. With `lambda < 1`, **the prior also
fades**, alongside older data; this is not constant-ridge regularization.

Consequently, the final result need not exactly match the companion's
*unregularized* batch fit. Tests compare against an augmented batch QR problem
with the **same prior, input normalization, weights, and observation prefix**.
They do this after every observation for both `f32` and `f64`, including a
forgetting-factor case with a nonzero initial mean.

For background on streaming QR factors, see MathWorks' description of
[Q-less QR with forgetting](https://www.mathworks.com/help/fixedpoint/ref/realpartialsystolicqlessqrdecompositionwithforgettingfactor.html).
The objective above specifies this example's initialization and weighting;
reference implementations can use different parameter conventions.

## 4. Watch the estimate develop

Early estimates use only a small part of the available information. Each
snapshot shows only observations already received, and a curve evaluated by
Rust from the estimator at that step. The dashed synthetic reference is
shown for explanation; it is not an input to the estimator.

![Recursive estimate after five observations, showing only those five observations.](generated/rls/rls-after-5.svg)

<details>
<summary>Intermediate estimates after 10 and 20 observations</summary>

![Recursive estimate after ten observations.](generated/rls/rls-after-10.svg)

![Recursive estimate after twenty observations.](generated/rls/rls-after-20.svg)

The final 40-observation estimate is shown at the top of this walkthrough.
Curves outside the span of observations received so far are extrapolations,
not evidence that the estimator already knows that region.

</details>

<details>
<summary>Coefficient histories in original x coordinates</summary>

![Evolution of the quadratic coefficient a.](generated/rls/rls-coefficient-a.svg)

![Evolution of the linear coefficient b.](generated/rls/rls-coefficient-b.svg)

![Evolution of the intercept c.](generated/rls/rls-coefficient-c.svg)

Each point is the coefficient after the indicated update. Individual
coefficients and prediction errors need not approach their references
monotonically. A small final error in one finite dataset is not a general
accuracy guarantee.

</details>

## 5. Do not confuse online error with final residuals

The **innovation** is the prediction error before learning from a new sample:

```text
innovation_k = y_k - prediction(theta_(k-1), x_k)
```

![Pre-update prediction errors versus observation count.](generated/rls/rls-online-error.svg)

The first prediction comes from the zero prior mean. Later predictions use
only earlier samples. These errors characterize the online learning process.
In contrast, a **final-fit residual** evaluates every observation with the
final model, which already used those observations:

```text
final_residual_i = y_i - prediction(theta_40, x_i)
```

![All final-fit residuals on a separate axis.](generated/rls/rls-final-residuals.svg)

The host driver replays the fixed inputs for this final diagnostic. The
embedded update does not need to retain them. Neither quantity is the same
as the injected synthetic noise, and regularization changes the usual
unregularized residual-orthogonality condition.

## 6. Experiments and deployment boundaries

Start by changing the shared fixture's noise scale and regenerating the
figures. Both fitting examples then use the same changed measurements. For a
rank experiment, repeatedly call the estimator with one `x` value: regularization
can still produce finite coefficients, but the observations do not determine
all three parameters. **Successful updates are not a rank or excitation certificate.**

For a changing calibration, `lambda < 1` can reduce the influence of old data.
Use a deliberately time-varying generating curve for that experiment and
compare against the matching weighted objective. Do not silently change the
forgetting factor in this fixed-curve example just to improve its appearance.

The reference figures use `f64`. The same generic estimator is tested in
`f32` and both versions are compiled through a `no_std` consumer for
`thumbv7em-none-eabihf`. This is compile/host numerical evidence, not a board
benchmark, measured worst-case execution time, or indefinite-operation
qualification. Account for local workspace, stack placement, input scale,
measurement validity, and hardware floating-point behavior in your application.

## Reproduce and verify

```sh
cargo test --no-default-features --test recursive_least_squares
cargo test --features std --test recursive_least_squares
cargo test --release --no-default-features --test recursive_least_squares
python3 scripts/generate_rls_assets.py
python3 scripts/generate_rls_assets.py --check
./scripts/build_docs.sh
```

`--csv` exports the actual learning history and final residuals;
`--curve-csv` exports the four snapshots on 201-point grids evaluated by Rust.
The [trace CSV](generated/rls/trace.csv), [curve CSV](generated/rls/curves.csv),
[provenance](generated/rls/provenance.json), and [saved lockfile](generated/rls/Cargo.lock)
are built with this page. Python draws the exports and checks their consistency;
it does not run an RLS implementation. Failed generation has no stale-image
fallback, and CI re-executes the example to compare all generated bytes.

<details>
<summary>Actual terminal output from this documentation build</summary>

```text
{{#include generated/rls/output.txt}}
```

</details>

Continue with [batch fitting from a borrowed buffer](tutorial-mapped-least-squares.md)
for the `Map`/QR workflow, or [the Kalman filter](tutorial-kalman-1d.md) to estimate
a changing physical state instead of fixed calibration parameters. These are
application examples built from stack-algebra, not new public estimator APIs.
