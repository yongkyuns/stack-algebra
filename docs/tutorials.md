# Tutorials

These tutorials are organized by the data layout and numerical operation that
drive a design. Start with the smallest matching path, then open the linked API
reference for method signatures and trait bounds. Runnable examples use small
models to teach library operations; they are not complete application frameworks
or performance benchmarks.

## Fixed-size dense algebra

Use `Matrix<M, N, T>` when dimensions are known at compile time. The path is:

1. Construct a matrix with `matrix!`, `Matrix::zeros`, or `Matrix::eye`.
2. Compose products and reductions with the operators and `mul_into` methods.
3. Choose a decomposition from the matrix assumptions.

See [Getting started](getting-started.md), [API usage](api-usage.md), and the
[`Matrix` API](api-reference.md).

## A two-state Kalman filter

The runnable `examples/kalman_1d.rs` example estimates position and velocity on
one line using scalar position observations. It is a linear Kalman filter, not
an EKF, ESKF, or navigation system. It teaches fixed-size construction,
prediction, a scalar correction, and `axpy_in_place` for updating an existing
state vector. The covariance expressions deliberately favor readable 2x2
arithmetic over a hand-optimized implementation.

The state is `[position (m), velocity (m/s)]`, initially `[0, 0]` with covariance
`diag(1 m^2, 1 (m/s)^2)` and no initial correlation. Each one-second step first
predicts and then incorporates that step's position observation:

```text
F = [[1, dt], [0, 1]]
G = [dt^2 / 2, dt]^T
Q = G G^T * acceleration_variance
H = [1, 0]
R = measurement_variance
```

Here acceleration is a zero-mean random value held constant over each interval
and independent between intervals, with variance `0.04 (m/s^2)^2`. This is a
discrete acceleration model, not a continuous white-noise spectral density.
Position measurement noise has variance `0.25 m^2`, is independent between
samples, and is independent of process noise. The sample values are a fixed
illustration of motion near 1 m/s, not a random simulation or tuning guidance.

Since `H = [1, 0]`, the innovation variance is the scalar `P[0,0] + R` and the
gain comes from the first covariance column. No inverse or Cholesky solve is
needed. The example uses the Joseph covariance formula with small matrix
expressions rather than presenting a subtractive update as unconditionally
safe in finite precision.

From the repository root, run:

```sh
cargo run --example kalman_1d --no-default-features
cargo test --no-default-features --test kalman_1d
```

The host executable uses `std` to print results while the library keeps its
default `no_std` configuration. The final line, rounded to three decimals, is:

```text
time_s measured_m position_m velocity_m_s
10 10.100 9.994 1.012
```

The program also prints the preceding nine updates. Its five integration tests
import the actual example helpers and check hand-calculated prediction and
correction, zero innovation, every sample against an independent scalar `f64`
reference, and a longer constant-velocity sequence. Those checks are separate
from the teaching code; they do not qualify a production navigation filter.

For a matrix factorization, use the small Cholesky example in
[Getting started](getting-started.md) instead of adding an unnecessary solver to
this scalar observation model. See `examples/mapped_least_squares.rs` for QR on
caller-owned storage and `examples/embedded_resource_budget.rs` for storage
accounting. Real consumer workloads, not the size of this teaching model,
should motivate future kernel/API optimization.

## Views and external buffers

Use `Map` for contiguous column-major storage and `StridedMap` when row or
column spacing is supplied by another system. Use `Block` for fixed-size
submatrices without copying. The view types borrow their source for the view's
lifetime; use `Matrix::from_view` only when an owned snapshot is intentional.

See [API usage — external buffers and views](api-usage.md) and the generated
[view APIs](api-reference.md).

## Dense factorizations

Select a factorization from the input assumptions:

- `Cholesky` for symmetric positive-definite systems.
- `Ldlt` for symmetric systems that may be indefinite.
- `PartialPivLu` for general square systems.
- `HouseholderQr` or `ColPivHouseholderQr` for least-squares systems.
- `Svd` when rank information or a robust pseudoinverse is required.

The [solver guide](api-usage.md) describes failure behavior, factor reuse, and
output-reuse methods.

## Geometry

Use `Quaternion`, `AngleAxis`, and `RotationMatrix` for rotations; use
`Isometry` for rigid transforms and `AffineTransform` for general affine
transforms. Keep the scalar type explicit and convert at boundaries with
`cast`. The [feature set](features.md) lists the available representations and
the [use-case guide](use-cases.md) shows how they compose with dense matrices.

## Sparse and block-sparse systems

Use `StaticCscPattern` and `StaticCscMatrix` when a scalar sparsity pattern is
known. Use block sparse storage when repeated fixed-size blocks describe the
problem more naturally. Build or reuse the symbolic pattern before numeric
factorization; see [API usage — sparse storage](api-usage.md) and [use cases —
sparse systems](use-cases.md).

## Embedded and bounded workflows

The fixed-size core is `no_std` and does not require a heap allocation. Use
bounded storage when active dimensions vary within a compile-time limit, and
map caller-owned memory when the buffer belongs to a device or driver. The
[feature set](features.md) and [use cases](use-cases.md) describe the supported
boundaries; target-specific validation remains separate from the API guide.
