# Tutorials

Learn the library through small, complete examples. Each guided walkthrough
starts with a model, explains the Rust calls in order, checks intermediate
numbers, and ends with the full runnable source and focused test commands.
The examples teach library usage, not full application frameworks or
performance benchmarks.

| Walkthrough | What you will practice |
| --- | --- |
| [Two-state Kalman filter](tutorial-kalman-1d.md) | Fixed-size state and covariance, prediction, a scalar observation, and an in-place vector correction |
| [Fit a line from a borrowed buffer](tutorial-mapped-least-squares.md) | Column-major layout, `Map`, QR, typed errors, output reuse, and residual interpretation |

## Before you start

Use Git and a stable Rust toolchain on a host computer. No microcontroller,
sensor data, Python environment, or external C++ library is needed to run
these examples. The published crate may lag the development API shown here;
use the repository checkout rather than assuming a registry version matches.

For a new checkout:

```sh
git clone https://github.com/yongkyuns/stack-algebra.git
cd stack-algebra
git rev-parse HEAD
cargo run --example kalman_1d --no-default-features
cargo run --example mapped_least_squares --no-default-features
```

Record the commit printed by `git rev-parse HEAD` when sharing results. In an
existing checkout, run the Cargo commands from the directory containing the
repository's `Cargo.toml`; there is no need to create a new application crate.
See [Getting started](getting-started.md) for using the library in your own
project and choosing an exact dependency revision.

The example executables use `std` for printing while `--no-default-features`
keeps the library in its `no_std` configuration. This is not a bare-metal
executable or a browser-playground exercise.

The walkthroughs include code from the example files in the **same source
checkout** during the guide build. Their complete listings are not separately
maintained implementations. GitHub source links point to `main`, which may
advance after a particular copy of the guide was built. Partial excerpts need
the surrounding program; use the complete listing or the Cargo command to run.

## Fixed-size dense algebra

Use `Matrix<M, N, T>` when dimensions are known at compile time. The path is:

1. Construct a matrix with `matrix!`, `Matrix::zeros`, or `Matrix::eye`.
2. Compose products and reductions with the operators and `mul_into` methods.
3. Choose a decomposition from the matrix assumptions.

See [Getting started](getting-started.md), [API usage](api-usage.md), and the
[`Matrix` API](api-reference.md).

## A two-state Kalman filter

[Follow the guided Kalman walkthrough](tutorial-kalman-1d.md) to build the
position/velocity model, inspect its first prediction and scalar correction,
and interpret the ten-sample output. It explains the discrete acceleration
noise assumption, matrix shapes, `axpy_in_place`, and the readable 2x2 Joseph
covariance expression. Exercises change one noise parameter at a time.

[Runnable source](https://github.com/yongkyuns/stack-algebra/blob/main/examples/kalman_1d.rs)
and [separate correctness tests](https://github.com/yongkyuns/stack-algebra/blob/main/tests/kalman_1d.rs)
remain small. This is a linear filter, not an EKF, ESKF, or navigation system.

## Views and external buffers

Use `Map` for contiguous column-major storage and `StridedMap` when row or
column spacing is supplied by another system. Use `Block` for fixed-size
submatrices without copying. The view types borrow their source for the view's
lifetime; use `Matrix::from_view` only when an owned snapshot is intentional.

See [API usage — external buffers and views](api-usage.md) and the generated
[view APIs](api-reference.md).

### Fit a line from a caller-owned buffer

[Follow the guided line-fitting walkthrough](tutorial-mapped-least-squares.md)
to represent `y = a*x + b` with a borrowed 5x2 design buffer, solve with
column-pivoted QR, and calculate fitted values and residuals using the same
map. It distinguishes borrowed input from the factor's own storage and
explains both a successful fit and a rank-deficient failure.

[Runnable source](https://github.com/yongkyuns/stack-algebra/blob/main/examples/mapped_least_squares.rs)
and [separate correctness tests](https://github.com/yongkyuns/stack-algebra/blob/main/tests/mapped_least_squares.rs)
show an unweighted fit to fixed perturbed observations, not a general
regression framework.

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
