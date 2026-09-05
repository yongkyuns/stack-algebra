# Getting started

This page is the shortest path from adding the crate to choosing a useful
matrix representation and solver.

## Install

Add the crate to your Cargo project:

```toml
[dependencies]
stack-algebra = "0.2"
```

The default crate is `no_std`. The fixed-size core stores data inline and does
not require heap allocation.

## Understand the matrix type

The primary dense type is:

```text
Matrix<ROWS, COLS, Scalar>
```

`ROWS` and `COLS` are compile-time constants. The scalar type is usually `f32`
or `f64`.

```rust
use stack_algebra::{matrix, Matrix};

let state: Matrix<3, 1, f32> = matrix![1.0; 2.0; 3.0];
let covariance: Matrix<3, 3, f64> = Matrix::eye();
let state_f64: Matrix<3, 1, f64> = state.cast();
```

Shape mismatches are rejected by the compiler. Mixed `f32`/`f64` expressions
are also explicit: cast when you intentionally cross a precision boundary.

Common aliases such as `Vector`, `RowVector`, `Matrix2`, `Matrix3`, and
`Matrix4` reduce repeated type parameters. Aliases ending in `f` and `d` select
`f32` and `f64`, respectively.

## Construct and combine matrices

The convenience macros cover common fixed-size values:

```rust
use stack_algebra::{eye, matrix, vector, zeros};

let a = matrix![
    1.0_f32, 2.0;
    3.0,     4.0;
];
let x = vector![5.0_f32; 6.0];
let y = a * x;
let identity = eye!(2);
let scratch = zeros!(2, 2);
```

For code that runs repeatedly, prefer explicit output reuse when it avoids
unnecessary copies:

```rust
use stack_algebra::Matrix;

let mut product = Matrix::<2, 2, f32>::zeros();
a.mul_into(&identity, &mut product);
```

## Choose storage by ownership and shape

`Matrix` is the normal starting point, but it is not the only storage model.

| Situation | Use |
| --- | --- |
| Dimensions are known at compile time | `Matrix<M, N, T>` |
| Active dimensions vary but have a fixed maximum | `MatrixBuf<MAX_R, MAX_C, T>` |
| Data is already in a contiguous external buffer | `Map` / `MapMut` |
| Data is padded, row-major, or otherwise strided | `StridedMap` / `StridedMapMut` |
| You need a fixed submatrix without copying | `Block` / `BlockMut` |
| The matrix is sparse with a bounded pattern | Static CSC or block-sparse storage |

Views borrow the original storage. Use an owned `Matrix` only when you actually
want an owned snapshot.

## Choose a solver from the mathematics

The most important solver choice is the structure of the problem, not the API
name.

| Problem | Good starting point |
| --- | --- |
| Symmetric positive-definite system | Cholesky |
| Symmetric system that may be indefinite | LDLT |
| General square system | Partial-pivoting LU |
| Full-rank least squares | Householder QR |
| Least squares where rank detection matters | Column-pivoted QR |
| Rank-deficient or ill-conditioned least squares | SVD |
| Symmetric eigenvalue problem | Self-adjoint eigendecomposition |

For example, solve a small SPD system with Cholesky:

```rust
use stack_algebra::{matrix, Cholesky};

let a = matrix![
    4.0_f64, 1.0;
    1.0,     3.0;
];
let b = matrix![1.0_f64; 2.0];

let factor = Cholesky::try_decompose(&a).expect("A must be positive definite");
let x = factor.solve(&b);
```

If the same kind of system is solved repeatedly, keep the factor object and
recompute it instead of rebuilding surrounding storage on every iteration.
[Choosing an API](api-usage.md) shows those reuse patterns.

## `no_std` and embedded use

The fixed-size core is designed to work without `alloc`. Inline storage does
not mean a value must live on the call stack: it can be a local, a field in a
long-lived state object, a `static`, or data exposed through a borrowed view.

For embedded code, the main questions are:

1. Are the dimensions fixed or bounded?
2. Where should the storage live?
3. Is `f32` sufficient for the numerical problem?
4. Can factor/output storage be reused across loop iterations?
5. Has the final workload been measured on the actual target?

See [Platforms and embedded use](targets.md) for tested targets and the limits
of host/QEMU evidence.

## Next steps

- [Choosing an API](api-usage.md) — storage, solver, reuse, views, errors, and
  sparse patterns.
- [Common use cases](use-cases.md) — estimation, least squares, control,
  geometry, sparse systems, and MCU workflows.
- [Capabilities and limits](features.md) — the current API surface and explicit
  boundaries.
- [API reference](api-reference.md) — generated signatures, trait bounds, and
  item-level documentation.