# Getting started

## Install

These guides describe the unreleased `0.2.0` API on `main`. Until it is
published, use a Git dependency for evaluation:

```toml
[dependencies]
stack-algebra = { git = "https://github.com/yongkyuns/stack-algebra.git" }
```

For a reproducible build, replace the moving branch with a reviewed `rev`.
The newest crates.io release is `0.1.0`; it predates most APIs documented here.
The development tree is `no_std` and its fixed-capacity core does not require
`alloc`, but it currently has a [Miri release blocker](DESIGN_REVIEW.md#confirmed-release-blocker).
See the [feature set](FEATURES.md) for storage and solver boundaries.

## Construct a matrix

Dimensions are const generics and the scalar type is explicit when needed:

```rust
use stack_algebra::{matrix, Matrix};

let state: Matrix<3, 1, f32> = matrix![1.0; 2.0; 3.0];
let covariance: Matrix<3, 3, f64> = Matrix::eye();
let wide: Matrix<3, 1, f64> = state.cast();
```

For common fixed dimensions, the aliases `Vector`, `RowVector`, `Matrix2`,
`Matrix3`, and `Matrix4` avoid repeating const generic parameters. The aliases
ending in `f` and `d` select `f32` and `f64`, respectively.

Dimension mismatches and implicit mixed-precision arithmetic are rejected by
the compiler. Use `cast` at an explicit precision boundary.

## Choose the next guide

- Use [API usage](API_USAGE.md) for storage, views, products, and solver
  selection.
- Use [Use cases](USE_CASES.md) for representative estimation, optimization,
  sparse, and embedded workflows.
- Use the [API reference](api-reference.md) for method and trait details.
