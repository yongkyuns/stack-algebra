# stack-algebra

`stack-algebra` is a standalone Rust linear-algebra library. Its public API
uses compile-time matrix dimensions and explicit scalar types, with bounded,
mapped, and sparse storage APIs for cases that need runtime-active data or
caller-owned buffers.

## Start here

- [Getting started](getting-started.md) — construct matrices and choose scalar
  types.
- [Tutorials](tutorials.md) — follow workflows by storage layout and solver
  assumptions.
- [API usage](api-usage.md) — select storage, views, products, and solvers.
- [Feature set](features.md) — review supported operations and boundaries.
- [Use cases](use-cases.md) — map common estimation, optimization, geometry,
  and sparse workflows to the API.
- [Benchmarking](benchmarking.md) — run and interpret numerical comparisons.
- [Target support and evidence](targets.md) — distinguish host builds, QEMU
  execution, and real-hardware validation.
- [Roadmap](roadmap.md) — see planned capability and validation work.
- [API reference](api-reference.md) — browse generated Rust documentation.

```rust
use stack_algebra::{matrix, Cholesky};

let matrix = matrix![4.0_f64, 1.0; 1.0, 3.0];
let rhs = matrix![1.0_f64; 2.0];
let factor = Cholesky::try_decompose(&matrix).expect("positive definite");
let solution = factor.solve(&rhs);
assert!((matrix * solution - rhs).norm() < 1.0e-12);
```

## Contents

- **Dense algebra** — fixed-size matrices, products, reductions, and
  decompositions.
- **Storage and views** — bounded buffers, external maps, strided layouts, and
  zero-copy blocks.
- **Geometry** — quaternions, rotation matrices, isometries, and affine
  transforms.
- **Sparse systems** — scalar and block CSC/CSR storage and factorization.
- **Validation** — correctness comparisons, benchmarks, target checks, and
  safety-validation reports.
