# stack-algebra

[![Crates.io Version](https://img.shields.io/crates/v/stack-algebra.svg)](https://crates.io/crates/stack-algebra)
[![Docs.rs Latest](https://img.shields.io/badge/docs.rs-latest-blue.svg)](https://docs.rs/stack-algebra)
![Build Status](https://github.com/yongkyuns/stack-algebra/actions/workflows/ci.yml/badge.svg?branch=main)

Predictable linear algebra for embedded and robotics workloads: fixed-size and tightly bounded matrices, explicit storage, reusable workspaces, zero-copy caller buffers, and fixed-capacity sparse/block-sparse solvers that work in `no_std` environments.

`stack-algebra` is intentionally **not** a general replacement for Eigen, nalgebra, or faer for large dynamic dense/sparse workloads. Its focus is small-to-medium compile-time or bounded problems where storage, allocation behavior, and target portability matter.

## Status

The development branch is currently `0.3.0-alpha.1`. The published crates.io version may therefore lag the API documented on the development branch.

- `cargo add stack-algebra` installs the current crates.io release.
- For the current development API, use this repository/branch explicitly.
- The [0.3 stabilization plan](docs/stabilization-plan.md) records release gates and completed work.
- The [feature set](docs/features.md) is the capability/limitation reference.

## Core design

- `Matrix<M, N, T>` uses compile-time dimensions and inline column-major storage.
- The core requires no heap allocation and supports `no_std`.
- `MatrixBuf<MAX_ROWS, MAX_COLS, T>` provides bounded runtime-active dimensions.
- `Map`, `MapMut`, `StridedMap`, blocks, rows, and columns provide zero-copy views over caller-owned storage.
- Dense solvers include Cholesky, pivoted/non-pivoted LDLT, LU, Householder QR, column-pivoted QR, SVD, and self-adjoint eigendecomposition.
- Fixed-capacity scalar and block CSC/CSR types provide sparse matvec, symbolic/numeric reuse, Cholesky, and LDLT paths with explicit capacity diagnostics and fallback semantics.
- Geometry includes quaternions, rotation matrices, angle-axis, isometries, and affine transforms.
- x86 SSE2/AVX2/FMA and AArch64 NEON kernels are selected at compile time where supported; portable scalar implementations remain the reference path.

Inline storage describes representation, not physical placement. Values may live on a task stack, in static storage, inside another state object, in an arena, or in caller-owned external buffers. Applications remain responsible for choosing placement that fits their RAM/stack budget.

## Install

For the latest published release:

```sh
cargo add stack-algebra
```

Then import the items you need:

```rust
use stack_algebra::{matrix, vector, Cholesky};

let a = matrix![4.0_f64, 1.0; 1.0, 3.0];
let b = vector![1.0_f64; 2.0];
let factor = Cholesky::try_decompose(&a).expect("positive definite");
let x = factor.solve(&b);
assert!((a * x - b).norm() < 1.0e-12);
```

For unreleased `0.3` development work, pin a Git revision rather than assuming crates.io exposes the same API:

```toml
[dependencies]
stack-algebra = { git = "https://github.com/yongkyuns/stack-algebra", rev = "<commit>" }
```

## Common API patterns

```rust
use stack_algebra::{matrix, vector, Matrix, MatrixBuf};

let a = matrix![
    1.0_f32, 2.0, 3.0;
    4.0,     5.0, 6.0;
];
let x = vector![1.0_f32; 2.0; 3.0];
let y = a.matvec(&x);
assert_eq!(y, vector![14.0; 32.0]);

let mut out = Matrix::<2, 2, f32>::zeros();
let eye = Matrix::<2, 2, f32>::eye();
eye.mul_into(&eye, &mut out);

let mut bounded = MatrixBuf::<6, 6, f32>::new(3, 3).unwrap();
bounded.resize_zeroed(4, 4).unwrap();
```

Dense factors support reusable solve/output paths, and mapped contiguous column-major inputs can reuse the optimized owned-matrix kernels without copying. Padded or arbitrary-stride views remain zero-copy and use the generic view path rather than silently materializing temporary matrices.

For complete examples, see [Getting started](docs/getting-started.md), [Tutorials](docs/tutorials.md), [API usage](docs/api-usage.md), and [Use cases](docs/use-cases.md).

## Validation and performance evidence

CI covers host tests, rustdoc examples, formatting, Clippy, Miri, cross-target builds, native AArch64 tests, and QEMU smoke execution for representative Cortex-M, RISC-V32, and AArch64 paths. The project also records reproducible Cortex-M static-size and painted-stack measurements from isolated QEMU qualification workloads.

Those checks are **not physical-device timing evidence**. No STM32/ESP/other real-device performance claim is made without a named-board measurement. Physical Cortex-M timing is a follow-up evidence tier, not a prerequisite for publishing the portable `0.3` library.

Performance claims are similarly evidence-scoped:

- short GitHub-hosted runs are regression triage;
- release comparisons require the pinned-machine qualification procedure;
- embedded timing claims require physical-target measurements.

See [Target support and evidence](docs/targets.md), [Target qualification](docs/target-qualification.md), [Benchmarking](docs/benchmarking.md), [Release benchmark qualification](docs/release-benchmarking.md), and [Solver invariant qualification](docs/solver-qualification.md).

## Documentation

The repository builds a combined mdBook guide and generated Rust API site. The documentation workflow tests examples, checks links/API coverage, and uploads the complete site artifact on pull requests. GitHub Pages publishing occurs from `main` when the repository variable `PUBLISH_DOCS=true` is configured.

Start at [docs/index.md](docs/index.md). The [roadmap](docs/roadmap.md) describes what is already established in the 0.3 line and what remains intentionally deferred.

## License

Dual-licensed under MIT or Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).