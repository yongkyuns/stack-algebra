# stack-algebra

`stack-algebra` provides predictable linear algebra for embedded and robotics workloads: small-to-medium compile-time or tightly bounded matrices, explicit storage and workspace reuse, zero-copy caller buffers, and fixed-capacity sparse/block-sparse solvers that work in `no_std` environments.

The project is intentionally not a general Eigen/nalgebra/faer replacement for large dynamic workloads.

## 0.3 development status

The current development line is `0.3.0-alpha.1`. The software qualification work covers API compatibility checks, solver invariants, host/native tests, Miri, cross-target builds, QEMU execution, Cortex-M static resource/stack reports, and reproducible release artifacts.

Physical-device timing is **not** currently available and is not required to publish the portable library. Until named-board measurements exist, the project makes no real-device timing or throughput claims. Release-level cross-library performance claims separately require the pinned-machine benchmark procedure.

## Start here

- [Getting started](getting-started.md) — construct matrices and choose scalar types.
- [Tutorials](tutorials.md) — follow workflows by storage layout and solver assumptions.
- [API usage](api-usage.md) — select storage, views, products, and solvers.
- [Feature set](features.md) — review supported operations and boundaries.
- [Use cases](use-cases.md) — map estimation, optimization, geometry, and sparse workflows to the API.
- [0.3 stabilization plan](stabilization-plan.md) — current release contract and remaining gates.
- [Solver invariant qualification](solver-qualification.md) — numerical evidence by public solver family.
- [Target support and evidence](targets.md) — distinguish compile, QEMU, resource, and physical-device evidence.
- [Benchmarking](benchmarking.md) — run and interpret development comparisons.
- [Release benchmark qualification](release-benchmarking.md) — requirements for publishable host performance evidence.
- [Release artifact qualification](release-artifacts.md) — package/API/dependency provenance for a release candidate.
- [Roadmap](roadmap.md) — established capabilities and future priorities.
- [API reference](api-reference.md) — browse generated Rust documentation.

```rust
use stack_algebra::{matrix, Cholesky};

let matrix = matrix![4.0_f64, 1.0; 1.0, 3.0];
let rhs = matrix![1.0_f64; 2.0];
let factor = Cholesky::try_decompose(&matrix).expect("positive definite");
let solution = factor.solve(&rhs);
assert!((matrix * solution - rhs).norm() < 1.0e-12);
```

## Scope

- **Dense algebra** — fixed-size matrices, products, reductions, and decompositions.
- **Storage and views** — bounded buffers, external maps, strided layouts, and zero-copy blocks.
- **Geometry** — quaternions, rotation matrices, isometries, and affine transforms.
- **Sparse systems** — fixed-capacity scalar and block CSC/CSR storage and factorization.
- **Validation** — invariant tests, differential references, target checks, resource reports, and reproducible release artifacts.
