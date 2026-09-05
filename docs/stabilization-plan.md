# 0.3 stabilization plan

Status: software qualification substantially complete; release-candidate evidence remains.

`0.3` is a stabilization release for the existing dense, bounded, sparse, block-sparse, geometry, SIMD, and embedded-portability surface. Major algorithm expansion remains secondary to making the current contract predictable and measurable.

The target position is:

> Predictable linear algebra for embedded and robotics workloads: small-to-medium compile-time or tightly bounded matrices, explicit storage and workspace reuse, zero-copy caller buffers, and fixed-capacity sparse/block-sparse solvers that work in `no_std` environments.

The project is not intended to replace Eigen, nalgebra, or faer for general large dynamic workloads.

## Release contracts

1. **API contract** — naming/failure behavior is predictable and accidental public breakage is caught in CI.
2. **Numerical contract** — solver correctness is specified by reconstruction, residual, orthogonality, rank, pivot, and convergence invariants; reference-library agreement is secondary evidence.
3. **Resource contract** — representative operations have reproducible storage, stack, code-size, and target evidence with explicit evidence levels.
4. **Scope contract** — unsupported workloads and unmeasured hardware claims are stated directly.

The development version is `0.3.0-alpha.1` while intentional contract changes are being finalized.

## Completed stabilization work

### Release/API qualification

- PR semver/API compatibility checking against the base revision.
- Compile-time rejection tests for incompatible dimensions/scalars.
- Public API smoke coverage across dense, bounded, view, solver, geometry, and sparse entry points.
- Rust 1.87 is the tested/declarative MSRV for both `no_std` and `std` library configurations.
- A guarded manual release workflow verifies the requested version, package, tests, docs, examples, and optional crates.io publication.
- Manual release-artifact workflow with pinned tools, generated dependency lock, package verification, human-readable API snapshot, rustdoc JSON, dependency metadata, and provenance.

### Numerical qualification

- Invariant-based coverage across public solver families.
- `f32`/`f64`, multiple scales, multiple RHS, singular/rank-deficient and failure cases where applicable.
- Independent round-trip/reconstruction checks rather than same-algorithm equivalence where practical.
- Eigen differential tests retained as secondary evidence.
- A versioned solver-evidence matrix in [Solver invariant qualification](solver-qualification.md).

### Public contract cleanup

- `FactorizationScalar` separates portable factor/update behavior from matrix-product/reduction specialization.
- `MatrixBuf` has explicit storage-preserving and initialized-growth resize forms.
- Sparse capacity failures report `required` and `capacity`.
- Recompute, `_into`, and `_in_place` conventions are aligned across the common factor/operation surface.

### Views and fused operations

- Contiguous column-major `Map`/compatible `StridedMap` products and matvecs reuse optimized owned kernels without copying.
- Arbitrary/padded strides remain on the generic zero-copy path.
- `axpy_in_place`, `axpy_into`, and `linear_combination_into` cover common estimation/control forms.
- Focused fused-operation benchmarks are part of regression triage.
- Representative EKF, mapped least-squares, and embedded storage-budget examples execute in normal CI and serve as workload probes for later performance/API work.

### Embedded/resource qualification

- Cross-target builds and representative Cortex-M/RISC-V32/AArch64 QEMU smoke execution.
- Reproducible Cortex-M isolated workload tables for code/static size and painted-stack high-water marks.
- Per-workload Cortex-M text and painted-stack budgets are enforced on a pinned Rust 1.98.0 qualification toolchain to catch accidental resource explosions.
- Source/tool/build provenance is captured with those reports.
- A physical Cortex-M DWT timing harness shares the same workload definitions and is kept buildable in CI.

Physical timing has not been measured on a named board. That absence is documented as an evidence limitation rather than represented as a failed portable-library qualification.

## Release benchmark policy

Short GitHub-hosted runs are regression triage only. Release-quality host performance evidence must use the dedicated pinned-machine procedure, record machine/toolchain/ISA/dependency provenance, execute longer sequential measurements, and retain raw measurements.

A pinned-machine run is required before publishing cross-library performance claims for the exact release. It is not necessary to invent a canonical performance number when no stable benchmark host is available; in that case, release without a cross-library release-performance claim.

## Remaining 0.3 release checklist

- [x] API/semver changes are intentional and documented.
- [x] Every public solver family has invariant-based numerical evidence.
- [x] Common failure semantics and bounded/sparse capacity failures are documented/tested.
- [x] README/docs state the supported workload envelope and avoid general Eigen/faer replacement language.
- [x] Rust 1.87 MSRV is declared and continuously tested.
- [x] Representative robotics/embedded examples execute in CI.
- [x] Cortex-M QEMU/static resource evidence is reproducible, provenance-carrying, and protected by regression budgets.
- [x] A physical Cortex-M timing harness exists and remains buildable.
- [ ] Capture the release artifact snapshot for the **exact** `0.3.0` release commit.
- [ ] Run pinned-host release benchmarks for the exact release commit **if** cross-library release performance claims will be published.
- [ ] Publish/update the combined documentation site from `main` for the release.

A named physical embedded target measurement is **not a `0.3` release blocker**. It is required before making real-device timing, throughput, or board-specific performance claims.

## Follow-up work

- Use the representative workload examples to decide whether GEMM-like accumulation materially reduces temporaries/runtime.
- Add broader leading-dimension/layout kernels only with measured benefit.
- Add sparse symbolic/workspace preflight sizing where it materially improves fixed-capacity planning.
- Add another ISA family only with maintainable validation.
- Record physical Cortex-M and a second real target when hardware becomes available.
- Continue to defer heap-owning fully dynamic matrices, general expression templates, general runtime sparse indefinite solving, GPU backends, and per-shape kernels without measured need.
