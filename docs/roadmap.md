# stack-algebra roadmap

This roadmap reflects the current `0.3.0-alpha.1` development line. `stack-algebra` targets predictable linear algebra for embedded and robotics workloads rather than general large dynamic dense/sparse computing.

## Design commitments

- Keep `Matrix<M, N, T>` fixed-size, column-major, `no_std`, and allocation-free.
- Keep scalar/packet kernel selection at compile time; avoid runtime dispatch in the embedded path.
- Preserve explicit scalar types and explicit casts for mixed precision.
- Prefer fixed, bounded, mapped, and fixed-capacity sparse storage over heap-owning dynamic matrices.
- Introduce any future dynamic owning layer only behind explicit `alloc`/`std` support and only for demonstrated workloads.
- Treat Eigen, faer, and nalgebra as external correctness/performance references, not API targets.

## Established in the 0.3 line

### Public API and storage

- Fixed-size dense matrices/vectors and compile-time dimensions.
- Shared `MatrixRead`/`MatrixWrite` view foundation, mapped/strided/block views, and owned snapshots through `Matrix::from_view`.
- `MatrixBuf<MAX_ROWS, MAX_COLS, T>` bounded runtime-active storage with checked resize forms.
- Reusable factor objects and consistent solve/recompute conventions.
- Actionable fixed-capacity sparse errors carrying required and available capacity.

### Dense numerical capability

- Cholesky, pivoted and non-pivoted LDLT, partial-pivot LU, Householder QR, column-pivoted QR, SVD, and self-adjoint eigendecomposition.
- Dense Bunch–Kaufman LDLT with compact 1x1/2x2 pivot metadata.
- Lower/upper self-adjoint views and checked symmetry validation.
- Solver invariant coverage for reconstruction, residuals, orthogonality, rank, pivoting, reuse, and failure semantics.

### Sparse and block-sparse capability

- Fixed-capacity scalar CSC storage plus symbolic/numeric Cholesky and LDLT reuse.
- Bounded sparse diagonal pivoting and explicit dense fallback for cases requiring global pivot behavior.
- Fixed-capacity block CSC/CSR storage, native block Cholesky, native block LDLT with local Bunch–Kaufman pivots, ordering support, and explicit dense global-pivot fallback.
- Scalar expansion paths retained as reference/comparison mechanisms rather than hidden fallback behavior.

### Performance architecture

- Portable scalar kernels remain the reference implementation.
- x86 SSE2/AVX2/FMA and AArch64 NEON backends are selected at compile time.
- Contiguous column-major mapped products reuse optimized owned-matrix kernels without copying; arbitrary strides stay on generic zero-copy loops.
- Explicit fused operations cover common estimation/control forms (`axpy_*`, `linear_combination_into`).
- Short hosted benchmarks are regression triage; release benchmark evidence has a separate pinned-machine contract.

### Validation and release infrastructure

- CI covers formatting, Clippy, host tests, docs, API/semver checks, Miri, cross-target builds, native AArch64 tests, and representative QEMU execution.
- Rust 1.87 is the declared MSRV and is built in CI for both `no_std` and `std` library configurations.
- Three executable robotics/embedded examples are compiled on every PR and serve as workload probes for future API decisions.
- Cortex-M qualification records isolated code/static size and painted-stack high-water marks with source/tool provenance and per-workload regression budgets on a pinned toolchain.
- A physical Cortex-M DWT timing harness exists and is kept buildable, but no named-board timing result is currently claimed.
- Release artifact qualification captures the crate package, generated dependency lock, public API listing, rustdoc JSON, dependency metadata, and provenance.
- A guarded manual release workflow verifies the requested version and package before optional crates.io publication.

## 0.3 release priorities

Before publishing `0.3.0`:

1. Keep API/semver changes intentional and documented.
2. Keep every public solver covered by invariant-based numerical evidence.
3. Keep failure and capacity semantics predictable/actionable.
4. Keep the Rust 1.87 MSRV and representative examples green.
5. Capture the release artifact snapshot for the exact release commit.
6. Run the release benchmark workflow on a deliberately pinned machine **before publishing cross-library release performance claims**.
7. Keep README/docs explicit that QEMU/static resource evidence is not physical-device timing evidence.

A physical board measurement is desirable follow-up evidence, but it is **not a portable-library release blocker**. It becomes mandatory before making real-device timing, throughput, or board-specific performance claims.

## Next priorities after 0.3

Work should be workload-driven rather than assigned to old phase/version buckets.

- Validate optimized kernels on additional maintained architectures before adding new ISA-specific code.
- Use the runnable estimation examples to decide whether a GEMM-like accumulate API materially reduces temporaries or runtime.
- Add broader leading-dimension/layout kernels only when mapped-workload benchmarks show material benefit.
- Add sparse symbolic/workspace preflight sizing where it improves fixed-capacity planning without hiding allocation.
- Improve sparse/block ordering or cross-block pivot behavior only where bounded storage semantics remain explicit.
- Add additional physical-target evidence when hardware and maintenance capacity are available.

## Deliberately deferred

- heap-owning fully dynamic matrices in the core;
- general Eigen-style expression templates;
- general runtime sparse indefinite solving;
- GPU/accelerator backends;
- per-shape hand-written kernels without measured need;
- new ISA families without maintained validation;
- broad geometry expansion unrelated to the linear-algebra core.
