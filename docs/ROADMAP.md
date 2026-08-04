# stack-algebra roadmap

This document defines the path from the current fixed-size core to a stable,
ergonomic, high-performance Rust algebra library for native and embedded
robotics workloads. The crate remains standalone and has no dependency on
SymForce or any other robotics framework.

## Design commitments

- Keep `Matrix<M, N, T>` fixed-size, column-major, `no_std`, and allocation-free.
- Keep scalar/packet kernel selection at compile time. Do not introduce runtime
  kernel dispatch into the embedded path.
- Preserve explicit `f32`/`f64` types and explicit casts for mixed precision.
- Add bounded runtime dimensions before adding heap-backed dynamic matrices.
- Treat Eigen and faer as external correctness and performance references only.

## Current status

- P0 sparse triangle semantics, compact dense permutations, typed dense
  decomposition failures, and target smoke validation are implemented.
- The P1 shared `MatrixRead`/`MatrixWrite` view foundation and
  `Matrix::from_view` are implemented; algorithms still materialize owning
  matrices until view-native kernels are added.
- The bounded `MatrixBuf<MAX_ROWS, MAX_COLS, T>` storage layer is implemented
  with checked active dimensions and no heap allocation.
- The first block-sparse layer, `StaticBlockCscMatrix`, is implemented with
  fixed-capacity CSC patterns and allocation-free block matvec. Block CSR and
  sparse block factorizations remain future work.

## Execution phases

### P0 — correctness and public API foundation

- Make sparse lower/upper triangle semantics explicit and Eigen-compatible.
- Replace dense permutation matrices in factorizations with compact index
  permutations shared by dense and sparse algorithms.
- Standardize reusable `*_into` and `*_in_place` solver APIs.
- Introduce structured decomposition errors while preserving compatibility
  shims where a breaking change is not yet justified.
- Split the monolithic sparse implementation into storage, ordering, LLT, and
  LDLT modules after behavior is covered by regression tests.
- Audit fixed-capacity workspaces and document or bound their RAM footprint.

### P1 — zero-copy and workspace ergonomics

- Add compile-time-dimension read/write view abstractions for `Map`, `Block`,
  and strided views.
- Allow products and decompositions to consume views without materializing a
  new owning matrix.
- Make caller-provided workspaces first-class for factorization and solves.
  Cholesky, LDLT, LU, QR, SVD, and self-adjoint eigendecomposition now expose
  the unified `compute`/`try_compute` recomputation convention; workspace-
  specific scratch buffers remain a follow-up measurement.

### P2 — portable performance architecture

- Keep portable scalar kernels as the reference implementation.
- Maintain separate ISA modules for x86 SSE2/AVX2/FMA and AArch64 NEON;
  add Arm32 NEON and RVV only with representative benchmarks and tests.
- Use packet-width tails and general blocking policies rather than
  size-specific bespoke kernels.
- Validate each optimized kernel against the scalar reference with numerical
  tolerances appropriate for FMA and reduction-order differences.

### P3 — numerical capability

- Match Eigen's dense diagonal-pivot LDLT behavior, including pivot
  selection, permutation conventions, and zero-pivot reporting. Keep any
  Bunch–Kaufman 2x2 implementation as a separate opt-in algorithm rather than
  changing the Eigen-compatible default.
- Add sparse pivoted LDLT with the same documented threshold and failure model.
- Strengthen QR, SVD, and self-adjoint eigensolver scaling, rank, convergence,
  and tolerance behavior.
- Add explicit lower/upper self-adjoint views, with strict symmetry checking as
  an opt-in mode. The zero-copy lower/upper views and checked constructors are
  implemented.

### P4 — bounded and sparse storage

- Add a bounded runtime-size matrix (`MatrixBuf<MAX_R, MAX_C, T>`-style) with
  no allocation and an Eigen `MaxRows`/`MaxCols`-like capacity contract.
- Add fixed-capacity block CSC/CSR for block-sparse robotics systems.
- Add heap-backed dynamic matrices only behind an optional `alloc`/`std` layer.

### P5 — release and target gates

- Differential tests against Eigen and faer for `f32`/`f64`, multi-RHS solves,
  malformed inputs, non-finite values, near-singular thresholds, ordering, and
  symbolic/numeric reuse.
- Fair benchmarks separating symbolic analysis, checked factorization, reused
  numeric factorization, and solve. Match storage, ordering, compiler flags,
  scalar type, dimensions, RHS count, and allocation model.
- CI builds portable, SSE2, AVX2/FMA, AArch64, `thumbv6m`, Cortex-M4F,
  RISC-V/ESP32-C3-class, and WASM targets; execute QEMU smoke tests where the
  machine model is meaningful.
- Use real STM32/ESP hardware smoke and cycle measurements for peripheral and
  device claims; QEMU alone cannot validate those peripherals.
- Run Miri/sanitizer checks for view and SIMD unsafe code, plus no-allocation
  checks for fixed sparse paths.

## Release intent

- `0.3`: P0 and P1 foundations with stable fixed-size behavior.
- `0.4`: P2 performance portability and robust dense solver behavior.
- `0.5`: P3/P4 sparse pivoting, bounded runtime dimensions, and block sparse
  storage.

Each phase is complete only when correctness parity, target checks, and
representative performance evidence are recorded alongside the implementation.
