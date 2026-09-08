# Changelog

All notable changes to `stack-algebra` are documented here. The project follows semantic versioning for the public Rust API.

## 0.3.0 — Pending publication

This section prepares the stable `0.3.0` release; no publication date or release tag is asserted. It builds on the development milestone below. Final qualification must identify the exact release commit before publication.

### Added

- Rust 1.87 minimum-supported-Rust-version declaration and CI gate.
- Executable estimation, mapped least-squares, and embedded resource-budget examples.
- Guarded manual release workflow with version verification, package validation, and optional crates.io publication.
- Reproducible Cortex-M code-size and stack regression budgets on the pinned qualification toolchain.
- Generated benchmark reference pages and retained benchmark provenance for the hosted performance snapshot.
- Extracted-package qualification runs six public consumer contract suites in debug/default, debug/`std`, and release/default configurations, including workspace rebuild, rejection-before-mutation, and inactive-value preservation tests.

### Fixed

- Matrix storage and row views handle empty dimensions, zero-sized values, length overflow, and partially initialized iterator collection without invalid pointer arithmetic or leaked initialized values.
- `f32`/`f64` norms avoid premature square overflow and subnormal underflow for extreme finite inputs.
- Zero-column sparse patterns reject hidden entries, while empty sparse analysis, factorization, recomputation, and solves remain valid.
- Reusable sparse Cholesky and LDLT validate source structure before changing reusable outputs and synchronize destination patterns when needed. Numeric source-offset caches are not reused for incompatible source layouts.
- Cached sparse permutation accepts arbitrary destination patterns, checks source-offset bounds before destination mutation, and preserves inactive backing values. The original source-pattern reuse requirement remains; same-length coordinate changes require rebuilding the map.
- Release validation explicitly installs `rustfmt` and Clippy with its pinned toolchain and checks release-workflow PRs without enabling publication.
- `Matrix::from_fn` now drops every value already produced if the callback panics, matching the panic-safety guarantee of iterator collection instead of leaking partially initialized elements.
- Safe `FactorizationScalar` and `MatrixScalar` hooks now validate matching slice lengths, block ranges, and column indices before SIMD dispatch or mutation. This closes an out-of-bounds access path reachable through safe calls; portable defaults reject invalid inputs consistently. Valid numerical operations are unchanged.

### Changed

- Eigen differential testing now lives in the non-published `tools/eigen-harness` package. The core crate no longer exposes an Eigen feature or carries a C++ build script/`cc` build dependency; comparison coverage remains in CI.
- `Matrix::swap_rows` and `Matrix::swap_columns` now panic before mutation when either index is out of bounds, instead of silently ignoring invalid indices. Valid swaps, including self-swaps and swaps across an empty counterpart dimension, are unchanged. Callers that previously relied on the silent no-op must validate indices explicitly.
- Dense LDLT reusable multi-RHS solves reduce temporary storage and reuse factor traversal across RHS columns.
- Dense triangular multi-RHS solves use column-update traversal for measured larger-dimension cases while preserving small-size paths.
- x86 AVX2/FMA `f32` dot reduction uses shorter packet paths and in-register reduction for small and medium vectors.
- Performance documentation distinguishes accepted production improvements, hosted nightly snapshots, and pinned release qualification.
- `0.3` documentation treats physical-device timing as optional evidence while prohibiting unmeasured hardware performance claims.
- Release qualification distinguishes short hosted regression measurements, pinned-host release comparisons, and physical-target timing evidence.

### Evidence scope

Existing hosted measurements are historical, workload-specific regression evidence, not cross-library performance claims for this release. No physical-device timing claim is made. The rejected cached-permutation optimization from PR #54 and its diagnostic rewrite are not included.

## 0.3.0-alpha.1 development milestone — 2026-08-21

This heading records development history, not a verified registry publication or release tag.

### Added

- Bounded runtime-active `MatrixBuf` storage and explicit initialized-growth resize APIs.
- Fixed-capacity scalar and block sparse storage, symbolic reuse, Cholesky and LDLT paths.
- Dense Cholesky, LDLT, partial-pivot LU, Householder QR, column-pivoted QR, SVD, and self-adjoint eigendecomposition qualification.
- Compile-time x86 SSE2/AVX2/FMA and AArch64 NEON kernel selection.
- Optimized contiguous mapped-view product paths and fused `axpy_*` / `linear_combination_into` operations.
- API/semver checks, Miri coverage, cross-target/QEMU validation, release-artifact qualification, and Cortex-M resource reporting.

### Changed

- Sparse capacity failures report both required and available capacity.
- Factorization/update scalar behavior is separated from matrix-product/reduction specialization through `FactorizationScalar`.

