# Changelog

All notable changes to `stack-algebra` are documented here. The project follows semantic versioning for the public Rust API.

## [Unreleased]

### Added

- Rust 1.87 minimum-supported-Rust-version declaration and CI gate.
- Executable estimation, mapped least-squares, and embedded resource-budget examples.
- Guarded manual release workflow with version verification, package validation, and optional crates.io publication.
- Reproducible Cortex-M code-size and stack regression budgets on the pinned qualification toolchain.
- Generated benchmark reference pages and retained benchmark provenance for the hosted performance snapshot.

### Fixed

- Safe `FactorizationScalar` and `MatrixScalar` hooks now validate matching slice lengths, block ranges, and column indices before SIMD dispatch or mutation. This closes an out-of-bounds access path reachable through safe calls; portable defaults reject invalid inputs consistently. Valid numerical operations are unchanged.

### Changed

- `Matrix::swap_rows` and `Matrix::swap_columns` now panic before mutation when either index is out of bounds, instead of silently ignoring invalid indices. Valid swaps, including self-swaps and swaps across an empty counterpart dimension, are unchanged. Callers that previously relied on the silent no-op must validate indices explicitly.
- Dense LDLT reusable multi-RHS solves reduce temporary storage and reuse factor traversal across RHS columns.
- Dense triangular multi-RHS solves use column-update traversal for measured larger-dimension cases while preserving small-size paths.
- x86 AVX2/FMA `f32` dot reduction uses shorter packet paths and in-register reduction for small and medium vectors.
- Performance documentation distinguishes accepted production improvements, hosted nightly snapshots, and pinned release qualification.
- `0.3` documentation treats physical-device timing as optional evidence while prohibiting unmeasured hardware performance claims.
- Release qualification distinguishes short hosted regression measurements, pinned-host release comparisons, and physical-target timing evidence.

## [0.3.0-alpha.1] - 2026-08-21

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

[Unreleased]: https://github.com/yongkyuns/stack-algebra/compare/v0.3.0-alpha.1...HEAD
[0.3.0-alpha.1]: https://github.com/yongkyuns/stack-algebra/releases/tag/v0.3.0-alpha.1
