# Changelog

All notable changes to `stack-algebra` are documented here. The project follows semantic versioning for the public Rust API.

## [Unreleased]

### Added

- Rust 1.87 minimum-supported-Rust-version declaration and CI gate.
- Executable estimation, mapped least-squares, and embedded resource-budget examples.
- Guarded manual release workflow with version verification, package validation, and optional crates.io publication.
- Reproducible Cortex-M code-size and stack regression budgets on the pinned qualification toolchain.

### Changed

- `0.3` documentation now treats physical-device timing as optional evidence while prohibiting unmeasured hardware performance claims.
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
