# Release process

`stack-algebra` separates portable-library release qualification from optional performance evidence. A release does not require physical hardware, but it must not make hardware-specific performance claims without named-device measurements.

## Supported Rust version

The declared MSRV is **Rust 1.87**. CI builds the library with Rust 1.87 for both `no_std` and `std` configurations. Raising the MSRV is a compatibility change and should be called out in the changelog/release notes.

The floor is evidence-based: Rust 1.85 exposed the library's use of `usize::is_multiple_of`, which became stable in Rust 1.87. The embedded resource qualification toolchain is separately pinned to Rust 1.98.0 so code-size/stack budgets do not drift merely because the hosted `stable` toolchain moves.

## Release candidate checklist

1. Update `CHANGELOG.md` and remove any stale `Unreleased` statements that belong to the release.
2. Set the exact release version in `Cargo.toml`.
3. Ensure Build, API stability, Documentation, and Release artifact qualification are green on the exact release commit.
4. Run **Release artifact qualification** on that commit and retain the package/API/dependency/provenance artifact.
5. If publishing canonical cross-library performance claims, run the pinned self-hosted **Release benchmark qualification** workflow on the same commit.
6. Do not require physical-target timing to ship the portable crate. If no named-device evidence exists, avoid timing/throughput/board-specific claims.
7. Run the manual **Release** workflow with `publish=false` and the expected version as a final dry validation.
8. After review, rerun **Release** with `publish=true`. Publication requires a `CRATES_IO_TOKEN` repository secret.
9. Tag the exact published commit and publish/update the combined documentation site from `main`.

## Release workflow safety

The manual release workflow never publishes by default. It verifies that the requested version exactly matches `Cargo.toml`, checks formatting/Clippy/tests/docs/examples, and builds the Cargo package before the optional publish step. The publish path is enabled only by the explicit boolean workflow input and requires the crates.io token secret.

Tag/release-note creation remains a separate deliberate GitHub action so crate publication cannot silently create or move source-control tags.

## Runnable workload examples

The repository keeps three examples compiling in normal CI:

- `ekf_measurement_update` — a Joseph-form covariance update using a Cholesky solve instead of an explicit inverse;
- `mapped_least_squares` — column-pivoted QR directly from a caller-owned mapped Jacobian buffer;
- `embedded_resource_budget` — compile-time storage budgeting for a 15-state estimator and bounded workspace.

These examples are intended to become workload probes for future API/performance decisions. New GEMM-accumulate or broader mapped-layout kernels should be justified by measurements on these or similarly representative workloads rather than by API parity alone.

## Resource regression policy

The Cortex-M qualification suite runs on the pinned Rust 1.98.0 toolchain and enforces deliberately generous per-workload text and painted-stack ceilings. The budgets are regression alarms, not optimization targets. Increasing a ceiling is allowed when a reviewed capability justifies it, but the change should include the before/after resource report and rationale.
