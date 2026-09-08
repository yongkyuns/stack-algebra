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

## Exact-commit package qualification

Release artifact qualification runs on relevant pull requests and pushes to `main` when the source, manifest, README/licenses, changelog, safety notes, examples, contract tests, or qualification tooling changes. It can also be run manually. The workflow explicitly checks out the pull request's head commit or the pushed commit, and the script rejects a requested source SHA that differs from the checkout. A pull-request snapshot does not replace qualification of the eventual merged release commit.

The Cargo package uses an explicit consumer-facing allowlist: library source, runnable examples, README, changelog, `UNSAFE.md`, and the two license files. Repository-only infrastructure such as GitHub workflows, benchmark sources, guide sources, QEMU harnesses, scripts, tests, and development tools is intentionally excluded from the `.crate`. Qualification fails if any of those repository-only paths reappear in the package inventory.

The artifact retains the `.crate` archive, package file inventory, public API listing, rustdoc JSON, dependency metadata, and source/toolchain/checksum provenance. It also extracts that archive and uses it as the dependency of a separate temporary consumer, rather than building against the repository's working copy.

The external-consumer checks execute the Cholesky quick start both with the library's default `no_std` configuration and with `std` enabled. They also run the matrix swap, safe scalar-hook, matrix storage, sparse reuse, sparse empty-shape, and sparse permutation contract suites against the packaged library in debug/default, debug/`std`, and release/default configurations. This verifies failure and storage behavior through the actual packaged API, including optimized builds where debug assertions are absent. The consumer itself is a host executable; embedded portability remains covered by the separate target CI.

Consumer execution logs, all six contract-test sources, lockfile, dependency tree, and metadata are retained with the package evidence. Any compile, link, runtime, package-boundary, or test failure fails qualification. The workflow has read-only repository permissions and never publishes, tags, or changes the version. A successful development snapshot is preparation for a release candidate, not a claim that a new release version has been published.

## Release workflow safety

The Release workflow explicitly installs Rust 1.98.0 with `rustfmt` and `clippy`; validation does not rely on those optional components already being present on the runner. It checks out the selected commit without persisting Git credentials, then checks formatting/Clippy/tests/docs/examples and builds the Cargo package.

Pull requests that change `.github/workflows/release.yml` exercise these same validation commands on the exact PR head. This is a non-publishing workflow check: it skips the manual version-input check and cannot run the publish step. Each PR has a separate concurrency group; manual release runs remain serialized in the existing `stack-algebra-release` group. This check does not replace the final manual `publish=false` validation or exact-release package qualification.

Manual dispatch still requires the expected version to exactly match `Cargo.toml`, and publication remains disabled by default. The publish step requires both a `workflow_dispatch` event and an explicit `publish=true` boolean input, as well as the crates.io token secret. PR validation does not require that secret. Repository permissions remain read-only.

Tag/release-note creation remains a separate deliberate GitHub action so crate publication cannot silently create or move source-control tags.

## Runnable teaching examples

The repository keeps three examples executing in normal CI:

- `kalman_1d` — a two-state position/velocity filter with scalar position observations and a readable 2x2 Joseph covariance update;
- `mapped_least_squares` — column-pivoted QR directly from a caller-owned mapped Jacobian buffer;
- `embedded_resource_budget` — compile-time storage budgeting for a 15-state estimator and bounded workspace.

These examples teach isolated library operations, not complete applications. The Kalman model assumptions and expected output are documented in [Tutorials](tutorials.md). Its integration tests remain separate from the example. New GEMM-accumulate or broader mapped-layout kernels require measurements on representative consumer workloads; teaching examples alone are not performance justification.

## Resource regression policy

The Cortex-M qualification suite runs on the pinned Rust 1.98.0 toolchain and enforces deliberately generous per-workload text and painted-stack ceilings. The budgets are regression alarms, not optimization targets. Increasing a ceiling is allowed when a reviewed capability justifies it, but the change should include the before/after resource report and rationale.
