# Release artifact qualification

`0.3` release candidates should preserve more than a version number. The release
artifact workflow captures the crate package, public API surface, dependency
resolution, and provenance needed to review what is actually being published.

## Workflow

The manual **Release artifact qualification** workflow uses Rust 1.98.0 for the
release/package checks and a date-pinned nightly (`nightly-2026-08-20`) for
rustdoc JSON generation. It also pins `cargo-public-api` 0.51.0. The workflow is
run automatically on pull requests only when the qualification workflow or its
script changes, so ordinary PRs do not repeatedly pay the tool-install and
rustdoc-JSON cost.

The implementation lives in `scripts/build_release_artifacts.sh`. It verifies a
clean source checkout, records the source commit/ref before generating files,
and records the PR source SHA separately from GitHub's synthetic merge checkout
when applicable.

This library does not intentionally commit a root `Cargo.lock`. Qualification
resolves a fresh lockfile after the clean-source check and then uses and
archives that exact dependency resolution.

## Produced evidence

A successful run uploads `release-artifacts-<sha>` containing:

- the verified `.crate` archive produced by Cargo;
- `public-api.txt`, a human-readable public API snapshot;
- `rustdoc-public-api.json`, the machine-readable rustdoc JSON used for deeper
  inspection or future tooling;
- `cargo-metadata.json` and `dependency-tree.txt`;
- the exact generated `Cargo.lock` used by the qualification;
- `commit.txt`; and
- `provenance.txt` with toolchain versions and SHA-256 hashes of the lockfile,
  package, public API snapshot, and rustdoc JSON.

Cargo package verification is part of the qualification gate rather than only
creating an archive, so packaging/build failures prevent the evidence from being
accepted.

## Public API policy

The PR semver workflow remains the fast compatibility gate. The release API
snapshot has a different purpose: it creates a durable artifact that can be
attached to a release candidate and diffed later without depending on the
current state of a third-party API-diff service or a moving nightly compiler.

Because rustdoc JSON is still tied to nightly Rust, both the nightly date and
`cargo-public-api` version are part of the recorded contract. Changing either
creates a new snapshot baseline and should be called out in release notes.

## Release use

Before publishing `0.3.0`, run **Release artifact qualification** on the exact
release commit and retain the resulting package/API/dependency/provenance
artifact.

Host-performance and embedded evidence are separate qualification classes:

- **release artifacts** answer what API/package/dependency surface is being
  shipped;
- **pinned-host benchmarks** answer how that exact release performs on a
  controlled host, and are required before publishing canonical cross-library
  release-performance claims;
- **QEMU/static target qualification** records reproducible portable embedded
  code/static/painted-stack evidence; and
- **physical target qualification** records what was actually observed on a
  named device when such hardware evidence exists.

A physical-target artifact is **not required merely to publish the portable
`0.3` library**. If no physical measurement exists, the release must avoid
real-device timing, throughput, cache, peripheral, or board-specific performance
claims and state that limitation explicitly. Any future hardware-specific
performance claim requires named physical hardware and controlled measurement
conditions.
