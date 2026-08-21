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

The implementation lives in `scripts/build_release_artifacts.sh`. It refuses a
dirty checkout and records the source commit/ref before generating files.

## Produced evidence

A successful run uploads `release-artifacts-<sha>` containing:

- the `.crate` archive produced by `cargo package --locked`;
- `public-api.txt`, a human-readable public API snapshot;
- `rustdoc-public-api.json`, the machine-readable rustdoc JSON used for deeper
  inspection or future tooling;
- `cargo-metadata.json` and `dependency-tree.txt`;
- the exact `Cargo.lock` used by the build;
- `commit.txt`; and
- `provenance.txt` with toolchain versions and SHA-256 hashes of the lockfile,
  package, public API snapshot, and rustdoc JSON.

`cargo package --locked` performs Cargo's package verification rather than only
creating an archive, so packaging failures are part of the qualification gate.

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
release commit and retain its artifact alongside the pinned-machine benchmark
report and physical-target qualification evidence. The three evidence classes
answer different questions:

- release artifacts: what API/package/dependency surface is being shipped;
- pinned-host benchmarks: how that exact release performs on a controlled host;
- physical target qualification: what embedded resource/timing behavior was
  observed on named hardware.
