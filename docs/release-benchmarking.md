# Release benchmark qualification

Nightly benchmarks are regression triage. Release performance evidence has a stricter contract: the machine must be intentionally selected and stable, the toolchain/dependency resolution must be recorded, the checkout must be clean, and raw measurements must remain attached to the report.

## Dedicated benchmark runner

The manual **Release benchmark qualification** workflow runs only on a self-hosted Linux x86-64 runner carrying the `stack-algebra-benchmark` label. That label should identify one deliberately maintained host for a release series, with documented CPU, firmware/power configuration, and OS setup.

The workflow pins Rust 1.98.0 and expects Eigen, Python 3, Git, `sha256sum`, and the normal native build toolchain to be preinstalled. It does not mutate the benchmark host with package-manager upgrades immediately before timing.

The default release profile uses 1 second Criterion warmup, 3 seconds measurement, 30 Criterion samples, 15 Eigen samples, and a 50 ms minimum Eigen sample duration. Benchmark groups execute sequentially to avoid competing benchmark processes.

## Dependency lock and provenance

This library does **not** intentionally commit a root `Cargo.lock`. Cargo resolves a lockfile for the qualification checkout; the release benchmark must preserve and hash the exact generated `Cargo.lock` used by that run. That generated lock is evidence for the benchmark dependency graph, not a claim that the library repository normally tracks an application-style lockfile.

Every successful release benchmark records:

- machine identifier and runner name;
- source commit/ref and pre-generation checkout cleanliness;
- runner OS/architecture, CPU model, and kernel;
- exact Rust and Cargo versions;
- SHA-256 of the generated `Cargo.lock` used for the run;
- Eigen include path;
- Rust/C++ target flags;
- Criterion/Eigen measurement parameters.

The artifact retains that lockfile, source commit, raw Eigen output/CSV, raw Criterion measurements consumed by the report, generated CSV, HTML report, and provenance.

Do not compute release-to-release percentage changes across different machine identities, CPU configurations, compilers, ISA flags, dependency resolutions, or benchmark semantics. Treat those as different baselines.

## Running on the pinned host

```text
BENCH_MACHINE_ID=my-pinned-benchmark-host \
EIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
sh scripts/run_release_benchmarks.sh
```

By default the script refuses a dirty Git checkout. `ALLOW_DIRTY_BENCHMARK=1` is for exploratory work only and should not be used for publishable release evidence.

## Release policy

A pinned-machine run is required before publishing **cross-library release performance claims** for the exact release commit. If no stable benchmark host is available, the portable library may still be released, but release notes should omit canonical comparative performance claims and point to development/nightly results only as non-release regression evidence.

This host benchmark says nothing about embedded timing. Named-device measurements remain a separate evidence tier described in [Target qualification](target-qualification.md).
