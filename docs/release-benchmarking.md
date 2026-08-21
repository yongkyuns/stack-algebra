# Release benchmark qualification

Nightly benchmarks are regression triage. Release performance evidence has a
stricter contract: the machine must be intentionally selected and stable, the
toolchain and dependency graph must be recorded, the checkout must be clean,
and the raw measurements must remain attached to the report.

## Dedicated benchmark runner

The manual **Release benchmark qualification** workflow runs only on a
self-hosted Linux x86-64 runner carrying the `stack-algebra-benchmark` label.
Do not put that label on interchangeable CI machines. It should identify the
one host used for a release series, with a documented CPU, firmware/power
configuration, and OS setup.

The workflow pins Rust 1.98.0 and invokes the release benchmark script through
`sh`. The runner must already provide Eigen at `/usr/include/eigen3`, Python 3,
Git, `sha256sum`, and the normal native build toolchain. Keeping those
prerequisites stable is part of the machine qualification; the workflow does
not mutate the host with package-manager upgrades immediately before timing.

The default release measurement profile is deliberately longer than nightly:

- Criterion warmup: 1 second;
- Criterion measurement: 3 seconds;
- Criterion sample count: 30;
- native Eigen samples: 15;
- minimum Eigen sample duration: 50 ms.

The script executes benchmark groups sequentially so concurrent benchmark jobs
do not compete for CPU resources. Environment variables can override the
measurement windows for an explicitly documented release experiment.

## Provenance contract

Every successful run records `benchmark-report/release/provenance.txt` with:

- an explicit machine identifier (`BENCH_MACHINE_ID` / GitHub runner name);
- source commit and ref;
- source checkout cleanliness before generated benchmark files are created;
- runner OS/architecture, CPU model, and kernel;
- exact Rust and Cargo versions;
- SHA-256 of the checked-in `Cargo.lock`;
- Eigen include path;
- Rust and C++ target flags;
- Criterion and Eigen measurement parameters.

The workflow also archives the exact `Cargo.lock`, source commit, raw Eigen CSV
and text output, raw Criterion measurements consumed by the report, generated
CSV, and self-contained HTML report. The artifact name includes the source SHA.

A release report is not comparable to another report if the machine identity,
CPU configuration, compiler, ISA flags, dependency lockfile, or benchmark
semantics changed. Treat such a run as a new baseline instead of computing a
percentage regression across unlike environments.

## Run on the pinned host outside GitHub Actions

The same runner can be invoked directly:

```text
BENCH_MACHINE_ID=my-pinned-benchmark-host \
EIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
sh scripts/run_release_benchmarks.sh
```

By default the script refuses a dirty Git checkout. `ALLOW_DIRTY_BENCHMARK=1`
exists only for exploratory work; results produced that way should not be used
as release evidence.

## Evidence hierarchy

A GitHub-hosted nightly result answers "did performance move enough to
investigate?" A pinned release benchmark answers "what did this release do on
this controlled machine?" Neither answers embedded timing questions. Physical
Cortex-M cycle measurements remain a separate target-qualification artifact as
described in [Target qualification](target-qualification.md).
