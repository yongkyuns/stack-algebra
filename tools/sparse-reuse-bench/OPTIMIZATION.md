# Profiling and comparing sparse reuse optimizations

Run from a checkout that descends from the qualified safe baseline:

```sh
rustup run 1.98.0 python3 tools/sparse-reuse-bench/optimize.py \
  --candidate HEAD --rounds 12 --output optimization-evidence
```

The baseline is fixed at `3c97ceb89ead00eed0c754e9f6bc8be38ccba326` (#49).
Unlike `compare.py`, this comparison never runs the historical pre-repair
implementation: both baseline and candidate support the changed-layout cases.
The existing public benchmark fixture source is copied unchanged into both
isolated builds. The script verifies candidate ancestry, uses a common lockfile
and separate build directories, pins CPU affinity where supported, and alternates
baseline/candidate execution order for twelve paired rounds.

## Component profiling

An additional temporary copy of the baseline exposes its private source-map
validator, factor-coverage validator, direct pattern assignment, and assignment
guarded by full pattern equality. The wrappers are appended only in that
profiling copy. They do not enter the repository library or either whole-call
comparison binary. Six component rounds reverse component order on alternate
rounds. Standalone component microbenchmarks have different inlining and code
layout from a full recomputation, so their times must not be added together or
subtracted from whole-call measurements as an exact decomposition.

## Retained evidence

- `samples.csv`: 3,456 whole-call measurements (144 cases, two revisions, twelve rounds).
- `components.csv`: 288 baseline component measurements (48 cases, six rounds).
- `summary.md`: paired median candidate/baseline ratios and observed ranges.
- `provenance.json`, `cpu.txt`, `Cargo.lock`, build logs, original harness and
  manifest, the driver, and the temporary profiling harness/shim.

The 144 whole-call cases retain all #49 fixtures: f32/f64, Cholesky/LDLT,
natural/identity-ordered retained-pattern recomputation, banded dimensions
6/15/64/128, fill-producing stars 64/128, and matching/repacked/rebuilt storage.
Numerical residual and output-pattern assertions remain outside timing.
Symbolic preparation is excluded even from repacked/rebuilt comparisons.

The dedicated workflow uses Rust 1.98.0, opt-level 3, one codegen unit, no LTO,
and target-cpu=native on x86-64 and native ARM64 runners. It retains evidence
without enforcing noisy wall-clock thresholds. Manual runs must use the same
compiler and environment for comparisons; inspect the recorded provenance.

These are warm-cache synthetic hosted-runner measurements, not cold-cache,
real-MCU latency, or an architecture ranking. Observed ranges are descriptive,
not confidence intervals. The profile-only initial revision provides an
unchanged-source control; small apparent gains within that control's variation
must not be interpreted as an optimization.

## Current optimization boundary

The fill-free lower-layout fast path checks actual column starts and active row
indices, not a hash or pointer identity. LDLT additionally verifies immutable
symbolic metadata proving that this factor layout is the original source
layout; equality with the factor alone would be insufficient. Fill-producing,
missing-diagonal and full-storage schedules keep the original source checks.
The full factor-coverage validator remains the fallback. Arithmetic kernels,
destination copying, representation, public API and numerical tolerances are
unchanged. The invariant and exhaustive coordinate-reference tests are described
in `UNSAFE.md`.
