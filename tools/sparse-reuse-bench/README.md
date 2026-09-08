# Sparse reuse safety-check measurements

This repository-only consumer measures the cost of the LDLT and Cholesky
reuse repairs without changing the library or its published dependencies.
The comparison script builds the same harness against three exact revisions:
#44's merge (before both repairs), #45's merge (LDLT repaired), and a candidate
containing #48. It rejects candidates that do not descend from that safe base.
Historical code is exercised **only with matching source layouts**.

Run with Rust 1.98.0, Python 3.12+, Git history and Linux `lscpu` available:

```sh
python3 tools/sparse-reuse-bench/compare.py --candidate HEAD \
    --rounds 12 --output /tmp/sparse-reuse-results
```

The output directory must not exist. The dedicated workflow runs x86_64 and
native ARM64 comparisons independently; absolute timings across hosts are not
an architecture comparison. It runs on changes to this harness or manual
dispatch, not on every library change. Normal library CI remains untouched.

## Workloads and timing

There are 48 cases per layout: f32/f64, natural/ordered retained-pattern
recomputation, Cholesky/LDLT, banded dimensions 6/15/64/128, and star dimensions
64/128 with symbolic fill. Matrices are strictly diagonally dominant SPD.
Ordered calls use identity coordinates to isolate validation from permutation.
Factor capacity matches the required symbolic fill; numeric storage retains
spare capacity. All data construction, allocation, symbolic analysis and dense
residual checks are outside timing. Residuals and destination patterns are
checked before/after each measurement. The timed loop black-boxes borrowed
inputs, pattern and destination; no factor is copied out for observation.

Each sample uses a 10 ms warmup and a calibrated batch targeting 30 ms. The
script uses separate build directories, one dependency lockfile, opt-level 3,
one codegen unit, no LTO, and `target-cpu=native` for all revisions on a host.
Measurements pin to one available CPU. Twelve rounds cycle through all six
revision orders twice. Candidate-only full-storage cases alternate layout order.

`matching` retains the analyzed lower CSC layout. `repacked` uses that lower
schedule with equivalent full symmetric numeric storage, exercising LDLT's
checked fallback. `rebuilt` analyzes the full layout once before timing and
uses it with the same full numeric input. Repacked/rebuilt is therefore a
candidate-only comparison, **not** a before/after timing of formerly unsound
calls. It excludes the cost of rebuilding a symbolic pattern.

Raw CSV, build logs, lockfile, harness, compiler/CPU information, source SHAs,
binary hashes and execution order are retained. The report shows median ns
and paired ratios with observed min/max ranges, not confidence intervals.
Shared-runner noise, warm caches and synthetic structure limit generalization;
this is evidence to guide optimization, not a timing gate or a real-MCU result.
