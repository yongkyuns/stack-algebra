# Sparse permutation repair measurements

This isolated, unpublished consumer measures the cost of #51 relative to #50.
It makes no library or package changes. Run with the workflow's pinned compiler:

```sh
rustup run 1.98.0 python3 tools/sparse-permutation-bench/compare.py \
  --candidate HEAD --rounds 12 --output permutation-evidence
```

The before revision is `f4fd7fdd8944c95a2951ef654c465271d1e4297c`.
Candidates must descend from the repaired `65c9d37d598984034abc18e31f0ffabb12798854`.
Both use identical input source patterns and matching destination patterns.
The old arbitrary-destination bug is never exercised for timing; candidate-only
correctness of that repair remains in `tests/sparse_permutation_contracts.rs`.
Same-length source-coordinate changes are not part of this benchmark or a newly
supported permutation contract.

## Cases and interpretation

There are 384 cases: banded dimensions 6/15/64/128, f32/f64, lower/full symmetric
source storage, compact/roomy capacity (5N/20N), identity/reverse/half-rotation
orderings, and four operations. Factor capacity is fixed at 8N for each size;
active factor entries are independently checked, including rotation-induced fill.
The capacities are budgets, not claims of exactly full storage.

`apply_into` measures the cached numeric permutation into a matching reusable
destination. `apply` measures the returning API, including its output construction.
`permute_cholesky` and `permute_ldlt` measure cached permutation followed by numeric
refactorization using retained symbolic metadata. They do not time the solve.
All setup, map construction, symbolic analysis, allocation and correctness checks
are outside timing. Coordinate references are constructed independently, and
factor solves are checked in original coordinates before and after timing.

Twelve rounds alternate before/candidate order; operation order reverses on odd
rounds. Both isolated builds share one lockfile, Rust 1.98.0, opt-level 3, one
codegen unit, no LTO, target-cpu=native and one selected CPU affinity. The driver
checks every expected case, round, positive measurement, capacity, active entry
count and symbolic fill count before generating a summary. Five Python tests
exercise malformed/missing measurement rejection. Rust fixture checks run in
debug and optimized builds before the two architecture evidence jobs start.

Artifacts retain 9,216 unique raw measurements per architecture, correctness-only
outputs for both revisions, the driver/harness/manifest, all source hashes,
compiler and CPU details, binary/lock hashes, build logs and regenerated summary.
Ratios are paired-round medians with observed ranges, not confidence intervals or
wall-clock gates. These are warm-cache synthetic hosted-runner workloads, not
real-MCU latencies, cold-cache results or an architecture ranking. Per-call and
pipeline numbers answer different questions and must not be combined as additive
component estimates. No performance conclusion is implied by a successful run.
