# Benchmarking

The benchmark suite compares `stack-algebra` with Eigen, faer, and nalgebra where storage/operation semantics are comparable. Benchmark evidence is deliberately split into two tiers: short hosted regression triage and pinned-machine release qualification.

## Nightly regression triage

The nightly workflow runs five Rust benchmark groups — `comparison`, `dense_solvers`, `sparse`, `block_sparse`, and `fused` — plus the native Eigen reference runner. Jobs run on GitHub-hosted Linux with native CPU flags and are merged into a self-contained `nightly-benchmark-report` artifact.

Nightly uses short measurement windows so broad coverage completes quickly. It records commit/ref, runner OS/architecture, CPU model, kernel, Rust/Cargo, dirty-tree state, generation time, and measurement count. These results answer **"did performance move enough to investigate?"**, not **"what is the canonical performance of this release?"**

The standalone `scripts/bench_fast.sh` additionally sweeps the six older broad benchmark targets (`comparison`, `dense_solvers`, `sparse`, `block_sparse`, `fixed_size`, and `small_fixed`) for local triage; the focused `fused` suite is exercised directly by the nightly workflow.

## Run locally

Install Eigen and run an individual Criterion group with native CPU flags, for example:

```sh
RUSTFLAGS="-C target-cpu=native" \
EIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
cargo bench --all-features --bench dense_solvers -- \
  --warm-up-time 0.1 --measurement-time 0.1 --sample-size 10 --noplot
```

For the broad fast sweep:

```sh
EIGEN3_INCLUDE_DIR=/usr/include/eigen3 scripts/bench_fast.sh
```

Use these modes for regression investigation and architecture exploration. For release-quality host comparisons, use [Release benchmark qualification](release-benchmarking.md).

## Comparison rules

- Report median steady-state time per operation and retain Criterion confidence information in the raw/report artifacts.
- Match shape, scalar type, storage model, ordering policy, compiler/ISA flags, setup semantics, RHS count, and allocation behavior before interpreting a ratio.
- Run correctness checks before timing; a failed check must fail the benchmark rather than emit a performance result.
- Separate one-time symbolic analysis, numeric assembly, factorization/refactorization, and solve phases.
- Do not compare a reusable-factor solve against factor-and-solve as if they were the same operation.
- Label dynamic allocation paths explicitly rather than presenting them as apples-to-apples fixed-storage results.
- Treat Eigen/faer/nalgebra as references, not as targets that define the public API.

Sparse cases intentionally separate symbolic analysis, numeric assembly into validated patterns, factorization, refactorization, ordering/permutation, and solve. Ordered cases must use comparable ordering policies. The stack-algebra sparse APIs expose fixed-capacity semantics that can differ materially from dynamic sparse libraries, so setup/storage differences belong in the interpretation rather than being hidden.

## Existing native baseline

[Native benchmark baseline](benchmark-baseline.md) records a historical development baseline and follow-up kernel measurements. It is useful for identifying regressions and optimization opportunities, but it predates the pinned-machine release-evidence contract and should not be promoted to a canonical `0.3` release result.

## Release evidence

A release benchmark must run on the deliberately pinned host and preserve the exact source/toolchain/dependency/ISA/machine provenance plus raw measurements. If that controlled run is unavailable, release documentation should omit canonical cross-library performance claims rather than promote variable GitHub-hosted numbers.

Host benchmark evidence does not establish embedded performance. Real-device timing requires named physical hardware; see [Target qualification](target-qualification.md).

## Artifacts

The nightly workflow uploads `nightly-benchmark-report`. The release workflow uploads a separate release report with raw Criterion/Eigen data and provenance. GitHub Pages is reserved for the combined documentation/API site rather than transient benchmark output.
