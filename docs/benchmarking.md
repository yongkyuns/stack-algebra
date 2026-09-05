# Performance

Performance measurements are included to help users understand the practical
cost of choosing `stack-algebra` for fixed-size workloads. They are **not a
leaderboard and not a claim that this project is intended to compete with or
replace Eigen, faer, or nalgebra**.

Those libraries are useful references because they are familiar, mature, and
represent different design points. The interesting question is whether
`stack-algebra` is in a reasonable performance range for the workloads it
serves while preserving its fixed/bounded storage and `no_std` model.

## What the comparisons are for

Use the benchmark results to answer questions such as:

- Is a fixed-size operation in roughly the expected performance class?
- Is a particular factorization unexpectedly expensive for my problem shape?
- What does factor reuse save compared with factor-and-solve?
- Does an optimized host kernel materially change the result?
- Is a performance regression visible from one revision to the next?

Do **not** use a single ratio to conclude that one library is generally
"faster." Different libraries make different choices about storage, allocation,
runtime sizing, vectorization, sparse ordering, and supported workloads.

## Representative snapshot

The table below is a representative native-host snapshot from **August 8,
2026**. Values are median nanoseconds per operation from the repository's fast
benchmark sweep and are included only to show the approximate scale of the
implementation at that point in time.

### 32x32 dense operations

| Operation | Scalar | stack-algebra | Eigen | faer |
| --- | --- | ---: | ---: | ---: |
| Matrix multiply | `f32` | 1,021 ns | 1,378 ns | 1,088 ns |
| Matrix multiply | `f64` | 1,637 ns | 2,493 ns | 1,827 ns |
| Matrix-vector multiply | `f32` | 33 ns | 52 ns | 174 ns |
| Matrix-vector multiply | `f64` | 68 ns | 61 ns | 163 ns |
| Cholesky factor | `f32` | 3,594 ns | 3,991 ns | 3,367 ns |
| Cholesky factor | `f64` | 3,294 ns | 3,847 ns | 2,869 ns |
| LDLT factor | `f32` | 4,793 ns | 4,953 ns | 2,783 ns |
| LDLT factor | `f64` | 5,244 ns | 6,245 ns | 3,519 ns |
| LU factor | `f32` | 3,671 ns | 4,757 ns | 7,576 ns |
| LU factor | `f64` | 5,689 ns | 7,693 ns | 6,960 ns |
| Householder QR factor | `f32` | 12,643 ns | 11,422 ns | — |
| Householder QR factor | `f64` | 9,499 ns | 12,938 ns | — |

### Tall SVD examples

| Shape | Scalar | stack-algebra | Eigen |
| --- | --- | ---: | ---: |
| 6x3 | `f32` | 740 ns | 930 ns |
| 6x3 | `f64` | 984 ns | 1,180 ns |
| 15x6 | `f32` | 4,400 ns | 4,449 ns |
| 15x6 | `f64` | 5,400 ns | 6,861 ns |

The useful takeaway is not which cell wins. For these selected fixed-size host
workloads, `stack-algebra` was broadly in the same order of magnitude as mature
native references, with individual operations landing on either side depending
on scalar type and algorithm. That is the level of interpretation this snapshot
is meant to support.

## Why the numbers are not universal

Microbenchmark results are highly sensitive to context:

- CPU model and frequency behavior;
- compiler and target features;
- `f32` vs `f64`;
- matrix dimensions and shape;
- whether storage is fixed, dynamic, sparse, or block sparse;
- whether allocation is inside the timed region;
- sparse ordering and symbolic-analysis policy;
- whether a factor is created once or reused;
- benchmark duration and concurrent load on the machine.

The nightly report records runner and toolchain provenance for this reason.
Comparisons across different machines or benchmark phases should be treated as
directional rather than exact.

## Matching the operation matters

The benchmark suite separates phases that have different meanings.

For dense solvers, **factor-and-solve** includes factorization while
**reusable-factor solve** measures a solve using an already-built factor. Those
numbers answer different application questions and should not be compared as if
they were the same operation.

For sparse systems, the suite distinguishes symbolic analysis, numeric assembly,
factorization, refactorization, permutation/setup, and solve. It also labels
cases where another library uses a dynamic or allocating path so that a user can
see when the storage model is not identical.

This distinction is more important than a small percentage difference between
two timings.

## Nightly measurements

The repository runs a nightly benchmark workflow that measures selected dense,
sparse, and block-sparse operations against Eigen, faer, and nalgebra where a
meaningful comparable case exists.

The workflow publishes a `nightly-benchmark-report` artifact containing:

- a self-contained HTML report;
- CSV measurements;
- raw benchmark inputs;
- commit, runner, CPU, compiler, and generation metadata.

Open the [nightly benchmark workflow](https://github.com/yongkyuns/stack-algebra/actions/workflows/nightly-bench.yml),
choose a completed run, and download the `nightly-benchmark-report` artifact for
the freshest measurements.

Nightly uses relatively short measurement windows so it can cover a broad set
of cases regularly. For an architecture or release decision, run the relevant
case for longer on the hardware that actually matters to your application.

## Running a focused benchmark locally

For the Rust comparison benchmark:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --bench comparison
```

Some comparison cases require Eigen headers and optional benchmark features.
The repository's nightly workflow is the canonical reference for the complete
setup and report generation.

For embedded work, host microbenchmarks are only a first check. The most useful
measurement is the actual operation, scalar type, matrix shape, compiler flags,
and memory placement on the target MCU or processor.

## How performance fits the library's goals

`stack-algebra` prioritizes a combination of:

- predictable fixed or bounded storage;
- `no_std` usability;
- explicit ownership and reuse;
- numerically appropriate solver choices;
- performance that is reasonable for the intended matrix sizes.

A performance improvement is valuable when it preserves those properties and
helps a real workload. Matching every specialized library or backend on every
shape is not a project requirement.