# Performance

Benchmarks are included to give users a practical sense of the runtime cost of
common `stack-algebra` operations. They are point-in-time measurements on a
particular machine and should be treated as orientation, not as a guarantee for
a specific target or workload.

## Representative measurements

The table below is a representative native-host snapshot from **August 8,
2026**. Values are median nanoseconds per operation from the repository's fast
benchmark sweep.

### 32x32 dense operations

| Operation | Scalar | Median |
| --- | --- | ---: |
| Matrix multiply | `f32` | 1,021 ns |
| Matrix multiply | `f64` | 1,637 ns |
| Matrix-vector multiply | `f32` | 33 ns |
| Matrix-vector multiply | `f64` | 68 ns |
| Cholesky factor | `f32` | 3,594 ns |
| Cholesky factor | `f64` | 3,294 ns |
| LDLT factor | `f32` | 4,793 ns |
| LDLT factor | `f64` | 5,244 ns |
| LU factor | `f32` | 3,671 ns |
| LU factor | `f64` | 5,689 ns |
| Householder QR factor | `f32` | 12,643 ns |
| Householder QR factor | `f64` | 9,499 ns |

### Tall SVD examples

| Shape | Scalar | Median |
| --- | --- | ---: |
| 6x3 | `f32` | 740 ns |
| 6x3 | `f64` | 984 ns |
| 15x6 | `f32` | 4,400 ns |
| 15x6 | `f64` | 5,400 ns |

These numbers are useful for estimating the approximate scale of a workload and
for spotting regressions between revisions. For a real application, measure the
exact operation, scalar type, matrix shape, compiler settings, and memory
placement that matter to that application.

## Why the numbers vary

Microbenchmark results are sensitive to context, including:

- CPU model and frequency behavior;
- compiler version and target features;
- `f32` vs `f64`;
- matrix dimensions and shape;
- dense, sparse, or block-sparse storage;
- whether allocation or setup is inside the timed region;
- whether a factor is created once or reused;
- benchmark duration and concurrent system load.

For that reason, benchmark results should be read as representative measurements
rather than universal constants.

## Matching the operation matters

The benchmark suite separates phases that answer different application
questions.

For dense solvers, **factor-and-solve** includes factorization while a
**reusable-factor solve** measures only the solve using an already-built factor.
If an application reuses a factor across many right-hand sides, the latter is
the more relevant number.

For sparse systems, symbolic analysis, numeric assembly, factorization,
refactorization, permutation/setup, and solve are measured separately where
applicable. Keeping those phases distinct makes the measurements easier to map
to a real application loop.

## Nightly measurements

The repository runs a nightly benchmark workflow covering selected dense,
sparse, and block-sparse operations. The workflow publishes a
`nightly-benchmark-report` artifact containing:

- a self-contained HTML report;
- CSV measurements;
- raw benchmark inputs;
- commit, runner, CPU, compiler, and generation metadata.

Open the [nightly benchmark workflow](https://github.com/yongkyuns/stack-algebra/actions/workflows/nightly-bench.yml),
choose a completed run, and download the `nightly-benchmark-report` artifact for
the freshest measurements.

Nightly uses relatively short measurement windows so it can cover a broad set
of cases regularly. For a release or architecture decision, run the relevant
case for longer on the hardware that actually matters to the application.

## Running focused benchmarks locally

For fixed-size dense operations:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --bench fixed_size
```

For dense solver workloads:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --bench dense_solvers
```

Sparse and block-sparse benchmark targets are also available under `benches/`.

For embedded work, host microbenchmarks are only a first check. The most useful
measurement is the complete operation on the target MCU or processor with the
same scalar type, dimensions, compiler flags, and memory placement used by the
application.
