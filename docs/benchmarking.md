# Performance

Benchmarks are included to give a rough sense of the runtime cost of common
`stack-algebra` operations. They are most useful for orientation and regression
tracking. For decisions that matter to a product or algorithm, measure the
actual workload on the actual target.

The charts below show representative native-host results from **August 8,
2026**. Values are median nanoseconds per operation. Lower is better.

## 32×32 dense operations

### `f32`

![32×32 dense f32 benchmark comparison](assets/benchmark-dense-f32.svg)

### `f64`

![32×32 dense f64 benchmark comparison](assets/benchmark-dense-f64.svg)

## Tall SVD examples

![Tall SVD benchmark comparison](assets/benchmark-svd-tall.svg)

## How to read the results

- Treat these as **representative measurements**, not guarantees.
- Matrix shape, scalar type, compiler flags, target features, memory placement,
  and machine load can materially change timings.
- The logarithmic horizontal axis is used so both very small and much larger
  operations remain visible on the same chart.
- Missing bars indicate that a result was not included in this snapshot.

## Matching the measured phase matters

Benchmark interpretation depends on what is being timed.

For dense solvers, **factor-and-solve** includes factorization, while a
**reusable-factor solve** measures only the solve using an already-built factor.
Those answer different application questions.

For sparse systems, symbolic analysis, numeric assembly, factorization,
refactorization, permutation/setup, and solve are measured separately where
applicable. Keeping those phases distinct makes it easier to map the benchmark
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
fresh measurements.

Nightly uses relatively short measurement windows so it can cover a broad set
of cases regularly. For a release or architecture decision, run the relevant
case for longer on the hardware that actually matters to the application.

## Running focused benchmarks locally

For fixed-size dense operations:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --bench comparison
```

For embedded work, host microbenchmarks are only a first check. The most useful
measurement is the complete operation on the target MCU or processor with the
same scalar type, dimensions, compiler flags, and memory placement used by the
application.
