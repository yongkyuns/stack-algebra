# Nightly benchmark comparison

The nightly workflow compares stack-algebra with Eigen, faer, and nalgebra
through a shared set of deterministic dense operations. It runs on a
GitHub-hosted Linux runner
with a native CPU target and publishes the raw measurements plus a
self-contained HTML report as the `nightly-benchmark-report` artifact. The
four Rust benchmark groups and the native Eigen runner execute as parallel
jobs; a final report job merges their measurements.

Each report also contains runner provenance: commit and ref, dirty-tree state,
runner OS/architecture, CPU model, kernel, Rust compiler, Cargo version, UTC
generation time, and measurement count. These fields are shown in the report
metadata line and written to `metadata.json` beside the HTML and CSV files.

Nightly uses short Criterion windows (0.1 seconds of warmup, 0.1 seconds of
measurement, and 10 samples) and five 5-millisecond Eigen samples so the full
comparison normally completes within a few minutes. Use longer windows and the
default 15 Eigen samples for release-quality measurements. CI caches the
Rust release artifacts and the source-keyed Eigen executable; the first run
after changing the benchmark sources still pays their compilation costs.

## Run locally

Install Eigen (`libeigen3-dev` on Ubuntu), then run the Criterion benches and
native Eigen runner:

```sh
EIGEN3_INCLUDE_DIR=/usr/include/eigen3 cargo bench \
  --bench comparison -- --warm-up-time 0.1 --measurement-time 0.1 --sample-size 10
for bench in dense_solvers sparse block_sparse; do
  EIGEN3_INCLUDE_DIR=/usr/include/eigen3 cargo bench --all-features \
    --bench "$bench" -- --warm-up-time 0.1 --measurement-time 0.1 --sample-size 10
done
mkdir -p benchmark-report/raw
CXXFLAGS='-march=native' EIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
  EIGEN_BENCH_SAMPLES=5 EIGEN_BENCH_MIN_SAMPLE_MS=5 \
  EIGEN_BENCH_CSV=benchmark-report/raw/eigen-f32.csv \
  ./eigen/run_native_bench.sh f32 > benchmark-report/raw/eigen-f32.txt
EIGEN_BENCH_SKIP_BUILD=1 CXXFLAGS='-march=native' EIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
  EIGEN_BENCH_SAMPLES=5 EIGEN_BENCH_MIN_SAMPLE_MS=5 \
  EIGEN_BENCH_CSV=benchmark-report/raw/eigen-f64.csv \
  ./eigen/run_native_bench.sh f64 > benchmark-report/raw/eigen-f64.txt
python3 scripts/generate_benchmark_report.py \
  --criterion-dir target/criterion \
  --eigen-csv benchmark-report/raw/eigen-f32.csv \
  --eigen-csv benchmark-report/raw/eigen-f64.csv \
  --output benchmark-report/index.html \
  --csv-output benchmark-report/results.csv \
  --require-eigen
```

For reproducible local reports, pass provenance as newline-delimited
`key=value` fields with `--metadata-file`; inline `--metadata` uses the same
keys separated by semicolons. The generator adds `generated_utc` and
`measurement_count` automatically.

The report generator also accepts one or more `--eigen-csv` files. CSV headers
may use `operation`/`benchmark`, `library`, `shape`, `scalar`, `phase`, and
`median_ns` (aliases such as `ns_per_op` and `time_ns` are accepted). Criterion
paths are discovered recursively, so nested groups such as
`sparse/llt/f64/.../stack-factor-reuse/15/new/estimates.json` do not require a
fixed directory layout.

## Interpreting results

- Times are median steady-state nanoseconds per operation. Lower is better;
  confidence intervals are retained in `results.csv` for Criterion rows.
- The dense comparison executes eight identical operations per Criterion
  iteration and normalizes the reported median back to one operation; the
  native Eigen runner similarly normalizes its 64-operation batches.
- Fixed-capacity stack-algebra matrices are compared with the corresponding
  fixed-size Eigen/nalgebra cases where available. faer and dynamic cases are
  labelled explicitly; a dynamic allocation path is not presented as an
  apples-to-apples static-storage result.
- Factorization, symbolic analysis, refactorization, and solve are separate
  phases. Reusable-factor solve timings exclude the one-time factorization;
  factor-and-solve timings include both. Do not compare these phases as if they
  were the same operation.
- The dense LDLT suite includes both factor-and-solve cases and reusable
  two-right-hand-side solve cases; the latter keeps factorization outside the
  timed region.
- Sparse LDLT includes a separate auto-pivot group for zero-leading-diagonal
  inputs, so sparse diagonal pivoting is not conflated with the no-pivot path;
  its factor and reusable-refactorization phases are measured separately.
- Inputs, dimensions, scalar type, and benchmark setup are owned by the bench
  sources. Correctness checks should run before timing and failed checks must
  fail the benchmark rather than produce a result.
- Results are machine-specific. The report records the commit and runner, but
  comparisons across different CPU models should be treated as directional.

## Nightly report artifacts

The nightly workflow always uploads a `nightly-benchmark-report` artifact. Open
the [nightly benchmark workflow](https://github.com/yongkyuns/stack-algebra/actions/workflows/nightly-bench.yml),
select a completed run, and download the artifact to view `index.html` and the
raw CSV/JSON inputs. GitHub Pages is reserved for the documentation site, so a
benchmark run cannot overwrite the published documentation.
