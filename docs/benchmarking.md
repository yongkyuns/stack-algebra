# Nightly benchmark comparison

The nightly workflow compares stack-algebra with Eigen, faer, and nalgebra
through a shared set of deterministic dense operations. It runs on a
GitHub-hosted Linux runner
with a native CPU target and publishes the raw measurements plus a
self-contained HTML report as the `nightly-benchmark-report` artifact.

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
EIGEN3_INCLUDE_DIR=/usr/include/eigen3 cargo bench --locked \
  --bench comparison -- --warm-up-time 0.1 --measurement-time 0.1 --sample-size 10
for bench in sparse block_sparse; do
  EIGEN3_INCLUDE_DIR=/usr/include/eigen3 cargo bench --locked --all-features \
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
- Inputs, dimensions, scalar type, and benchmark setup are owned by the bench
  sources. Correctness checks should run before timing and failed checks must
  fail the benchmark rather than produce a result.
- Results are machine-specific. The report records the commit and runner, but
  comparisons across different CPU models should be treated as directional.

## Optional Pages publishing

The workflow runs nightly and always uploads an artifact. Publishing to Pages
is opt-in so forks and repositories without Pages enabled do not fail: set the
repository variable `PUBLISH_BENCHMARKS=true`, enable **GitHub Actions** as the
Pages source, and grant the `github-pages` environment's normal deployment
permissions. Scheduled runs then deploy `benchmark-report/index.html`; manual
dispatches still produce an artifact but do not overwrite the published page.
