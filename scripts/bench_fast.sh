#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

warmup_time="${BENCH_WARMUP_TIME:-0.01}"
measurement_time="${BENCH_MEASUREMENT_TIME:-0.02}"
sample_size="${BENCH_SAMPLE_SIZE:-10}"
log_dir="${BENCH_LOG_DIR:-benchmark-report/raw/fast}"

if (( sample_size < 10 )); then
    echo "BENCH_SAMPLE_SIZE must be at least 10 (Criterion's minimum)" >&2
    exit 2
fi

# Match the native Rust target's instruction set in the Eigen reference unless
# the caller has selected a specific C++ target explicitly.
if [[ -z "${CXXFLAGS:-}" ]]; then
    export CXXFLAGS="-march=native"
fi
if [[ -z "${RUSTFLAGS:-}" ]]; then
    export RUSTFLAGS="-C target-cpu=native"
fi

if [[ -n "${EIGEN3_INCLUDE_DIR:-}" ]]; then
    if [[ ! -f "${EIGEN3_INCLUDE_DIR}/Eigen/Core" ]]; then
        echo "Eigen headers not found under EIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}" >&2
        exit 1
    fi
elif ! command -v pkg-config >/dev/null || ! pkg-config --exists eigen3; then
    echo "Eigen headers not found; set EIGEN3_INCLUDE_DIR or install eigen3 pkg-config metadata" >&2
    exit 1
fi

mkdir -p "$log_dir"

benchmarks=(comparison dense_solvers sparse block_sparse fixed_size small_fixed)

started_at=$SECONDS

# Build all targets once before launching the independent benchmark processes.
# This prevents Cargo's target-directory lock from serializing the sweep.
cargo bench --all-features --benches --no-run >"$log_dir/build.log" 2>&1

pids=()
for bench in "${benchmarks[@]}"; do
    log_file="$log_dir/${bench}.log"
    cargo bench --all-features --bench "$bench" -- \
        --warm-up-time "$warmup_time" \
        --measurement-time "$measurement_time" \
        --sample-size "$sample_size" \
        --noplot >"$log_file" 2>&1 &
    pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "benchmark failed: ${benchmarks[$index]}" >&2
        failed=1
    fi
done
if (( failed != 0 )); then
    exit 1
fi

mkdir -p benchmark-report/raw
eigen_dir="${EIGEN3_INCLUDE_DIR:-}"
if [[ -n "$eigen_dir" ]]; then
    export EIGEN3_INCLUDE_DIR="$eigen_dir"
fi
EIGEN_BENCH_SAMPLES="${EIGEN_BENCH_SAMPLES:-3}" \
EIGEN_BENCH_MIN_SAMPLE_MS="${EIGEN_BENCH_MIN_SAMPLE_MS:-2}" \
EIGEN_BENCH_CSV=benchmark-report/raw/eigen-f32-fast.csv \
./eigen/run_native_bench.sh f32 >benchmark-report/raw/eigen-f32-fast.txt
EIGEN_BENCH_SAMPLES="${EIGEN_BENCH_SAMPLES:-3}" \
EIGEN_BENCH_MIN_SAMPLE_MS="${EIGEN_BENCH_MIN_SAMPLE_MS:-2}" \
EIGEN_BENCH_SKIP_BUILD=1 \
EIGEN_BENCH_CSV=benchmark-report/raw/eigen-f64-fast.csv \
./eigen/run_native_bench.sh f64 >benchmark-report/raw/eigen-f64-fast.txt

python3 scripts/generate_benchmark_report.py \
    --criterion-dir target/criterion \
    --eigen-csv benchmark-report/raw/eigen-f32-fast.csv \
    --eigen-csv benchmark-report/raw/eigen-f64-fast.csv \
    --output benchmark-report/index.html \
    --csv-output benchmark-report/results.csv \
    --require-eigen \
    --metadata "profile=fast; warmup=${warmup_time}; measurement=${measurement_time}; samples=${sample_size}"

elapsed=$((SECONDS - started_at))
echo "fast benchmark sweep complete in ${elapsed}s: $log_dir"
