#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

warmup=${BENCH_WARMUP_TIME:-1}
measurement=${BENCH_MEASUREMENT_TIME:-3}
sample_size=${BENCH_SAMPLE_SIZE:-30}
eigen_samples=${EIGEN_BENCH_SAMPLES:-15}
eigen_min_sample_ms=${EIGEN_BENCH_MIN_SAMPLE_MS:-50}
machine_id=${BENCH_MACHINE_ID:-}
report_dir=${BENCH_REPORT_DIR:-benchmark-report/release}
raw_dir="$report_dir/raw"
criterion_dir=target/criterion

if [ -z "$machine_id" ]; then
    echo "BENCH_MACHINE_ID is required for release benchmark provenance" >&2
    exit 2
fi

for command in cargo rustc git python3 sha256sum uname; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "$command is required" >&2
        exit 2
    }
done

if [ -z "${EIGEN3_INCLUDE_DIR:-}" ] || [ ! -d "$EIGEN3_INCLUDE_DIR" ]; then
    echo "EIGEN3_INCLUDE_DIR must point to an installed Eigen include directory" >&2
    exit 2
fi

if [ "${ALLOW_DIRTY_BENCHMARK:-0}" != 1 ] && [ -n "$(git status --porcelain)" ]; then
    echo "release benchmarks require a clean checkout (set ALLOW_DIRTY_BENCHMARK=1 only for exploratory runs)" >&2
    exit 2
fi

case "$sample_size" in
    ''|*[!0-9]*) echo "BENCH_SAMPLE_SIZE must be an integer" >&2; exit 2 ;;
esac
if [ "$sample_size" -lt 10 ]; then
    echo "BENCH_SAMPLE_SIZE must be at least 10" >&2
    exit 2
fi

rm -rf "$criterion_dir" "$report_dir"
mkdir -p "$raw_dir"

rustflags=${RUSTFLAGS:--C target-cpu=native}
cxxflags=${CXXFLAGS:--march=native}
export RUSTFLAGS="$rustflags"
export CXXFLAGS="$cxxflags"

run_criterion() {
    bench=$1
    shift
    cargo bench "$@" --bench "$bench" -- \
        --warm-up-time "$warmup" \
        --measurement-time "$measurement" \
        --sample-size "$sample_size" \
        --noplot
}

run_criterion comparison
for bench in fixed_size small_fixed dense_solvers sparse block_sparse fused; do
    run_criterion "$bench" --all-features
done

EIGEN_BENCH_SAMPLES="$eigen_samples" \
EIGEN_BENCH_MIN_SAMPLE_MS="$eigen_min_sample_ms" \
EIGEN_BENCH_CSV="$raw_dir/eigen-f32.csv" \
./eigen/run_native_bench.sh f32 > "$raw_dir/eigen-f32.txt"

EIGEN_BENCH_SKIP_BUILD=1 \
EIGEN_BENCH_SAMPLES="$eigen_samples" \
EIGEN_BENCH_MIN_SAMPLE_MS="$eigen_min_sample_ms" \
EIGEN_BENCH_CSV="$raw_dir/eigen-f64.csv" \
./eigen/run_native_bench.sh f64 > "$raw_dir/eigen-f64.txt"

commit=$(git rev-parse HEAD)
ref=$(git symbolic-ref --short -q HEAD || git describe --always --exact-match 2>/dev/null || printf detached)
cpu=$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' || true)
lock_sha=$(sha256sum Cargo.lock | awk '{print $1}')

{
    printf 'evidence_level=release-candidate\n'
    printf 'machine_id=%s\n' "$machine_id"
    printf 'commit=%s\n' "$commit"
    printf 'ref=%s\n' "$ref"
    printf 'git_dirty=%s\n' "$(if test -n "$(git status --porcelain)"; then echo true; else echo false; fi)"
    printf 'runner_name=%s\n' "${RUNNER_NAME:-local}"
    printf 'runner_os=%s\n' "${RUNNER_OS:-$(uname -s)}"
    printf 'runner_arch=%s\n' "${RUNNER_ARCH:-$(uname -m)}"
    printf 'cpu=%s\n' "${cpu:-unknown}"
    printf 'kernel=%s\n' "$(uname -sr)"
    printf 'rustc=%s\n' "$(rustc --version)"
    printf 'cargo=%s\n' "$(cargo --version)"
    printf 'cargo_lock_sha256=%s\n' "$lock_sha"
    printf 'eigen_include=%s\n' "$EIGEN3_INCLUDE_DIR"
    printf 'rustflags=%s\n' "$RUSTFLAGS"
    printf 'cxxflags=%s\n' "$CXXFLAGS"
    printf 'criterion_warmup_s=%s\n' "$warmup"
    printf 'criterion_measurement_s=%s\n' "$measurement"
    printf 'criterion_sample_size=%s\n' "$sample_size"
    printf 'eigen_samples=%s\n' "$eigen_samples"
    printf 'eigen_min_sample_ms=%s\n' "$eigen_min_sample_ms"
} > "$report_dir/provenance.txt"

python3 scripts/generate_benchmark_report.py \
    --criterion-dir "$criterion_dir" \
    --eigen-csv "$raw_dir/eigen-f32.csv" \
    --eigen-csv "$raw_dir/eigen-f64.csv" \
    --output "$report_dir/index.html" \
    --csv-output "$report_dir/results.csv" \
    --metadata-file "$report_dir/provenance.txt" \
    --require-eigen

cp Cargo.lock "$report_dir/Cargo.lock"
printf '%s\n' "$commit" > "$report_dir/commit.txt"
printf 'Release benchmark report: %s\n' "$report_dir"
