#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

out_dir=${RELEASE_ARTIFACT_DIR:-release-artifacts}
nightly=${PUBLIC_API_NIGHTLY:-nightly-2026-08-20}
public_api_version=${PUBLIC_API_TOOL_VERSION:-0.51.0}

for command in cargo rustc rustup git sha256sum tar; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "$command is required" >&2
        exit 2
    }
done

if [ -n "$(git status --porcelain)" ]; then
    echo "release artifacts require a clean checkout" >&2
    exit 2
fi

checkout_commit=$(git rev-parse HEAD)
source_commit=${QUALIFICATION_SOURCE_SHA:-$checkout_commit}
if [ "$source_commit" != "$checkout_commit" ]; then
    echo "qualification source must match the exact checked-out commit" >&2
    exit 2
fi
ref=$(git symbolic-ref --short -q HEAD || git describe --always --exact-match 2>/dev/null || printf detached)

if ! rustc +"$nightly" --version >/dev/null 2>&1; then
    echo "Rust toolchain $nightly is required" >&2
    exit 2
fi
if ! cargo public-api --version 2>/dev/null | grep -Fq "cargo-public-api $public_api_version"; then
    echo "cargo-public-api $public_api_version is required" >&2
    exit 2
fi

# The library intentionally does not commit Cargo.lock. Resolve a fresh lockfile
# from the clean source tree so every release artifact captures the exact
# dependency graph used for that qualification run.
rm -f Cargo.lock
cargo generate-lockfile
lock_sha=$(sha256sum Cargo.lock | awk '{print $1}')

rm -rf "$out_dir" target/package
mkdir -p "$out_dir"

cargo +"$nightly" public-api -sss > "$out_dir/public-api.txt"
cargo +"$nightly" rustdoc --lib -- -Z unstable-options --output-format json
cp target/doc/stack_algebra.json "$out_dir/rustdoc-public-api.json"

cargo metadata --locked --format-version 1 > "$out_dir/cargo-metadata.json"
cargo tree --locked --edges normal,build > "$out_dir/dependency-tree.txt"
cargo package --locked --list > "$out_dir/package-files.txt"

# The published crate is a consumer artifact, not a repository snapshot.
# Fail qualification if repository-only infrastructure re-enters the package.
for forbidden in .github/ benches/ docs/ qemu-tests/ scripts/ tests/ tools/ .gitignore; do
    if grep -Fq "$forbidden" "$out_dir/package-files.txt"; then
        echo "package unexpectedly contains repository-only path: $forbidden" >&2
        exit 1
    fi
done

cargo package --locked
package=$(find target/package -maxdepth 1 -type f -name 'stack-algebra-*.crate' | sort | tail -n 1)
if [ -z "$package" ]; then
    echo "cargo package did not produce a stack-algebra crate archive" >&2
    exit 1
fi
cp "$package" "$out_dir/"

# Exercise the actual archive as a dependency of a separate consumer. Copying
# public contract tests into this consumer gives them no path to the repo crate.
consumer_root=$(mktemp -d)
trap 'rm -rf "$consumer_root"' EXIT HUP INT TERM
tar -xzf "$package" -C "$consumer_root"
package_source=$(find "$consumer_root" -mindepth 1 -maxdepth 1 -type d -name 'stack-algebra-*' | sort | head -n 1)
if [ -z "$package_source" ]; then
    echo "could not extract packaged stack-algebra source" >&2
    exit 1
fi
consumer_dir="$consumer_root/consumer"
mkdir -p "$consumer_dir/src" "$consumer_dir/tests"
cat > "$consumer_dir/Cargo.toml" <<EOF
[package]
name = "stack-algebra-package-smoke"
version = "0.0.0"
edition = "2021"
publish = false

[features]
default = []
std = ["stack-algebra/std"]

[dependencies]
stack-algebra = { path = "$package_source", default-features = false }
EOF
cat > "$consumer_dir/src/main.rs" <<'EOF'
use stack_algebra::{matrix, vector, Cholesky};

fn main() {
    let a = matrix![4.0_f64, 1.0; 1.0, 3.0];
    let b = vector![1.0_f64; 2.0];
    let factor = Cholesky::try_decompose(&a).expect("positive definite");
    let x = factor.solve(&b);
    assert!((a * x - b).norm() < 1.0e-12);
}
EOF
cp tests/matrix_swap_contracts.rs "$consumer_dir/tests/matrix_swap_contracts.rs"
cp tests/scalar_hook_contracts.rs "$consumer_dir/tests/scalar_hook_contracts.rs"
cp tests/matrix_storage.rs "$consumer_dir/tests/matrix_storage.rs"
cp tests/sparse_reuse_contracts.rs "$consumer_dir/tests/sparse_reuse_contracts.rs"
cargo generate-lockfile --manifest-path "$consumer_dir/Cargo.toml"

# Retain logs and fail immediately on any compile, link, runtime, or test error.
run_consumer_check() {
    log_name=$1
    shift
    if "$@" > "$out_dir/$log_name" 2>&1; then
        cat "$out_dir/$log_name"
    else
        cat "$out_dir/$log_name" >&2
        return 1
    fi
}

run_consumer_check package-consumer-default.log \
    cargo run --locked --manifest-path "$consumer_dir/Cargo.toml" --no-default-features
run_consumer_check package-consumer-std.log \
    cargo run --locked --manifest-path "$consumer_dir/Cargo.toml" --no-default-features --features std
run_consumer_check package-consumer-tests.log \
    cargo test --locked --manifest-path "$consumer_dir/Cargo.toml" --no-default-features --tests
run_consumer_check package-consumer-tests-std.log \
    cargo test --locked --manifest-path "$consumer_dir/Cargo.toml" --no-default-features --features std --tests
run_consumer_check package-consumer-tests-release.log \
    cargo test --release --locked --manifest-path "$consumer_dir/Cargo.toml" --no-default-features --tests

cp "$consumer_dir/Cargo.lock" "$out_dir/package-consumer-Cargo.lock"
cp "$consumer_dir/src/main.rs" "$out_dir/package-consumer-main.rs"
cp "$consumer_dir/tests/matrix_swap_contracts.rs" "$out_dir/package-consumer-matrix-swap-contracts.rs"
cp "$consumer_dir/tests/scalar_hook_contracts.rs" "$out_dir/package-consumer-scalar-hook-contracts.rs"
cp "$consumer_dir/tests/matrix_storage.rs" "$out_dir/package-consumer-matrix-storage.rs"
cp "$consumer_dir/tests/sparse_reuse_contracts.rs" "$out_dir/package-consumer-sparse-reuse-contracts.rs"
cargo metadata --locked --manifest-path "$consumer_dir/Cargo.toml" --format-version 1 > "$out_dir/package-consumer-metadata.json"
cargo tree --locked --manifest-path "$consumer_dir/Cargo.toml" --edges normal,build > "$out_dir/package-consumer-dependency-tree.txt"

{
    printf 'source_commit=%s\n' "$source_commit"
    printf 'checkout_commit=%s\n' "$checkout_commit"
    printf 'ref=%s\n' "$ref"
    printf 'cargo_lock_sha256=%s\n' "$lock_sha"
    printf 'rustc=%s\n' "$(rustc --version)"
    printf 'cargo=%s\n' "$(cargo --version)"
    printf 'public_api_nightly=%s\n' "$nightly"
    printf 'public_api_nightly_rustc=%s\n' "$(rustc +"$nightly" --version)"
    printf 'cargo_public_api=%s\n' "$(cargo public-api --version)"
    printf 'package_file=%s\n' "$(basename "$package")"
    printf 'package_sha256=%s\n' "$(sha256sum "$package" | awk '{print $1}')"
    printf 'package_surface=consumer-only-allowlist\n'
    printf 'package_consumer_smoke=passed\n'
    printf 'package_consumer_smoke_mode=executed-default-and-std\n'
    printf 'package_consumer_contracts=swap-and-scalar-hooks-passed-debug-default-debug-std-release-default\n'
    printf 'package_consumer_matrix_storage=passed-debug-default-debug-std-release-default\n'
    printf 'package_consumer_sparse_reuse=passed-debug-default-debug-std-release-default\n'
    printf 'package_consumer_lock_sha256=%s\n' "$(sha256sum "$out_dir/package-consumer-Cargo.lock" | awk '{print $1}')"
    printf 'public_api_sha256=%s\n' "$(sha256sum "$out_dir/public-api.txt" | awk '{print $1}')"
    printf 'rustdoc_json_sha256=%s\n' "$(sha256sum "$out_dir/rustdoc-public-api.json" | awk '{print $1}')"
} > "$out_dir/provenance.txt"

cp Cargo.lock "$out_dir/Cargo.lock"
printf '%s\n' "$source_commit" > "$out_dir/source-commit.txt"
printf 'Release artifacts written to %s\n' "$out_dir"
