#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
consumer=$(mktemp -d)
trap 'rm -rf "$consumer"' EXIT HUP INT TERM
mkdir -p "$consumer/src"
cat > "$consumer/Cargo.toml" <<EOF
[package]
name = "stack-algebra-dependency-boundary"
version = "0.0.0"
edition = "2021"
publish = false

[features]
default = []
std = ["stack-algebra/std"]

[dependencies]
stack-algebra = { path = "$repo_root", default-features = false }
EOF
cat > "$consumer/src/lib.rs" <<'EOF'
#![no_std]

use stack_algebra::Matrix;

pub fn identity() -> Matrix<2, 2, f32> {
    Matrix::eye()
}
EOF

check_configuration() {
    name=$1
    shift
    # An external consumer excludes the repository's dev-dependencies, which
    # legitimately use native build tooling for comparison libraries.
    cargo tree --manifest-path "$consumer/Cargo.toml" --no-default-features \
        --edges normal,build --prefix none --format '{p}' "$@" > "$consumer/$name.txt"
    cat "$consumer/$name.txt"
    if grep -Eq '^cc v[0-9]' "$consumer/$name.txt"; then
        echo "ordinary $name consumers must not activate the cc build helper" >&2
        exit 1
    fi
    # No C++ compiler or Eigen installation should be consulted in either mode.
    CXX="$consumer/no-cxx-compiler" EIGEN3_INCLUDE_DIR="$consumer/no-eigen-headers" \
        CARGO_TARGET_DIR="$consumer/target" \
        cargo build --locked --lib --manifest-path "$consumer/Cargo.toml" \
        --no-default-features "$@"
}

check_configuration default
check_configuration std --features std
