#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$ROOT_DIR/docs"
BUILD_DIR="$ROOT_DIR/build/site"
CARGO_TARGET_DIR="$ROOT_DIR/build/cargo"

if ! command -v mdbook >/dev/null 2>&1; then
  echo "mdbook is required; install a pinned version before building docs" >&2
  exit 1
fi

rm -rf "$BUILD_DIR" "$CARGO_TARGET_DIR"

python3 "$ROOT_DIR/scripts/generate_docs_performance_charts.py"
cargo doc --no-deps --target-dir "$CARGO_TARGET_DIR"
mdbook build "$DOCS_DIR" --dest-dir "$BUILD_DIR"

mkdir -p "$BUILD_DIR/api/stack_algebra"
cp -R "$CARGO_TARGET_DIR/doc/." "$BUILD_DIR/api/"

echo "Documentation site written to $BUILD_DIR"
