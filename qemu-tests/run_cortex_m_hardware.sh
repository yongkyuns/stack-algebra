#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
manifest="$repo_root/qemu-tests/Cargo.toml"
target=${CORTEX_M_TARGET:-thumbv7em-none-eabihf}
target_dir="$repo_root/qemu-tests/target/hardware"
memory_x=${MEMORY_X:-}

if [ -z "$memory_x" ]; then
    echo "MEMORY_X must point to the board-specific memory.x file" >&2
    exit 2
fi
if [ ! -f "$memory_x" ]; then
    echo "MEMORY_X does not exist: $memory_x" >&2
    exit 2
fi
if ! rustup target list --installed | grep -qx "$target"; then
    echo "Rust target $target is required" >&2
    exit 2
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM
cp "$memory_x" "$tmpdir/memory.x"

RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Tlink.x -C link-arg=-L$tmpdir" cargo build \
    --manifest-path "$manifest" \
    --target-dir "$target_dir" \
    --target "$target" \
    --release \
    --features cortex-m \
    --bin cortex-m-bench

binary="$target_dir/$target/release/cortex-m-bench"
printf 'Cortex-M benchmark ELF: %s\n' "$binary"
printf 'Target: %s\n' "$target"
printf 'The benchmark reports DWT CYCCNT minima over 32 iterations.\n'

if [ -n "${RUNNER:-}" ]; then
    printf 'Running with: %s <elf>\n' "$RUNNER"
    sh -c "$RUNNER \"$binary\""
else
    printf 'RUNNER is unset; flash/run the ELF with a debugger that supports semihosting.\n'
fi
