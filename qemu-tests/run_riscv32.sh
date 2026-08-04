#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
manifest="$repo_root/qemu-tests/Cargo.toml"
target="riscv32imc-unknown-none-elf"
binary="$repo_root/qemu-tests/target/$target/release/riscv32"

command -v cargo >/dev/null 2>&1 || {
  echo "cargo is required" >&2
  exit 1
}
command -v qemu-system-riscv32 >/dev/null 2>&1 || {
  echo "qemu-system-riscv32 is required" >&2
  exit 1
}

rustup target list --installed | grep -qx "$target" || {
  echo "Rust target $target is not installed; run: rustup target add $target" >&2
  exit 1
}

RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-T$repo_root/qemu-tests/riscv-memory.x -C link-arg=-Tlink.x" \
  cargo build \
    --manifest-path "$manifest" \
    --target-dir "$repo_root/qemu-tests/target" \
    --target "$target" \
    --release \
    --features riscv32 \
    --bin riscv32

output=$(qemu-system-riscv32 \
  -M virt \
  -nographic \
  -bios none \
  -kernel "$binary")

printf '%s\n' "$output" | sh "$repo_root/qemu-tests/check_stack_usage.sh" riscv32
