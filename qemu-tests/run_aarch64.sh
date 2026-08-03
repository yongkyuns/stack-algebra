#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
manifest="$repo_root/qemu-tests/Cargo.toml"
target="aarch64-unknown-none"
binary="$repo_root/qemu-tests/target/$target/release/aarch64"
image="$binary.bin"

command -v cargo >/dev/null 2>&1 || {
  echo "cargo is required" >&2
  exit 1
}
command -v qemu-system-aarch64 >/dev/null 2>&1 || {
  echo "qemu-system-aarch64 is required" >&2
  exit 1
}
command -v rust-objcopy >/dev/null 2>&1 || {
  echo "rust-objcopy is required" >&2
  exit 1
}

rustup target list --installed | grep -qx "$target" || {
  echo "Rust target $target is not installed; run: rustup target add $target" >&2
  exit 1
}

RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-T$repo_root/qemu-tests/aarch64-memory.x" \
  cargo build \
    --manifest-path "$manifest" \
    --target-dir "$repo_root/qemu-tests/target" \
    --target "$target" \
    --release \
    --features aarch64 \
    --bin aarch64

rust-objcopy -O binary "$binary" "$image"

output=$(timeout 10 qemu-system-aarch64 \
  -M virt \
  -cpu cortex-a53 \
  -nographic \
  -kernel "$image" 2>&1 || true)
printf '%s\n' "$output"
printf '%s\n' "$output" | grep -q "stack-algebra qemu aarch64: PASS"
