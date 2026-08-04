#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
manifest="$repo_root/qemu-tests/Cargo.toml"
target="thumbv7em-none-eabihf"
binary="$repo_root/qemu-tests/target/$target/release/cortex-m"

command -v cargo >/dev/null 2>&1 || {
    echo "cargo is required" >&2
    exit 2
}
command -v qemu-system-arm >/dev/null 2>&1 || {
    echo "qemu-system-arm is required" >&2
    exit 2
}

if ! rustup target list --installed | grep -qx "$target"; then
    echo "Rust target $target is required" >&2
    exit 2
fi

RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Tlink.x" cargo build \
    --manifest-path "$manifest" --target-dir "$repo_root/qemu-tests/target" \
    --target "$target" --release --features cortex-m --bin cortex-m

output=$(qemu-system-arm \
  -M mps2-an386 \
  -nographic \
  -monitor none \
  -semihosting-config enable=on,target=native \
  -kernel "$binary")

printf '%s\n' "$output" | sh "$repo_root/qemu-tests/check_stack_usage.sh" cortex-m
