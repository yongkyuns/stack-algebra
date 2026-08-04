#!/usr/bin/env sh
set -eu

target=$1
output=$(cat)
printf '%s\n' "$output"

line=$(printf '%s\n' "$output" | grep -F "stack-algebra qemu $target: PASS" || true)
if [ -z "$line" ]; then
  echo "missing stack usage line for $target" >&2
  exit 1
fi

used=$(printf '%s\n' "$line" | sed -n 's/.*stack_used=\([0-9][0-9]*\).*/\1/p')
budget=$(printf '%s\n' "$line" | sed -n 's/.*stack_budget=\([0-9][0-9]*\).*/\1/p')
if [ -z "$used" ] || [ -z "$budget" ]; then
  echo "malformed stack usage line for $target: $line" >&2
  exit 1
fi

if [ "$used" -gt "$budget" ]; then
  echo "$target stack usage ($used bytes) exceeds linker budget ($budget bytes)" >&2
  exit 1
fi

limit=${STACK_USAGE_LIMIT_BYTES:-$budget}
case "$limit" in
  ''|*[!0-9]*)
    echo "STACK_USAGE_LIMIT_BYTES must be a non-negative integer" >&2
    exit 1
    ;;
esac
if [ "$used" -gt "$limit" ]; then
  echo "$target stack usage ($used bytes) exceeds configured limit ($limit bytes)" >&2
  exit 1
fi
