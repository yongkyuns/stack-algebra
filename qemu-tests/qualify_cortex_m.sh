#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
manifest="$repo_root/qemu-tests/Cargo.toml"
target="thumbv7em-none-eabihf"
target_dir="$repo_root/qemu-tests/target/qualification"
report_tsv="$repo_root/qemu-tests/resource-report.tsv"
report_md="$repo_root/qemu-tests/resource-report.md"
provenance="$repo_root/qemu-tests/resource-provenance.txt"
global_stack_limit=${STACK_USAGE_LIMIT_BYTES:-8192}

for command in cargo qemu-system-arm rust-size rustc rustup; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "$command is required" >&2
        exit 2
    }
done

if ! rustup target list --installed | grep -qx "$target"; then
    echo "Rust target $target is required" >&2
    exit 2
fi

case "$global_stack_limit" in
    ''|*[!0-9]*)
        echo "STACK_USAGE_LIMIT_BYTES must be a non-negative integer" >&2
        exit 2
        ;;
esac

checkout_commit=unknown
if command -v git >/dev/null 2>&1; then
    checkout_commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf unknown)
fi
source_commit=${QUALIFICATION_SOURCE_SHA:-$checkout_commit}
{
    printf 'source_commit=%s\n' "$source_commit"
    printf 'checkout_commit=%s\n' "$checkout_commit"
    printf 'target=%s\n' "$target"
    printf 'rustc=%s\n' "$(rustc --version)"
    printf 'cargo=%s\n' "$(cargo --version)"
    printf 'rust_size=%s\n' "$(rust-size --version 2>/dev/null || printf unknown)"
    printf 'qemu=%s\n' "$(qemu-system-arm --version | head -n 1)"
    printf 'profile=release opt-level=z lto=true codegen-units=1 debug=true\n'
    printf 'rustflags=%s\n' "${RUSTFLAGS:-}"
    printf 'global_stack_limit_bytes=%s\n' "$global_stack_limit"
    printf 'budget_policy=pinned Rust 1.98.0 workload ceilings with deliberate review for increases\n'
} > "$provenance"

printf 'workload\ttext_bytes\ttext_limit\ttext_delta\tdata_bytes\tdata_delta\tbss_bytes\tbss_delta\tflash_bytes\tstack_used\tstack_limit\tobject_bytes\n' > "$report_tsv"

baseline_text=
baseline_data=
baseline_bss=

for profile in baseline dense3 dense6 dense6-f64 dense15 sparse block-sparse; do
    case "$profile" in
        baseline) text_limit=5200; stack_limit=256 ;;
        dense3) text_limit=11500; stack_limit=768 ;;
        dense6) text_limit=7500; stack_limit=1500 ;;
        dense6-f64) text_limit=10000; stack_limit=2600 ;;
        dense15) text_limit=7600; stack_limit=7200 ;;
        sparse) text_limit=22500; stack_limit=3500 ;;
        block-sparse) text_limit=10000; stack_limit=512 ;;
    esac
    if [ "$stack_limit" -gt "$global_stack_limit" ]; then
        stack_limit=$global_stack_limit
    fi

    feature="resource-$profile"
    binary="$target_dir/$target/release/cortex-m-resource"

    RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Tlink.x" cargo build \
        --manifest-path "$manifest" \
        --target-dir "$target_dir" \
        --target "$target" \
        --release \
        --features "cortex-m,$feature" \
        --bin cortex-m-resource

    set -- $(rust-size "$binary" | tail -n 1)
    text=$1
    data=$2
    bss=$3

    output=$(qemu-system-arm \
        -M mps2-an386 \
        -nographic \
        -monitor none \
        -semihosting-config enable=on,target=native \
        -kernel "$binary" 2>&1)
    printf '%s\n' "$output"

    line=$(printf '%s\n' "$output" | grep -F 'stack-algebra resource cortex-m: PASS' || true)
    if [ -z "$line" ]; then
        echo "missing resource result for $profile" >&2
        exit 1
    fi

    stack_used=$(printf '%s\n' "$line" | sed -n 's/.*stack_used=\([0-9][0-9]*\).*/\1/p')
    object_bytes=$(printf '%s\n' "$line" | sed -n 's/.*object_bytes=\([0-9][0-9]*\).*/\1/p')
    if [ -z "$stack_used" ] || [ -z "$object_bytes" ]; then
        echo "malformed resource result for $profile: $line" >&2
        exit 1
    fi
    if [ "$text" -gt "$text_limit" ]; then
        echo "$profile text size ($text bytes) exceeds budget ($text_limit bytes)" >&2
        exit 1
    fi
    if [ "$stack_used" -gt "$stack_limit" ]; then
        echo "$profile stack usage ($stack_used bytes) exceeds budget ($stack_limit bytes)" >&2
        exit 1
    fi

    if [ "$profile" = baseline ]; then
        baseline_text=$text
        baseline_data=$data
        baseline_bss=$bss
    fi

    text_delta=$((text - baseline_text))
    data_delta=$((data - baseline_data))
    bss_delta=$((bss - baseline_bss))
    flash_bytes=$((text + data))
    workload=$(printf '%s' "$profile" | tr '-' '_')

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$workload" "$text" "$text_limit" "$text_delta" "$data" "$data_delta" \
        "$bss" "$bss_delta" "$flash_bytes" "$stack_used" "$stack_limit" \
        "$object_bytes" >> "$report_tsv"
done

awk -F '\t' '
BEGIN {
    print "| workload | text B | text budget B | Δtext B | data B | Δdata B | bss B | Δbss B | flash B | peak stack B | stack budget B | object B |";
    print "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |";
}
NR > 1 {
    printf "| `%s` | %s | %s | %+d | %s | %+d | %s | %+d | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12;
}
' "$report_tsv" > "$report_md"

printf '\nCortex-M resource provenance:\n'
cat "$provenance"
printf '\nCortex-M resource report:\n'
cat "$report_md"
