#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
manifest="$repo_root/qemu-tests/Cargo.toml"
target="thumbv7em-none-eabihf"
target_dir="$repo_root/qemu-tests/target/qualification"
report_tsv="$repo_root/qemu-tests/resource-report.tsv"
report_md="$repo_root/qemu-tests/resource-report.md"
stack_limit=${STACK_USAGE_LIMIT_BYTES:-8192}

for command in cargo qemu-system-arm rust-size rustup; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "$command is required" >&2
        exit 2
    }
done

if ! rustup target list --installed | grep -qx "$target"; then
    echo "Rust target $target is required" >&2
    exit 2
fi

case "$stack_limit" in
    ''|*[!0-9]*)
        echo "STACK_USAGE_LIMIT_BYTES must be a non-negative integer" >&2
        exit 2
        ;;
esac

printf 'workload\ttext_bytes\ttext_delta\tdata_bytes\tdata_delta\tbss_bytes\tbss_delta\tflash_bytes\tstack_used\tstack_limit\tobject_bytes\n' > "$report_tsv"

baseline_text=
baseline_data=
baseline_bss=

for profile in baseline dense3 dense6 dense15 sparse block-sparse; do
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
    if [ "$stack_used" -gt "$stack_limit" ]; then
        echo "$profile stack usage ($stack_used bytes) exceeds configured limit ($stack_limit bytes)" >&2
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

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$workload" "$text" "$text_delta" "$data" "$data_delta" "$bss" "$bss_delta" \
        "$flash_bytes" "$stack_used" "$stack_limit" "$object_bytes" >> "$report_tsv"
done

awk -F '\t' '
BEGIN {
    print "| workload | text B | Δtext B | data B | Δdata B | bss B | Δbss B | flash B | peak stack B | object B |";
    print "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |";
}
NR > 1 {
    printf "| `%s` | %s | %+d | %s | %+d | %s | %+d | %s | %s | %s |\n", $1, $2, $3, $4, $5, $6, $7, $8, $9, $11;
}
' "$report_tsv" > "$report_md"

printf '\nCortex-M resource report:\n'
cat "$report_md"
