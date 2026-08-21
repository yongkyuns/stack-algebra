#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

out_dir=${RELEASE_ARTIFACT_DIR:-release-artifacts}
nightly=${PUBLIC_API_NIGHTLY:-nightly-2026-08-20}
public_api_version=${PUBLIC_API_TOOL_VERSION:-0.51.0}

for command in cargo rustc rustup git sha256sum; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "$command is required" >&2
        exit 2
    }
done

if [ -n "$(git status --porcelain)" ]; then
    echo "release artifacts require a clean checkout" >&2
    exit 2
fi

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

checkout_commit=$(git rev-parse HEAD)
source_commit=${QUALIFICATION_SOURCE_SHA:-$checkout_commit}
ref=$(git symbolic-ref --short -q HEAD || git describe --always --exact-match 2>/dev/null || printf detached)

cargo +"$nightly" public-api -sss > "$out_dir/public-api.txt"
cargo +"$nightly" rustdoc --lib -- -Z unstable-options --output-format json
cp target/doc/stack_algebra.json "$out_dir/rustdoc-public-api.json"

cargo metadata --locked --format-version 1 > "$out_dir/cargo-metadata.json"
cargo tree --locked --edges normal,build > "$out_dir/dependency-tree.txt"

cargo package --locked
package=$(find target/package -maxdepth 1 -type f -name 'stack-algebra-*.crate' | sort | tail -n 1)
if [ -z "$package" ]; then
    echo "cargo package did not produce a stack-algebra crate archive" >&2
    exit 1
fi
cp "$package" "$out_dir/"

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
    printf 'public_api_sha256=%s\n' "$(sha256sum "$out_dir/public-api.txt" | awk '{print $1}')"
    printf 'rustdoc_json_sha256=%s\n' "$(sha256sum "$out_dir/rustdoc-public-api.json" | awk '{print $1}')"
} > "$out_dir/provenance.txt"

cp Cargo.lock "$out_dir/Cargo.lock"
printf '%s\n' "$source_commit" > "$out_dir/source-commit.txt"
printf 'Release artifacts written to %s\n' "$out_dir"
