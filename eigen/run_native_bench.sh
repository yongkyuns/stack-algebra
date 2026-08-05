#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compiler="${CXX:-c++}"
eigen_flags=()

if [[ -n "${EIGEN3_INCLUDE_DIR:-}" ]]; then
  eigen_flags=(-I "${EIGEN3_INCLUDE_DIR}")
elif command -v pkg-config >/dev/null && pkg-config --exists eigen3; then
  read -r -a eigen_flags <<<"$(pkg-config --cflags eigen3)"
else
  echo "Eigen headers not found. Set EIGEN3_INCLUDE_DIR or install pkg-config metadata for eigen3." >&2
  exit 1
fi

user_flags=()
if [[ -n "${CXXFLAGS:-}" ]]; then
  read -r -a user_flags <<<"${CXXFLAGS}"
fi

mkdir -p "${root_dir}/target"
binary="${root_dir}/target/eigen-native-bench"
compile_command=("${compiler}" -std=c++17 -O3 -DNDEBUG)
if ((${#user_flags[@]})); then
  compile_command+=("${user_flags[@]}")
fi
compile_command+=("${eigen_flags[@]}" "${root_dir}/eigen/native_bench.cpp" \
  -o "${binary}")

# The executable accepts the scalar type at runtime, so f32 and f64 can share
# one compilation. Rebuild when the source changed unless explicitly skipped.
needs_build=0
if [[ ! -x "${binary}" ]]; then
  needs_build=1
elif [[ "${EIGEN_BENCH_SKIP_BUILD:-0}" != "1" &&
        "${root_dir}/eigen/native_bench.cpp" -nt "${binary}" ]]; then
  needs_build=1
fi
if [[ "${needs_build}" == "1" ]]; then
  "${compile_command[@]}"
fi

exec "${binary}" "$@"
