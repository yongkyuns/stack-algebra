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
"${compiler}" -std=c++17 -O3 -DNDEBUG \
  "${user_flags[@]}" "${eigen_flags[@]}" "${root_dir}/eigen/native_bench.cpp" \
  -o "${root_dir}/target/eigen-native-bench"

exec "${root_dir}/target/eigen-native-bench" "$@"
