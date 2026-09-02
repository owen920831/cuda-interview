#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build/release}"
for target in vector_add reduce_sum transpose gemm avg_pool fusion streams; do
  echo
  echo "### $target"
  "$build_dir/$target" --bench-only --iters 20
done
