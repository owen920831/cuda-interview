#!/usr/bin/env bash
set -euo pipefail

target="${1:-}"
version="${2:-all}"
kernel_regex="${3:-}"
if [[ -z "$target" ]]; then
  echo "usage: $0 <target> [version] [kernel-regex]"
  echo "example: $0 reduce_sum v3 reduce_v3"
  exit 2
fi

command -v ncu >/dev/null || { echo "ncu is not installed"; exit 1; }
binary="build/release/$target"
[[ -x "$binary" ]] || { echo "missing $binary; run: make build"; exit 1; }

mkdir -p out
safe_version="${version//[^[:alnum:]_-]/-}"
report="out/${target}-${safe_version}"
ncu_set="${NCU_SET:-full}"

ncu_args=(
  --force-overwrite
  --target-processes all
  --kernel-name-base demangled
  --set "$ncu_set"
  -o "$report"
)
if [[ -n "$kernel_regex" ]]; then
  ncu_args+=(--kernel-name "regex:$kernel_regex")
fi

ncu "${ncu_args[@]}" \
  "$binary" --bench-only --version "$version" --warmup 1 --iters 1

echo "report: ${report}.ncu-rep"
echo "open:   ncu-ui ${report}.ncu-rep"
