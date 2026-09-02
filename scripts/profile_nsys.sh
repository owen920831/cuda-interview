#!/usr/bin/env bash
set -euo pipefail

target="${1:-streams}"
version="${2:-all}"
if [[ -z "$target" ]]; then
  echo "usage: $0 <target> [version]"
  exit 2
fi

command -v nsys >/dev/null || { echo "nsys is not installed"; exit 1; }
binary="build/release/$target"
[[ -x "$binary" ]] || { echo "missing $binary; run: make build"; exit 1; }

mkdir -p out
safe_version="${version//[^[:alnum:]_-]/-}"
report="out/${target}-${safe_version}"
program_args=(--bench-only --iters 3)
if [[ "$target" != "streams" ]]; then
  program_args+=(--version "$version")
fi

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas \
  --stats=true \
  -o "$report" \
  "$binary" "${program_args[@]}"

echo "report: ${report}.nsys-rep"
echo "open:   nsys-ui ${report}.nsys-rep"
