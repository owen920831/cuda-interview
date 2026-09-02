#!/usr/bin/env bash
set -euo pipefail

target="${1:-}"
if [[ -z "$target" ]]; then
  echo "usage: $0 <target>"
  exit 2
fi

command -v cuobjdump >/dev/null || { echo "cuobjdump is not installed"; exit 1; }
binary="build/release/$target"
[[ -x "$binary" ]] || { echo "missing $binary; run: make build"; exit 1; }
mkdir -p out

cuobjdump --dump-resource-usage "$binary" > "out/${target}-resources.txt"
cuobjdump --dump-ptx "$binary" > "out/${target}.ptx"
cuobjdump --dump-sass "$binary" > "out/${target}.sass"

echo "resource usage: out/${target}-resources.txt"
echo "PTX:            out/${target}.ptx"
echo "SASS:           out/${target}.sass"
