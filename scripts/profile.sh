#!/usr/bin/env bash
set -euo pipefail

target="${1:-}"
version="${2:-all}"
kernel_regex="${3:-}"
if [[ -z "$target" ]]; then
  echo "usage: $0 <target> [version] [kernel-regex]"
  exit 2
fi
if [[ "$target" == "streams" ]]; then
  exec "$(dirname "$0")/profile_nsys.sh" "$target"
else
  exec "$(dirname "$0")/profile_ncu.sh" "$target" "$version" "$kernel_regex"
fi
