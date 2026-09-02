#!/usr/bin/env bash
set -u

missing=0

check_command() {
  local command_name="$1"
  local description="$2"
  if command -v "$command_name" >/dev/null 2>&1; then
    printf '[ok]      %-8s %s\n' "$command_name" "$(command -v "$command_name")"
  else
    printf '[missing] %-8s %s\n' "$command_name" "$description"
    missing=1
  fi
}

check_optional() {
  local command_name="$1"
  local description="$2"
  if command -v "$command_name" >/dev/null 2>&1; then
    printf '[ok]      %-8s %s\n' "$command_name" "$(command -v "$command_name")"
  else
    printf '[later]   %-8s %s\n' "$command_name" "$description"
  fi
}

check_command nvidia-smi 'NVIDIA driver / WSL GPU passthrough'
check_command nvcc 'CUDA Toolkit compiler'
check_command cmake 'CMake 3.24 or newer'
check_command ninja 'Ninja build tool'
check_command python3 'repo integrity checker'
check_optional ncu 'Nsight Compute CLI (needed from Day 7)'
check_optional nsys 'Nsight Systems CLI (needed from Day 19)'

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
    --format=csv,noheader 2>/dev/null || true
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 1
fi

if [[ "$missing" -ne 0 ]]; then
  printf '\nOne or more tools are missing. See docs/setup.md.\n'
  exit 1
fi

printf '\nEnvironment is ready. Next: make build && make test\n'
