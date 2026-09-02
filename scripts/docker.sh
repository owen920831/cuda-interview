#!/usr/bin/env bash
set -euo pipefail

course_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
course_image="${COURSE_NGC_IMAGE:-nvcr.io/nvidia/pytorch:25.02-py3}"
course_action="${1:-help}"

mkdir -p "${course_root}/.cache/container-home"

docker_args=(
  run --rm
  --gpus all
  --ipc=host
  --entrypoint bash
  --user "$(id -u):$(id -g)"
  --env HOME=/workspace/.cache/container-home
  --volume "${course_root}:/workspace"
  --workdir /workspace
)

run_bash() {
  local course_script="$1"
  docker "${docker_args[@]}" "${course_image}" -lc "${course_script}"
}

case "${course_action}" in
  env)
    run_bash 'nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader && nvcc --version | tail -n 1 && cmake --version | head -n 1 && ninja --version && command -v ncu && command -v nsys'
    ;;
  build)
    run_bash 'cmake --preset release && cmake --build --preset release -j'
    ;;
  test)
    run_bash 'cmake --preset release && cmake --build --preset release -j && ctest --preset release'
    ;;
  check)
    run_bash 'make check'
    ;;
  shell)
    docker "${docker_args[@]}" --interactive --tty "${course_image}"
    ;;
  run)
    shift
    if [[ "$#" -eq 0 ]]; then
      echo 'usage: ./scripts/docker.sh run <command> [args...]' >&2
      exit 2
    fi
    docker "${docker_args[@]}" "${course_image}" -lc 'exec "$@"' course-command "$@"
    ;;
  help|-h|--help)
    cat <<'USAGE'
Usage: ./scripts/docker.sh <action>

Actions:
  env                 Show GPU and toolchain versions in the container
  build               Configure and build the Release preset
  test                Build and run all CUDA correctness tests
  check               Run GPU-independent repository checks
  shell               Open an interactive shell in the NGC container
  run <command...>    Run one repository command in the container

Override the pinned image with COURSE_NGC_IMAGE.
USAGE
    ;;
  *)
    echo "unknown action: ${course_action}" >&2
    exit 2
    ;;
esac
