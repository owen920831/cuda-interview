# Docker / NGC 环境

这是 Windows + WSL2 的推荐运行方式：Windows 提供 NVIDIA driver，Docker 使用 GPU passthrough，CUDA Toolkit、compiler、CMake、Ninja、PyTorch 与 Nsight CLI 留在 NGC image 内。

本 repo 默认固定：

```text
nvcr.io/nvidia/pytorch:25.02-py3
```

在目前的 RTX 4080 环境已经实际验证：container 能看到 GPU，CUDA 12.8 能编译全部 targets，8 个 CUDA correctness tests 全部通过。

## 一分钟开始

在 WSL terminal 的 repo root：

```bash
make docker-env
make docker-test
```

或直接使用脚本：

```bash
./scripts/docker.sh env
./scripts/docker.sh build
./scripts/docker.sh test
./scripts/docker.sh shell
```

build output 留在 host 的 `build/`，所以 container 退出后仍可继续用，而且不会写进 Git。

## 执行单一程序或 profiler

```bash
./scripts/docker.sh run ./build/release/reduce_sum
./scripts/docker.sh run ./build/release/gemm --check-only
./scripts/docker.sh run ./scripts/profile_nsys.sh streams
./scripts/docker.sh run ./scripts/profile_ncu.sh reduce_sum v3 reduce_v3
```

NCU 若报告没有 performance-counter 权限，这是 host driver 的 profiling permission，不是 CUDA compiler 或 container 没安装。不要直接把 container 改成 `--privileged` 当永久解法；先确认 Windows/NVIDIA driver 的 performance-counter policy。

## 换 NGC image

若已有其他 NGC PyTorch image：

```bash
COURSE_NGC_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3 \
  ./scripts/docker.sh test
```

换 image 后必须记录 image tag、CUDA、compiler、TensorRT/PyTorch 版本并完整重建。不同 toolkit 不能共用旧的 CMake cache；必要时建立新的 binary directory，或先移走对应的 `build/release`。

## NGC 登录

当前 image 已在本机，不需要重新 pull。若未来 pull private/restricted NGC image，按 NGC portal 的认证流程执行 `docker login nvcr.io`；API key 只输入 credential helper/终端提示，不写入 repo、脚本或 shell history。
