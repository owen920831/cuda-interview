# 环境与故障排除

## 最低环境

- NVIDIA GPU 与可用驱动（先运行 `nvidia-smi`）
- CUDA Toolkit，建议 12.x 或 13.x；必须能运行 `nvcc --version`
- CMake 3.24+、Ninja、支持 C++17 的 host compiler
- Nsight Compute CLI (`ncu`)；Nsight Systems CLI (`nsys`) 建议安装

运行：

```bash
./scripts/check_env.sh
```

脚本只诊断，不会替你修改系统。

## WSL2 常见情况

`nvidia-smi` 可用但 `nvcc` 不存在，表示 Windows GPU 驱动已经透传，但 WSL 内还没有 CUDA Toolkit。安装 Toolkit 时不要在 WSL 里重复安装 Linux GPU driver；依 NVIDIA WSL 指南只安装 toolkit。

如果 Docker GPU passthrough 已经可用，可以不在 WSL host 安装 Toolkit，直接走已验证的 [Docker / NGC 环境](docker.md)。

## 构建模式

```bash
cmake --preset release
cmake --build --preset release -j
ctest --preset release
```

debug build 使用 `-G`，适合查错误，不用于性能比较：

```bash
cmake --preset debug
cmake --build --preset debug -j
compute-sanitizer ./build/debug/reduce_sum --check-only
```

## 性能数据的基本纪律

- 只比较 Release build；
- 先 warm up，再用 CUDA Event 量 GPU 时间；
- correctness 与 benchmark 分开；
- 固定输入尺寸、dtype、GPU、clock/power 环境；
- 报告 median 或多次稳定结果，不用单次最小值讲故事；
- 第一次运行可能含 context/JIT 成本，不纳入 kernel latency。
