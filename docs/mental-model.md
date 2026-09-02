# CUDA 性能心智模型

每次优化前只问两个一级问题：GPU 在等资料，还是在等计算？

## 1. 从映射开始

```text
problem element/tile
        ↓
grid → block → warp(32 lanes) → thread
        ↓
SM resources: registers + shared memory + warp slots
```

先写清一个 thread 负责什么、一个 block 共同处理什么。边界判断必须覆盖非 32 倍数与非方阵，不能只测“漂亮尺寸”。

## 2. 再看 bytes 与 reuse

- Global memory：容量大、延迟高；先检查相邻 lane 是否访问相邻地址。
- Shared memory：block scope 的 software-managed storage；用来重排访问或复用资料。
- Registers：thread-private accumulator；增加 ILP/reuse，但过多会降低 residency 或 spill。
- Cache：硬体管理，不能把“可能命中”当优化论证。

shared memory 不是自动加速。若资料只用一次、原访问已 coalesced，加入 copy 与 barrier 可能更慢。

## 3. 同步范围必须匹配沟通范围

- `__syncthreads()`：整个 block 的 barrier；所有存活 threads 必须一致到达。
- `__syncwarp(mask)`：参与 mask 的 warp lanes。
- `__shfl_down_sync`：warp 内直接交换 register value，常用于 reduction。
- 不同 block 默认不能在同一个普通 kernel 内做 global barrier。

## 4. 用 Roofline 约束幻想

```text
arithmetic intensity = operations / bytes moved
attainable performance ≤ min(peak compute, bandwidth × intensity)
```

Vector add、pool、reduce 通常偏 memory-bound；GEMM 透过 tiling 提高 reuse 后可转向 compute-bound。优化时要同时写出预期减少多少 bytes、增加多少 reuse 或暴露多少 parallelism。

## 5. Occupancy 是条件，不是目标

occupancy 太低可能无法隐藏 latency，但最高 occupancy 不保证最快。register tile 会增加每 thread registers，却可能因减少 global loads、增加 ILP 而更快。正确做法是一起看 achieved occupancy、eligible warps、stall reason、memory/compute utilization 与实际时间。

## 6. 本课的固定优化问句

每一版都回答：

1. 正确性如何验证？包含了哪些边界？
2. 目前 global load/store traffic 是什么？
3. access coalesced 吗？有 redundant load 吗？
4. divergence、barrier、atomic contention 在哪里？
5. registers/shared memory 的代价是什么？
6. 预期瓶颈与 profiler 证据一致吗？
7. 下一版只改变哪一个主要变量？
