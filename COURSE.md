# 21 天实作课程（每天约 120 分钟）

## 使用规则

每天固定节奏：

```text
00–15  不看答案写出今天的性能预测与 kernel mapping
15–40  只补当天必需知识
40–100 写 code、跑 correctness、修边界
100–115 benchmark/profile，并记录证据
115–120 commit 或写下明确 blocker
```

如果 60 分钟实作还没通过测试，可以看 `docs/`，仍不要看 `src/`。到 90 分钟才允许对照 reference，而且看完必须关掉 reference，从空白重写关键部分。

课程假设你会基本 C++（pointer、RAII、template 不要求熟练）。若 C++ 编译/ownership 卡住，把当日 profiler 时间挪来补 [C++/CUDA 补齐清单](docs/prerequisites.md)，但总时数仍控制在两小时。

---

## Week 1：从 execution model 到 reduction

### Day 1 — 环境、测量闭环与 GPU 映射

**知识（25 分）**：host/device、kernel launch、grid/block/thread、warp=32、SM、global/shared/register scope；Release 与 `-G` 的区别。

**实作（70 分）**：运行 `make setup`；读 `src/00_device_info.cu`；画出 100003 个元素如何映射到 256-thread blocks。建立 release build，并运行 `device_info`。

**验收（25 分）**：在记录中写出 RTX 4080 的 compute capability、SM count、warp size、max threads/block；解释为什么 GPU 可见不代表 `nvcc` 已安装。

### Day 2 — Vector add naive：先把 correctness 做硬

**知识（20 分）**：linear index、ceil division、H2D/D2H、异步 launch 与错误检查。

**实作（80 分）**：开启 labs：

```bash
cmake -S . -B build/labs -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native -DCOURSE_BUILD_LABS=ON
cmake --build build/labs --target lab_vector_add
./build/labs/lab_vector_add --version 0
```

完成 `labs/vector_add_lab.cu` 的 v0。必须通过 `n=1,31,100003`，不可只测 2 的幂。

**验收（20 分）**：解释漏掉 `index < n` 会发生什么，并展示 `compute-sanitizer` 或 correctness 输出。

### Day 3 — Coalescing、grid-stride 与 vectorized access

**知识（25 分）**：warp 的 global transactions、alignment、`__restrict__`、grid-stride loop、`float4` tail。

**实作（70 分）**：完成 vector lab v1/v2；用 CUDA Event 比三版；计算 logical bytes `3*N*sizeof(float)` 与 effective GB/s。

**验收（25 分）**：v2 必须对非 4 倍数正确；写出 vectorization 没变快的两个合理原因（bandwidth ceiling、指令/transaction 已足够好等）。

### Day 4 — Reduction 问题拆解与 atomic baseline

**知识（30 分）**：sum 的 associativity、floating-point order、race、atomic contention、为什么 CPU/GPU 不能期待 bitwise equal。

**实作（65 分）**：先自己在新文件写 per-element `atomicAdd` baseline，与 CPU double accumulation 对比；再跑 `src/02_reduce.cu` v0。

**验收（25 分）**：至少测试 `n=1,31,1025,2^20+13`；记录容差与理由，不得用“差不多”。

### Day 5 — Shared-memory block reduction

**知识（25 分）**：block cooperation、barrier、sequential addressing、divergence；`__syncthreads()` 放在条件分支里的 deadlock 风险。

**实作（75 分）**：完成 `labs/reduce_lab.cu` v0（shared tree + one atomic/block）。画出 8 threads 的每轮 shared index。

**验收（20 分）**：test 通过；说明 global atomic 次数从 O(N) 变成多少；指出 shared memory 读写和 barrier 次数。

### Day 6 — First-add、warp shuffle 与 vectorized reduce

**知识（25 分）**：load two values/thread、warp shuffle、lane/warp id、block 最后一层 reduction。

**实作（75 分）**：完成 reduce lab v1/v2；再独立加一个 `float4` load 版本。先预测哪版快，再量。

**验收（20 分）**：不看答案解释 `__shfl_down_sync(mask, value, offset)` 五轮如何把 32 lanes 合并；所有 odd sizes 正确。

### Day 7 — Week 1 闭卷 gate + 第一次 Nsight Compute

**闭卷（60 分）**：从空文件写出 grid-stride vector add 与 block+warp reduce；45 分钟内编译、通过 odd-size test。

**Profiler（45 分）**：先读 [NCU 单 kernel 指南](docs/profiling/ncu.md)，对 atomic 与 shuffle 版各采一份报告，只比较 launch、memory、scheduler 与 atomic 相关证据。

**复盘（15 分）**：用五句话回答“为什么优化版快”，每句必须可由 code 或 metric 支持。未通过闭卷则 Day 8 前先重做，不增加新内容。

---

## Week 2：资料重排与 GEMM 优化阶梯

### Day 8 — Matrix transpose 与 strided access

**知识（25 分）**：row-major indexing、read coalesced/write strided、effective bandwidth。

**实作（75 分）**：完成 transpose lab v0；支持非方阵与 odd dimensions。手算 `input[y*W+x] → output[x*H+y]`。

**验收（20 分）**：用 `777×1003` exact compare；说明读和写哪一侧 coalesced。

### Day 9 — Shared tile 与 bank-conflict padding

**知识（25 分）**：shared memory bank mapping、为什么 `[32][32]` column access 冲突、为何 `[32][33]` 能打散。

**实作（70 分）**：完成 transpose lab v1/v2；用同一个 32×8 block 循环搬 32×32 tile。

**验收（25 分）**：三版 correctness 全过；从 Nsight Compute 比较 shared bank conflicts 与时间，结果若与预测不同也要记录。

### Day 10 — GEMM naive 与算术强度

**知识（30 分）**：`C[M,N]=A[M,K]×B[K,N]`、2D launch、FLOPs=`2MNK`、naive 重复 global loads。

**实作（70 分）**：完成 `labs/gemm_lab.cu` v0；先用 `M=3,N=5,K=2` 手算，再测 `129×133×71`。

**验收（20 分）**：边界正确；报告 GFLOP/s；列出一个 output 需要的 loads/FMA。

### Day 11 — Shared-memory tiled GEMM

**知识（20 分）**：block tile、cooperative load、reuse、两个 barrier、K-tail zero fill。

**实作（80 分）**：完成 GEMM lab v1，不允许假设 M/N/K 是 tile 倍数；比较 tile 8/16/32（32 可能受 1024-thread 与资源影响）。

**验收（20 分）**：解释每个 A/B tile 在 block 内复用几次；提供三种 tile 的 correctness+latency。

### Day 12 — Register tiling / thread coarsening

**知识（25 分）**：一 thread 多 outputs、register accumulator、ILP、register pressure 与 occupancy trade-off。

**实作（75 分）**：完成 GEMM lab v2，让一个 thread 计算至少 2 个 N 方向 outputs；比较 1/2/4 outputs per thread。

**验收（20 分）**：记录 registers/thread、achieved occupancy、GFLOP/s；不能只凭 occupancy 决定优劣。

### Day 13 — cuBLAS ceiling、Roofline 与 profiler-driven tuning

**知识（30 分）**：library ceiling、row-major 与 column-major 视角、arithmetic intensity、memory/compute roof。

**实作（65 分）**：先完成 [Roofline 公式与计算器](docs/profiling/roofline.md)，再运行 `gemm` 的 v0/v1/v2/cuBLAS；用 Nsight Compute 分别看 v0、最佳自写版与 cuBLAS（采集 library kernel 时注意 kernel 数量）。

**验收（25 分）**：报告自写版达到 cuBLAS 的百分比；指出差距可能来自 Tensor Core、pipeline、tile hierarchy、vectorized loads 中的哪些证据。不要把 Tensor Core 当本周必写项目。

### Day 14 — Week 2 闭卷 gate

**闭卷（70 分）**：从空白写一个 boundary-safe tiled GEMM，60 分钟内通过 `M=129,N=133,K=71`。

**诊断（35 分）**：刻意制造一个 missing barrier 或错误 leading dimension，用 sanitizer/test 找到它。

**复盘（15 分）**：口头讲完整 ladder：naive → coalesced/cooperative load → shared reuse → register tile → library ceiling。

---

## Week 3：真实算子、LLM inference 与双岗位 capstone

本周同时对齐 [TensorRT-LLM 与 DevTech HPC/AI 两个岗位](JOB_ALIGNMENT.md)。每天仍是两小时；每个新名词都必须落到计算、code、trace 或口头设计。

### Day 15 — Average Pool：naive → shared → specialized

**知识（20 分）**：NCHW、`OH=(H-K)/S+1`、window/stride、overlap、halo；input tile 尺寸 `(OUT_TILE-1)*stride+kernel`。

**实作（85 分）**：完成 avg-pool lab v0/v1；v0 direct，v1 cooperative-load input tile。若时间足够做 v2 的 2×2/S2 specialized path；不满足条件必须 fallback。

**验收（15 分）**：odd H/W correctness；分别测 K3/S1 与 K2/S2，解释 shared 版为什么在无 overlap 时可能更慢。

### Day 16 — Fusion、occupancy、streams：从 kernel 到 timeline

**知识（25 分）**：fusion 的 launch/traffic 收益与 register 代价；occupancy 不等于性能；pinned memory、stream ordering、overlap 条件。

**实作（75 分）**：对 `src/06_fusion.cu` 做 two-kernel vs fused A/B；再运行 `src/07_streams.cu` 的 streams=1/2/4，用 NSYS 观察 copy/compute。对最佳 fused kernel 保存 registers、occupancy 与 latency。

**验收（20 分）**：一张完整 timeline + 一张 block/register/occupancy/latency 表；所有结论附 metric，不用“GPU utilization 高”代替分析。

### Day 17 — PyTorch/Hugging Face：从 model 追到 operators

**知识（30 分）**：decoder-only block、RMSNorm/LayerNorm、QKV projection、RoPE、attention、MLP、residual、LM head；MHA/GQA/MQA；prefill 与 decode。

**实作（70 分）**：完成 `labs/llm-inference/pytorch_hf_trace.py`：pin model revision/versions，分别 profile 短/长 prompt 的 prefill 与至少 8 个 decode steps，记录 op shapes，并比较 `use_cache=False/True`。

**验收（20 分）**：画出 model→PyTorch op→GEMM/attention/reduce/elementwise kernel 的对应；指出这两周写的 GEMM、reduce、fusion 分别落在哪。

### Day 18 — LLM 容量：weights、KV cache、TTFT/TPOT

**知识（35 分）**：读 [基本模型](docs/llm-inference/fundamentals.md) 与 [白板计算](docs/llm-inference/calculations.md)；区分 weight、KV、activation/workspace、allocator headroom。

**实作（65 分）**：不看 reference 完成 `kv_cache_lab.py`；选一个真实 model config 手算 FP16/BF16 或量化 weight memory、KV bytes/token、4K/8K context 并发上界，再用 `tools/llm_calculator.py` 检查。

**验收（20 分）**：10 分钟闭卷容量题误差 <2%；说明为什么 TP degree 不一定就是 KV shard factor；用逐 token timestamp 算 TTFT、TPOT、E2E。

### Day 19 — TensorRT-LLM runtime：paged KV、IFB 与 scheduler

**知识（30 分）**：读 [TensorRT-LLM](docs/llm-inference/tensorrt-llm.md) 与 [Serving](docs/llm-inference/serving.md)；paged allocation、prefix reuse、continuous batching、chunked prefill、preemption、speculative decoding。

**实作（70 分）**：不看 reference 完成 `scheduler_lab.py` 的 static/continuous 两版；用 `[2,8,3,6,1]`、batch=2 输出 timeline。之后加入 arrival time 或 prefill chunk 中一项。

**验收（20 分）**：从 request lifecycle 画到 KV blocks 与每轮 packed batch；构造 throughput 提升但 P99 恶化的 workload，并提出可证伪的 A/B。

### Day 20 — TensorRT 与 TensorRT-LLM 部署/性能实验

**知识（30 分）**：读 [TensorRT](docs/llm-inference/tensorrt.md)、[量化与多 GPU](docs/llm-inference/optimization.md)、[LLM profiling](docs/llm-inference/profiling.md)。理解 builder→tactic→engine→context、MIN/OPT/MAX profile、plugin、quantization、TP/PP/DP/EP。

**实作（70 分）**：在可用环境用小 ONNX model 建 TensorRT engine，跑 small/OPT/MAX/odd shape matrix；再用 pinned TensorRT-LLM container/version 跑一次离线 generate 或 `trtllm-bench`。硬件/套件暂不可用时，先交命令、config、预期 artifacts 与 profiler 实验设计，不伪造结果。

**验收（20 分）**：能区分 TensorRT、TensorRT-LLM、Triton；报告版本、GPU、shape、precision、warmup、TTFT/TPOT/throughput/P99，且指出 NSYS→NCU 下钻路径。

### Day 21 — 双岗位 capstone 与 final gate

**实作（65 分）**：从 reduce/GEMM/avg pool 选一个，从空白做 CPU reference→naive→optimized→odd-shape tests→CUDA Event→NCU；并把同一证据写成两种说明：

- TensorRT-LLM team：连接到 model operator/runtime feature、LLM SLO 与 test/fallback。
- DevTech team：连接到 customer workload、Roofline、平台限制与可移植性。

**系统题（35 分）**：完成一题 KV capacity + 一题 continuous batching + 一题 unknown HPC workload profiling plan。

**表达（15 分）**：英文 5 分钟：problem、baseline、hypothesis、evidence、trade-off、next step；准备 technical appendix 回答 C++ ownership、GPU architecture、PyTorch/HF、quant/multi-GPU。

**收尾（5 分）**：运行 `make test && make check`，完成 [LLM 面试清单](docs/llm-inference/interview-checklist.md) 与 [CUDA 面试清单](docs/interview-checklist.md)。未通过项变成下一轮 backlog。

---

## 三周后的延伸（不计入主线）

1. CUTLASS/CuTe 的 threadblock→warp→instruction tile。
2. Tensor Core MMA/WMMA 与 mixed precision correctness。
3. `cp.async`、multi-stage pipeline、TMA、warp specialization（依 GPU 架构选择）。
4. LayerNorm/Softmax 与 attention kernel 的完整 CUDA optimization ladder。
5. Triton 写同一组算子，与 CUDA reference 比较生成 code 与开发效率。
6. Graph analytics 与 DPU/network offload：用相同的 data-movement/Roofline 方法迁移，不把名词当实作。
