# CUDA 图解：从 thread mapping 到 memory traffic

这份图解先建立空间关系，再进入 profiling。每张图都对应一个必须能手算或在 code 中指出的位置。

## 1. Grid、block、warp、SM

```mermaid
flowchart TB
  LAUNCH[kernel<<<gridDim, blockDim>>>] --> GRID[Grid: one kernel launch]
  GRID --> B0[Block 0]
  GRID --> B1[Block 1]
  GRID --> BN[Block ...]

  subgraph GPU[GPU]
    subgraph SM0[SM 0]
      RB0[resident block A]
      RB1[resident block B]
      W0[warp 0: lanes 0..31]
      W1[warp 1: lanes 0..31]
      REG0[registers per thread]
      SH0[shared memory per block]
      RB0 --> W0
      RB0 --> W1
      W0 --> REG0
      RB0 --> SH0
    end
    subgraph SM1[SM 1]
      RB2[resident block C]
      WX[warps scheduled over time]
      RB2 --> WX
    end
  end

  B0 -. scheduled whole .-> RB0
  B1 -. scheduled whole .-> RB2
  BN -. waits or runs on an SM .-> RB1
```

一个 block 只驻留在一个 SM；同一 block 的 threads 才能用普通 shared memory 和 `__syncthreads()` 协作。warp 是 32 个连续 thread IDs 的 scheduling group。

## 2. 一维 index 怎么算

```mermaid
flowchart LR
  BI[blockIdx.x] --> MUL[blockIdx.x × blockDim.x]
  BD[blockDim.x] --> MUL
  TI[threadIdx.x] --> ADD[global index = block offset + threadIdx.x]
  MUL --> ADD
  ADD --> GUARD{index < N?}
  GUARD -->|yes| ADDR[address = base + index × sizeof T]
  GUARD -->|no| EXIT[do not access memory]
```

以 `N=100003`、`blockDim=256` 为例：

```text
gridDim = ceil(100003 / 256) = 391 blocks
launched threads = 391 × 256 = 100096
inactive tail threads = 100096 - 100003 = 93
```

tail guard 是 correctness contract；多出的 threads 不是 bug。

## 3. Memory hierarchy 与 scope

```mermaid
flowchart TB
  subgraph Thread[Thread scope]
    R[Registers<br/>accumulator · scalar]
    L[Local memory<br/>logical thread-private, physically off-chip/cacheable]
  end
  subgraph Block[Block scope]
    S[Shared memory<br/>tile · cooperation · reorder]
  end
  subgraph Device[Device scope]
    C1[L1/TEX cache]
    C2[L2 cache]
    G[Global / DRAM<br/>all blocks and kernels]
  end
  R --> C1
  L --> C1
  S --> C1
  C1 <--> C2 <--> G
```

这不是严格 latency 比例图。关键是 ownership、scope、capacity 与 traffic：register tile 增加 thread-private reuse；shared tile 增加 block reuse；二者都可能降低 residency。

## 4. Coalescing：warp 地址是否连续

```mermaid
flowchart TB
  subgraph Good[Coalesced pattern]
    WG[warp lanes 0,1,2,...,31] --> AG[address base + lane]
    AG --> TG[few aligned memory sectors]
  end
  subgraph Bad[Strided pattern]
    WB[warp lanes 0,1,2,...,31] --> AB[address base + lane × stride]
    AB --> TB[many sectors, many unused bytes]
  end
  TG --> SAME[same useful float count]
  TB --> SAME
```

coalescing 是一个 warp 的地址集合属性，不是“用了连续数组”就自动成立。Matrix transpose v0 的 input read 连续，但 output write 以 `height` 为 stride。

## 5. Shared bank conflict 为什么 `+1` 有效

对 32-bit words，可先用 `bank = linear_index mod 32` 建立直觉。

```mermaid
flowchart LR
  subgraph Tile32[tile 32 × 32, transposed column access]
    L0[lanes x = 0..31] --> I0[index = x × 32 + fixed]
    I0 --> B0[bank = fixed for every lane]
    B0 --> C0[32-way conflict]
  end
  subgraph Tile33[tile 32 × 33, padded]
    L1[lanes x = 0..31] --> I1[index = x × 33 + fixed]
    I1 --> B1[bank = x + fixed mod 32]
    B1 --> C1[32 distinct banks]
  end
```

padding 没有改变 global matrix shape；它只改变 shared-memory row stride，让同一个 warp 的 bank index 旋转。

## 6. 同步的范围

```mermaid
flowchart TB
  DATA[producer writes data] --> WHO{consumer 在哪里?}
  WHO -->|same thread| NONE[program order; usually no barrier]
  WHO -->|same warp| WARP[warp primitive / __syncwarp mask]
  WHO -->|same block| BLOCKSYNC[__syncthreads or block primitive]
  WHO -->|different blocks| GRID[split kernels, cooperative mechanism,<br/>or redesign; no ordinary block barrier]
  BLOCKSYNC --> RULE[all participating live threads<br/>must reach compatible barrier]
```

barrier 解决 ordering/visibility，不自动解决多个 threads 同时更新同一地址的 race；后者还需要 ownership、reduction 或 atomic。

## 7. 从代码改变到 profiler metric

```mermaid
flowchart LR
  CHANGE[code change] --> MAP[mapping / addresses / reuse]
  MAP --> TRAFFIC[transactions · bytes · bank conflicts]
  MAP --> RESOURCE[registers · shared · blocks]
  MAP --> INST[instruction count · dependency]
  TRAFFIC --> READY[warp ready / stalled]
  RESOURCE --> READY
  INST --> READY
  READY --> ISSUE[scheduler issue rate]
  ISSUE --> TIME[kernel duration]
  TIME --> CLAIM[performance claim]
```

因此“改成 shared memory”不是解释。完整解释要走完：它如何改变地址/reuse → traffic/resource → ready warps/issue → measured duration。

接下来阅读 [完整 profiling 路线](profiling.md) 和 [算子案例图](profiling/case-studies.md)。
