# 从这里重新 recap CUDA

不要直接从 GEMM 或 TensorRT-LLM 开始。先用一天把 programming model 写进肌肉记忆；Day 1 没通过，就继续做 Day 1。

## 目前什么已经通过

| 范围 | 状态 | 意义 |
|---|---|---|
| `src/` reference | NGC container 编译通过，9 个 CUDA correctness tests 通过 | 完整范例可运行 |
| `tools/` | 9 个 Python tests 通过 | 计算器与 scheduler reference 可运行 |
| `labs/` | 刻意保留 TODO，不应该一开始通过 | 这些是你要亲手完成的题目 |
| NCU/NSYS | container 内已安装 | 实际采集仍受 host performance-counter 权限与 workload 影响 |
| PyTorch/HF、TensorRT-LLM | 有课程、scaffold 与实验规范 | 需要按 Day 17–20 选 model/image 实跑，不声称预先完成 |

所以“环境与完整参考实现都能过”是肯定的；“所有学生 lab 已经过”不是，因为那样你就只是在运行答案。

## 第 0 步：确认环境（5 分钟）

```bash
cd /home/owen920831/Documents/Codex/2026-09-02/https-chatgpt-com-share-6a96f9e0-077c-2
make docker-env
make docker-test
```

预期：RTX 4080、CUDA/CMake/NCU/NSYS 路径，以及 9/9 CUDA tests passed。

## 第 1 天：Programming model（120 分钟）

### 00–35：边读边手写

读 [CUDA Programming Model](docs/cuda-programming-model.md)，在纸上写出：

```text
global_x = blockIdx.x * blockDim.x + threadIdx.x
x = blockIdx.x * blockDim.x + threadIdx.x
y = blockIdx.y * blockDim.y + threadIdx.y
stride = blockDim.x * gridDim.x
```

再画出：grid → block → warp → thread → registers/shared/global memory。

### 35–55：运行并读 reference

```bash
./scripts/docker.sh run ./build/release/programming_model
```

只读 [00_programming_model.cu](src/00_programming_model.cu) 的三个 kernels：

1. `record_1d`
2. `record_2d`
3. `reverse_within_block`

逐行指出每个 built-in index 的值、tail guard 与 barrier 的参与者。

### 55–105：关掉答案，完成 lab

```bash
./scripts/docker.sh run cmake -S . -B build/labs -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCOURSE_BUILD_LABS=ON

./scripts/docker.sh run cmake --build build/labs \
  --target lab_programming_model

./scripts/docker.sh run ./build/labs/lab_programming_model
```

完成 [programming_model_lab.cu](labs/programming_model_lab.cu) 的 7 个 TODO。顺序是 1D mapping → 2D mapping → shared memory/barrier。

### 105–120：闭卷验收

不看文档回答：

1. `threadIdx` 与 `blockIdx` 的 scope 有何不同？
2. `blockDim` 和 `gridDim` 是 threads 数还是坐标维度？
3. 2D block 如何算 row-major offset？
4. 为什么 tail block 需要 guard？
5. `__syncthreads()` 同步谁？为什么 barrier 前不能让部分 threads `return`？
6. `cudaDeviceSynchronize()` 与 `__syncthreads()` 为什么不能互换？
7. atomic 为什么不是 barrier？
8. 不同 blocks 要如何安全地分阶段沟通？

答不出来的题，回到相应代码改一个实验验证。通过后才进入 [COURSE.md](COURSE.md) Day 2 vector add。

## 第一周正确顺序

```text
Day 1  programming model + synchronization
Day 2  naive vector add + boundary/error handling
Day 3  grid-stride + coalescing + float4
Day 4  race + atomic reduction baseline
Day 5  shared memory + __syncthreads block reduction
Day 6  warp/lane + __shfl_down_sync
Day 7  闭卷重写 + 第一次 NCU
```

每一天都先写预测，再做 correctness，最后才 benchmark/profile。
