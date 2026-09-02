# Student labs

这里故意没有实现。`src/` 是 reference，`labs/` 是你必须亲手完成、由同一套 CPU reference 验收的版本。

第一次使用请先做 `programming_model_lab.cu`；它覆盖 built-in indices、1D/2D mapping、tail guard、shared memory 与 block barrier，再进入 vector add。

## Docker / NGC（推荐）

WSL host 不需要安装 CMake 或 CUDA Toolkit；所有编译命令都透过已验证的 NGC container：

```bash
./scripts/docker.sh run cmake -S . -B build/labs -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCOURSE_BUILD_LABS=ON

./scripts/docker.sh run cmake --build build/labs \
  --target lab_programming_model

./scripts/docker.sh run ./build/labs/lab_programming_model
```

Day 2 以后再逐题 build/run：

```bash
./scripts/docker.sh run cmake --build build/labs --target lab_vector_add
./scripts/docker.sh run ./build/labs/lab_vector_add --version 0

./scripts/docker.sh run cmake --build build/labs --target lab_reduce
./scripts/docker.sh run ./build/labs/lab_reduce --version all
```

只有在所有 TODO 都完成后，才运行整个 student suite：

```bash
./scripts/docker.sh run ctest --test-dir build/labs --output-on-failure
```

## WSL host 原生工具链（可选）

只有 `make setup` 已确认 host 具备 `nvcc`、CMake 与 Ninja 时，才直接执行：

```bash
cmake -S . -B build/labs -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native -DCOURSE_BUILD_LABS=ON
cmake --build build/labs --target lab_programming_model
./build/labs/lab_programming_model
```

每个 `TODO(student)` 都是一个 gate。允许新增 helper/kernel，不要删除 CPU reference 或把 reference result 直接 copy 到 output。

建议纪律：

1. 第一次只追 correctness；
2. 加一个 odd/non-multiple shape；
3. 写下 bottleneck 预测；
4. 只做一个优化；
5. 再计时/profile；
6. 90 分钟后仍卡住才看 `src/`，关掉后重写。

速度目标以你自己的 GPU、相同 shape、相同 build 为准。仓库不设硬编码 speedup gate，因为小尺寸、架构与工具链都会改变结果；正确性 gate 则不能妥协。
