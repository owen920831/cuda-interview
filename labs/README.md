# Student labs

这里故意没有实现。`src/` 是 reference，`labs/` 是你必须亲手完成、由同一套 CPU reference 验收的版本。

```bash
cmake -S . -B build/labs -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native -DCOURSE_BUILD_LABS=ON
cmake --build build/labs -j

./build/labs/lab_vector_add --version 0
./build/labs/lab_reduce --version all
ctest --test-dir build/labs --output-on-failure
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
