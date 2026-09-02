# 先前知识补齐清单

只补会阻碍当天实作的部分，不先读完整本书。

## C++（最多 2–3 小时，分散补）

- pointer/array 与 row-major linear index；
- `const`、`__restrict__` 的承诺；
- RAII 为什么用于 device buffer/event/handle；
- template 常量如何让 loop unroll；
- integer overflow：shape 乘积使用 `size_t`；
- 编译、链接、debug/release、undefined behavior。

## 数值

- IEEE-754 float、rounding；
- reduction 改变加法顺序，因此用 abs+relative tolerance；
- FMA 与 CPU reference 可能产生细微差异；
- mixed precision 必须分别谈 input、accumulator、output dtype。

## 平行程序正确性

- race：多个 threads 未协调读写同一地址；
- barrier 不等于 atomic，atomic 不等于 global barrier；
- block 间默认无同步；
- kernel launch 异步，错误常在之后的 synchronize 才显现。

## 线性代数与张量 layout

- `A[M,K] * B[K,N] = C[M,N]`；
- NCHW index：`((n*C+c)*H+h)*W+w`；
- transpose 改 logical layout，不只是改变量名；
- stride、leading dimension 与 contiguous 的区别。
