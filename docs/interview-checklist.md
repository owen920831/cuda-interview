# 最终闭卷 / 面试验收

对每题先口答，再用白板或空文件写；看答案不计通过。

- [ ] 从 `threadIdx/blockIdx` 推导 1D 与 2D global index。
- [ ] 解释 grid、block、warp、SM，以及 block 为什么不能任意互相同步。
- [ ] 画出 register/shared/L1/L2/global 的 scope 与用途。
- [ ] 判断一段 global access 是否 coalesced。
- [ ] 用 bank modulo 解释 `[32][32]` transpose conflict 与 `+1` padding。
- [ ] 写 boundary-safe vector add 与 grid-stride loop。
- [ ] 写 shared tree reduction，再改成 warp shuffle。
- [ ] 解释 `__syncthreads()` 的两个常见 correctness bug。
- [ ] 写 naive GEMM 与 shared tiled GEMM，包括 K tail。
- [ ] 解释 register tiling 为什么可能降低 occupancy 但提高速度。
- [ ] 从 AvgPool 的 K/S 推导 input tile 与 reuse。
- [ ] 计算一次 fusion 前后的 global traffic。
- [ ] 区分 CUDA Event、Nsight Systems、Nsight Compute 的用途。
- [ ] 用 Roofline 判断 optimization 方向。
- [ ] 解释 occupancy 不是性能目标。
- [ ] 说明 async memcpy 为何需要 pinned host memory，以及如何确认 overlap。
- [ ] 对任一 v0→v2 给出 hypothesis、metric、result、trade-off。
- [ ] 说明 cuBLAS/CUTLASS/Tensor Core 是 ceiling/进阶路线，不用假装自写 kernel 已达到 library 水平。
