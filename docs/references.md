# 精选权威资料

资料刻意少而精。先完成当天 lab，再只读对应段落；不要把阅读文档当成实作进度。

## 主干

- [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)：thread hierarchy、warps/SIMT、memory hierarchy。
- [Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)：kernel、memory management、shared memory 与同步的入门路径。
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：coalescing、shared-memory matrix multiplication、occupancy 与测量纪律。
- [CUDA C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)：`__syncwarp`、memory fence 与 CUDA-specific syntax。

## 对应本 repo 的优化阶梯

- [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)：shuffle、mask、warp synchronization；对应 reduction。
- [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)：naive → shared tile → `+1` padding 的完整推导。
- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html)：GEMM API 与 column-major layout；对应 library ceiling。
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)：threadblock/warp/thread tile 与 register reuse；Day 20 后再读。

## Profiling

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：scheduler、memory workload、Roofline、source counters。
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)：UI、occupancy calculator 与报告操作。
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)：kernel filter、section/metric query、报告导出。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：CUDA API/workload timeline、streams 与 copy/compute overlap。

## TensorRT 与 LLM inference

- [How TensorRT Works](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html)：build/runtime、engine、execution context、memory 与对象生命期。
- [TensorRT Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-basics.html)：runtime dimensions 与 MIN/OPT/MAX optimization profiles。
- [TensorRT Performance Optimization](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)：batching、CUDA Graph、streams、fusion 与 builder 调优。
- [TensorRT-LLM Architecture](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html)：LLM API、executor、scheduler 与执行路径。
- [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)：blocks、reuse、MQA/GQA、offload 与 cache policy。
- [TensorRT-LLM Paged Attention / IFB / Scheduling](https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html)：continuous batching、token budget 与 chunked prefill。
- [TensorRT-LLM Parallelism](https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html)：TP、PP、DP、EP、CP。
- [TensorRT-LLM Quantization](https://nvidia.github.io/TensorRT-LLM/features/quantization.html)：weight/activation/KV quantization recipes 与版本支持。
- [TensorRT-LLM Benchmarking](https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-benchmarking.html)：离线与在线性能实验。

版本化的 metric 名称与 UI 可能改变；以你安装版本的文档为准。算法原理和 correctness contract 不应绑死到某个 profiler metric 名称。
