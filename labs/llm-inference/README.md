# LLM inference labs

这组 lab 不依赖 GPU，先把 runtime 数学与 scheduler 写对；参考实现分别在 `tools/llm_calculator.py` 与 `tools/batching_simulator.py`。规则：先完成 TODO 和内建测试，再看 reference。

```bash
python3 labs/llm-inference/kv_cache_lab.py
python3 labs/llm-inference/scheduler_lab.py
```

完成后再做 GPU/runtime 实验：

1. 用实际 model config 填 `layers/kv_heads/head_dim/dtype`，和 framework 显存变化比对。
2. 给 scheduler 加 arrival time、prefill tokens 与 chunk size。
3. 输出每个 request 的 TTFT/finish time，并构造 throughput 上升但 P99 变差的 workload。
4. 把 simulator timeline 与一次 TensorRT-LLM/其他 LLM runtime trace 对照，写出抽象遗漏了什么。
