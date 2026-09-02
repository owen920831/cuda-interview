# TensorRT / TensorRT-LLM / LLM inference 面试验收

每题先闭卷说 60–120 秒；再看对应章节补洞。通过标准不是说出关键字，而是能画图、写公式、给 trade-off 和验证方法。

## A. LLM execution

- [ ] 画出 request 的 queue→prefill→decode→sampling→stream 路径。
- [ ] 为什么 prefill 和 decode 的算术强度/平行度不同？何时常见结论会失效？
- [ ] TTFT、TPOT/ITL、E2E、output tok/s、request/s、goodput 各是什么？
- [ ] 为什么 E2E 近似是 `TTFT + (O-1)×TPOT`？
- [ ] MHA、GQA、MQA 如何改变 KV memory 与 attention kernel？
- [ ] 为什么“GPU utilization 95%”不能证明服务已经最佳？

## B. KV cache 与 paged attention

- [ ] 从 model config 推导 bytes/token，并手算 4K context、batch 8。
- [ ] 权重能 fit 为什么仍可能在并发下 OOM？
- [ ] contiguous KV allocation 有哪些 fragmentation/扩容问题？
- [ ] paged KV 的 page table、block size、尾块浪费如何权衡？
- [ ] prefix reuse 何时命中？多租户为什么要隔离/salt？
- [ ] host offload 是容量优化还是无条件 latency 优化？
- [ ] GQA/MQA + TP 时为什么 KV memory 不一定除以 TP degree？

## C. Scheduler 与 serving

- [ ] 用一个 output length 不同的例子比较 static 与 continuous batching。
- [ ] `max batch size`、`max sequence length`、iteration token budget 的角色有何不同？
- [ ] chunked prefill 如何影响 TTFT、TPOT、kernel 效率与 KV budget？
- [ ] queueing、head-of-line blocking、priority starvation 怎么出现？
- [ ] preemption 时 recompute 与 swap/offload 如何选？
- [ ] speculative decoding 的 draft、verify、acceptance rate 与适用 batch？
- [ ] prefill/decode disaggregation 的收益为何必须覆盖 KV transfer？
- [ ] open-loop request rate 和 closed-loop concurrency benchmark 有何不同？

## D. TensorRT

- [ ] 画 builder/network/config/parser/engine/runtime/context 的两阶段与生命期。
- [ ] tactic 是什么？workspace limit 与 timing cache 如何影响 build？
- [ ] engine 为什么不能被当成任意环境通用的 ONNX？
- [ ] dynamic dimension 与 optimization profile 有何不同？MIN/OPT/MAX 分别做什么？
- [ ] 为什么 MAX 设很大可能变慢或占更多 memory？
- [ ] 一个 engine 如何安全地并发执行？context/profile/stream/buffer 如何对应？
- [ ] 什么情形需要 plugin？plugin 的 format、serialization、version 有哪些风险？
- [ ] CUDA Graph 解决什么瓶颈？dynamic shape 下怎样验证 capture/fallback？
- [ ] 如何用 `trtexec` 区分 GPU compute、enqueue、H2D/D2H 与 E2E？

## E. TensorRT-LLM

- [ ] TensorRT、TensorRT-LLM、Triton Inference Server 各在哪一层？
- [ ] 旧 engine-oriented workflow 与当前 high-level LLM API 怎么对应？
- [ ] Executor/scheduler 每个 iteration 做哪些事？
- [ ] packed input 为什么对 IFB 重要？padding 会浪费什么？
- [ ] KV block reuse、eviction、offload 需要哪些 telemetry？
- [ ] overlap scheduler 如何隐藏 CPU work？代价/限制是什么？
- [ ] 为什么不能照抄另一版本的 CLI/default config？如何保证可复现？

## F. Quantization 与 multi-GPU

- [ ] W4A16、W4A8、FP8 与 KV FP8 分别量化什么？
- [ ] 为什么权重缩到 1/4 不代表端到端 latency 也缩到 1/4？
- [ ] 量化验收为何同时需要 quality、memory、TTFT、TPOT、P99？
- [ ] TP、PP、DP、EP、CP 分别切什么，主要 collective/idle 是什么？
- [ ] 模型放不下时，如何在 quantization、TP、PP 之间形成实验方案？
- [ ] 为什么跨 node TP 常被网络拓扑限制？你会在 NSYS 看什么？
- [ ] MoE 的 expert parallelism 为什么会有 all-to-all 与负载不均？

## G. 白板与实作 gate

- [ ] 10 分钟完成一题 KV capacity，不看计算器，误差 < 2%。
- [ ] 30 分钟完成 `kv_cache_lab.py`，所有 unit tests 通过。
- [ ] 45 分钟完成 `scheduler_lab.py` 的 static + continuous 两版。
- [ ] 能从一张 NSYS timeline 指出 queue、CPU gap、prefill/decode、NCCL critical path。
- [ ] 能从一次 A/B 结果明确说“证据不支持假设”，而不是硬编优化故事。
- [ ] 设计一个 shape×concurrency×precision benchmark matrix，并写出控制变量。

## 最后六题系统设计

1. 24 GiB GPU 部署 7B FP16，如何估 4K context 的最大并发？哪些 overhead 不能漏？
2. 吞吐上涨但 P99 TTFT 爆掉，你会从哪四层定位？
3. 长 prompt 干扰 chat decode，要调 chunked prefill、priority 还是 P/D disaggregation？如何 A/B？
4. 4 GPU 上 TP=4 比 TP=2+DP=2 慢，列出至少四个可测原因。
5. 打开 FP8 后 memory 降了但 TPOT 不变，你会检查哪些 kernel、shape 与非权重瓶颈？
6. 共享 system prompt 命中 prefix cache 后，如何确保 tenant isolation、correctness 与真实收益？
