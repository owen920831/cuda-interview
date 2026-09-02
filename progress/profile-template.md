# Profiling 实验报告

## 0. Reproducibility

```text
Date:
Git commit / working tree:
GPU:
Driver:
CUDA / nvcc:
Nsight Systems / Compute version:
Build type and flags:
Power / clock conditions:
```

## 1. Workload contract

```text
Operator/version:
Shape/layout/dtype:
Grid/block:
Warm-up/iterations:
Correctness reference:
Max abs/relative error:
```

## 2. Hand calculation before profiler

```text
FLOPs:
Direct/source-requested bytes:
Minimum compulsory bytes:
Estimated AI:
Device compute/bandwidth ceilings:
Ridge point:
Predicted region and bottleneck:
```

## 3. CUDA Event baseline

| Version | Median/mean latency | GB/s | GFLOP/s | Notes |
|---|---:|---:|---:|---|
| v0 | | | | |
| candidate | | | | |

## 4. Nsight Systems

- Stable range selected:
- CPU gaps / launch overhead:
- H2D, kernel, D2H share:
- Synchronization points:
- Streams/overlap evidence:
- Screenshot/report path:

## 5. Nsight Compute

| Section | Baseline | Candidate | Interpretation |
|---|---:|---:|---|
| Duration | | | |
| SM SOL | | | |
| Memory/DRAM SOL | | | |
| Roofline AI / achieved | | | |
| Registers/thread | | | |
| Static/dynamic shared | | | |
| Theoretical/achieved occupancy | | | |
| Eligible/issued warps | | | |
| Global sectors/requests | | | |
| Shared bank conflicts | | | |
| Local load/store (spill) | | | |
| Primary stall reason | | | |

## 6. Source → PTX → SASS

- Hot source line:
- Expected generated instruction:
- Actual PTX/SASS evidence:
- Alignment/aliasing/tail condition:
- Resource usage change:

## 7. Conclusion

```text
Hypothesis:
Evidence supporting/refuting it:
Why latency changed:
Trade-off introduced:
What this result does NOT prove:
Next single-variable experiment:
Stop/continue decision:
```
