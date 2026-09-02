#!/usr/bin/env python3
"""Day 17 scaffold: trace a small Hugging Face causal LM with PyTorch profiler.

Install torch/transformers in a separate environment. Use a tiny, openly
accessible checkpoint selected for your pinned environment; do not make the
course repo depend on a network download.
"""

# TODO 1: load tokenizer/model, record exact model revision and versions.
# TODO 2: create short and long prompts with known token counts.
# TODO 3: profile one prefill and at least eight decode steps separately.
# TODO 4: export a Chrome trace outside git (for example under out/).
# TODO 5: print op names/shapes for linear/GEMM, attention, normalization,
#         elementwise/fusion candidates, and sampling.
# TODO 6: compare use_cache=False/True for correctness, latency, and memory.

raise SystemExit("Complete the TODOs using the pinned PyTorch/Hugging Face environment.")
