#!/usr/bin/env python3
"""GPU-independent structural checks for the course repository."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md", "COURSE.md", "CMakeLists.txt", "CMakePresets.json",
    "src/01_vector_add.cu", "src/02_reduce.cu", "src/03_transpose.cu",
    "src/04_gemm.cu", "src/05_avg_pool.cu", "src/06_fusion.cu",
    "src/07_streams.cu", "labs/vector_add_lab.cu", "labs/reduce_lab.cu",
    "labs/transpose_lab.cu", "labs/gemm_lab.cu", "labs/avg_pool_lab.cu",
    "docs/mental-model.md", "docs/visual-guide.md", "docs/profiling.md", "progress/template.md",
    "docs/profiling/nsys.md", "docs/profiling/ncu.md",
    "docs/profiling/roofline.md", "docs/profiling/occupancy.md",
    "docs/profiling/source-to-sass.md", "docs/profiling/case-studies.md",
    "progress/profile-template.md", "tools/perf_calculator.py",
    "JOB_ALIGNMENT.md", "docs/llm-inference/README.md",
    "docs/llm-inference/fundamentals.md", "docs/llm-inference/calculations.md",
    "docs/llm-inference/tensorrt.md", "docs/llm-inference/tensorrt-llm.md",
    "docs/llm-inference/serving.md", "docs/llm-inference/optimization.md",
    "docs/llm-inference/profiling.md", "docs/llm-inference/interview-checklist.md",
    "labs/llm-inference/kv_cache_lab.py", "labs/llm-inference/scheduler_lab.py",
    "labs/llm-inference/pytorch_hf_trace.py", "tools/llm_calculator.py",
    "tools/batching_simulator.py", "tools/test_llm_calculator.py",
    "tools/test_batching_simulator.py", "docs/docker.md", "scripts/docker.sh",
]

errors: list[str] = []

for relative in REQUIRED:
    if not (ROOT / relative).is_file():
        errors.append(f"missing required file: {relative}")

course_path = ROOT / "COURSE.md"
if course_path.exists():
    course = course_path.read_text(encoding="utf-8")
    days = [int(value) for value in re.findall(r"^### Day (\d+)\b", course, re.MULTILINE)]
    if days != list(range(1, 22)):
        errors.append(f"COURSE.md must contain Day 1..21 exactly once; got {days}")

source_contracts = {
    "src/01_vector_add.cu": ["vector_add_v0", "vector_add_v1", "vector_add_float4"],
    "src/02_reduce.cu": ["reduce_v0", "reduce_v1", "reduce_v2", "reduce_v3", "reduce_v4"],
    "src/03_transpose.cu": ["transpose_v0", "transpose_tiled<0>", "transpose_tiled<1>"],
    "src/04_gemm.cu": ["gemm_v0", "gemm_v1", "gemm_v2", "cublasSgemm"],
    "src/05_avg_pool.cu": ["avg_pool_v0", "avg_pool_v1", "avg_pool_v2_2x2"],
    "src/06_fusion.cu": ["add_bias", "bias_relu_fused", "bias_relu_float4"],
}

for relative, needles in source_contracts.items():
    path = ROOT / relative
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            errors.append(f"{relative} is missing optimization stage {needle}")

lab_checkpoints = 0
labs_dir = ROOT / "labs"
if labs_dir.exists():
    lab_checkpoints = sum(len(re.findall(r"__global__\s+void\s+student_v\d+", path.read_text(encoding="utf-8")))
                          for path in labs_dir.glob("*.cu"))
if lab_checkpoints < 10:
    errors.append(f"labs should expose at least 10 implementation checkpoints; got {lab_checkpoints}")

llm_lab_todos = 0
for relative in [
    "labs/llm-inference/kv_cache_lab.py",
    "labs/llm-inference/scheduler_lab.py",
    "labs/llm-inference/pytorch_hf_trace.py",
]:
    path = ROOT / relative
    if path.exists():
        llm_lab_todos += path.read_text(encoding="utf-8").count("TODO")
if llm_lab_todos < 8:
    errors.append(f"LLM labs should expose at least 8 implementation checkpoints; got {llm_lab_todos}")

diagram_count = 0
for path in (ROOT / "docs").rglob("*.md"):
    diagram_count += path.read_text(encoding="utf-8").count("```mermaid")
diagram_count += (ROOT / "README.md").read_text(encoding="utf-8").count("```mermaid")
if diagram_count < 15:
    errors.append(f"profiling docs should contain at least 15 Mermaid diagrams; got {diagram_count}")

if errors:
    print("Repository check FAILED:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print("Repository check PASS")
print("  21 daily modules present")
print(f"  {len(source_contracts)} optimization ladders present")
print(f"  {lab_checkpoints} student implementation checkpoints present")
print(f"  {llm_lab_todos} LLM/runtime implementation checkpoints present")
print(f"  {diagram_count} Mermaid learning diagrams present")
