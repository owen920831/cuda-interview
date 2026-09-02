#!/usr/bin/env python3
"""Dependency-free LLM inference capacity and latency calculator."""

from __future__ import annotations

import argparse
import math

GIB = 1024 ** 3


def kv_bytes_per_token(
    layers: int,
    kv_heads: int,
    head_dim: int,
    dtype_bytes: float,
    shard_factor: int = 1,
) -> float:
    values = (layers, kv_heads, head_dim, shard_factor)
    if any(value <= 0 for value in values) or dtype_bytes <= 0:
        raise ValueError("KV dimensions, dtype bytes, and shard factor must be positive")
    return 2.0 * layers * kv_heads * head_dim * dtype_bytes / shard_factor


def weight_bytes(params_billions: float, bits: float) -> float:
    if params_billions <= 0 or bits <= 0:
        raise ValueError("parameter count and weight bits must be positive")
    return params_billions * 1e9 * bits / 8.0


def e2e_latency_ms(ttft_ms: float, tpot_ms: float, output_tokens: int) -> float:
    if ttft_ms < 0 or tpot_ms < 0 or output_tokens < 0:
        raise ValueError("latencies and output token count must be non-negative")
    if output_tokens == 0:
        return 0.0
    return ttft_ms + (output_tokens - 1) * tpot_ms


def add_kv_arguments(parser: argparse.ArgumentParser, include_workload: bool) -> None:
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--dtype-bytes", type=float, required=True)
    parser.add_argument(
        "--shard-factor", type=int, default=1,
        help="Actual KV sharding factor, not automatically the TP degree.",
    )
    if include_workload:
        parser.add_argument("--sequence", type=int, required=True)
        parser.add_argument("--batch", type=int, default=1)


def command_kv(args: argparse.Namespace) -> None:
    per_token = kv_bytes_per_token(
        args.layers, args.kv_heads, args.head_dim, args.dtype_bytes, args.shard_factor
    )
    total_tokens = args.sequence * args.batch
    total = per_token * total_tokens
    print(f"KV bytes/token/sequence/rank: {per_token:.0f}")
    print(f"Total cached tokens: {total_tokens}")
    print(f"Total KV per rank: {total / GIB:.4f} GiB")
    if args.shard_factor > 1:
        print("Assumption: KV heads are actually sharded by the supplied factor.")


def command_capacity(args: argparse.Namespace) -> None:
    if not 0 < args.memory_utilization <= 1:
        raise ValueError("memory utilization must be in (0, 1]")
    if args.gpu_gib <= 0 or args.overhead_gib < 0 or args.tokens_per_request <= 0:
        raise ValueError("GPU size/tokens must be positive and overhead non-negative")

    usable = args.gpu_gib * GIB * args.memory_utilization
    weights = weight_bytes(args.params_billions, args.weight_bits) / args.weight_shard_factor
    overhead = args.overhead_gib * GIB
    budget = usable - weights - overhead
    per_token = kv_bytes_per_token(
        args.layers, args.kv_heads, args.head_dim, args.dtype_bytes, args.shard_factor
    )
    max_tokens = max(math.floor(budget / per_token), 0)
    requests = max_tokens // args.tokens_per_request

    print(f"Usable memory per rank: {usable / GIB:.4f} GiB")
    print(f"Approximate weights per rank: {weights / GIB:.4f} GiB")
    print(f"Reserved runtime/workspace: {overhead / GIB:.4f} GiB")
    print(f"KV budget per rank: {budget / GIB:.4f} GiB")
    print(f"KV bytes/token/sequence/rank: {per_token:.0f}")
    print(f"Maximum total cached tokens: {max_tokens}")
    print(f"Rough active-request cap: {requests}")
    if budget < 0:
        print("WARNING: weights plus overhead exceed the usable memory budget.")
    print("This is a capacity upper bound, not the performance-optimal batch size.")


def command_latency(args: argparse.Namespace) -> None:
    total = e2e_latency_ms(args.ttft_ms, args.tpot_ms, args.output_tokens)
    rate = args.output_tokens / (total / 1000.0) if total > 0 else 0.0
    print(f"Approximate E2E latency: {total:.3f} ms")
    print(f"Single-request average output rate: {rate:.3f} token/s")
    print("Server throughput needs a concurrent workload and cannot be inferred from this rate.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    kv = subparsers.add_parser("kv", help="Calculate KV cache memory")
    add_kv_arguments(kv, include_workload=True)
    kv.set_defaults(func=command_kv)

    capacity = subparsers.add_parser("capacity", help="Estimate KV and request capacity")
    capacity.add_argument("--gpu-gib", type=float, required=True)
    capacity.add_argument("--memory-utilization", type=float, default=0.9)
    capacity.add_argument("--params-billions", type=float, required=True)
    capacity.add_argument("--weight-bits", type=float, required=True)
    capacity.add_argument("--weight-shard-factor", type=int, default=1)
    capacity.add_argument("--overhead-gib", type=float, default=2.0)
    capacity.add_argument("--tokens-per-request", type=int, required=True)
    add_kv_arguments(capacity, include_workload=False)
    capacity.set_defaults(func=command_capacity)

    latency = subparsers.add_parser("latency", help="Relate TTFT, TPOT, and E2E")
    latency.add_argument("--ttft-ms", type=float, required=True)
    latency.add_argument("--tpot-ms", type=float, required=True)
    latency.add_argument("--output-tokens", type=int, required=True)
    latency.set_defaults(func=command_latency)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
