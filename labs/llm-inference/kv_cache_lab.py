#!/usr/bin/env python3
"""Implement the two TODOs without looking at tools/llm_calculator.py."""


def kv_bytes_per_token(
    layers: int, kv_heads: int, head_dim: int, dtype_bytes: float, shard_factor: int = 1
) -> float:
    # TODO: include K and V, every layer/head/element, and explicit sharding.
    raise NotImplementedError


def max_cached_tokens(kv_budget_bytes: int, bytes_per_token: float) -> int:
    # TODO: reject invalid input and return a whole-token capacity.
    raise NotImplementedError


def run_tests() -> None:
    assert kv_bytes_per_token(32, 8, 128, 2) == 128 * 1024
    assert kv_bytes_per_token(32, 8, 128, 2, 4) == 32 * 1024
    assert max_cached_tokens(4 * 1024**3, 128 * 1024) == 32768
    print("KV cache lab PASS")


if __name__ == "__main__":
    run_tests()
