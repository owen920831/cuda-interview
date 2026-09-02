import unittest

from tools.llm_calculator import e2e_latency_ms, kv_bytes_per_token, weight_bytes


class LlmCalculatorTest(unittest.TestCase):
    def test_kv_example_is_128_kib_per_token(self) -> None:
        result = kv_bytes_per_token(32, 8, 128, 2)
        self.assertEqual(result, 128 * 1024)

    def test_kv_explicit_sharding(self) -> None:
        unsharded = kv_bytes_per_token(32, 8, 128, 2)
        sharded = kv_bytes_per_token(32, 8, 128, 2, shard_factor=4)
        self.assertEqual(sharded, unsharded / 4)

    def test_weight_memory_uses_decimal_parameter_count(self) -> None:
        self.assertEqual(weight_bytes(7, 16), 14_000_000_000)

    def test_e2e_counts_inter_token_intervals(self) -> None:
        self.assertEqual(e2e_latency_ms(120, 18, 128), 2406)
        self.assertEqual(e2e_latency_ms(120, 18, 1), 120)
        self.assertEqual(e2e_latency_ms(120, 18, 0), 0)


if __name__ == "__main__":
    unittest.main()
