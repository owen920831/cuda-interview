import unittest

from tools.batching_simulator import simulate_continuous, simulate_static


class BatchingSimulatorTest(unittest.TestCase):
    def test_continuous_reduces_idle_slots_and_makespan(self) -> None:
        lengths = [2, 8, 3, 6, 1]
        static = simulate_static(lengths, 2)
        continuous = simulate_continuous(lengths, 2)
        self.assertLess(continuous.idle_slots, static.idle_slots)
        self.assertLess(len(continuous.timeline), len(static.timeline))

    def test_all_requests_finish(self) -> None:
        result = simulate_continuous([1, 2, 3], 2)
        self.assertEqual(set(result.finished_at), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
