#!/usr/bin/env python3

import argparse
import unittest

from tools import perf_calculator as calculator


class OperatorWorkTest(unittest.TestCase):
    def test_vector_add(self) -> None:
        work = calculator.vector_add(argparse.Namespace(n=100))
        self.assertEqual(work.flops, 100)
        self.assertEqual(work.direct_bytes, 1200)
        self.assertEqual(work.minimum_bytes, 1200)

    def test_gemm_direct_and_minimum_traffic(self) -> None:
        work = calculator.gemm(argparse.Namespace(m=2, n=3, k=4))
        self.assertEqual(work.flops, 48)
        self.assertEqual(work.direct_bytes, 216)
        self.assertEqual(work.minimum_bytes, 104)

    def test_overlapping_pool_has_reuse_gap(self) -> None:
        args = argparse.Namespace(
            batches=1, channels=1, height=4, width=4, kernel=3, stride=1
        )
        work = calculator.avg_pool(args)
        self.assertEqual(work.flops, 36)
        self.assertEqual(work.direct_bytes, 160)
        self.assertEqual(work.minimum_bytes, 80)


if __name__ == "__main__":
    unittest.main()
