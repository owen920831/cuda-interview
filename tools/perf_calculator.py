#!/usr/bin/env python3
"""Operator FLOP/byte and Roofline calculator with no third-party dependencies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorWork:
    name: str
    flops: int
    direct_bytes: int
    minimum_bytes: int
    shape: str


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def vector_add(args: argparse.Namespace) -> OperatorWork:
    return OperatorWork("vector-add", args.n, 12 * args.n, 12 * args.n, f"N={args.n}")


def reduce_sum(args: argparse.Namespace) -> OperatorWork:
    return OperatorWork(
        "reduce", args.n - 1, 4 * args.n + 4, 4 * args.n + 4, f"N={args.n}"
    )


def gemm(args: argparse.Namespace) -> OperatorWork:
    flops = 2 * args.m * args.n * args.k
    direct = 4 * args.m * args.n * (2 * args.k + 1)
    minimum = 4 * (args.m * args.k + args.k * args.n + args.m * args.n)
    return OperatorWork(
        "gemm", flops, direct, minimum, f"M={args.m}, N={args.n}, K={args.k}"
    )


def avg_pool(args: argparse.Namespace) -> OperatorWork:
    out_h = (args.height - args.kernel) // args.stride + 1
    out_w = (args.width - args.kernel) // args.stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("kernel/stride produce a non-positive output shape")
    input_elements = args.batches * args.channels * args.height * args.width
    output_elements = args.batches * args.channels * out_h * out_w
    flops = output_elements * args.kernel * args.kernel
    direct = 4 * output_elements * (args.kernel * args.kernel + 1)
    minimum = 4 * (input_elements + output_elements)
    shape = (
        f"N={args.batches}, C={args.channels}, H={args.height}, W={args.width}, "
        f"K={args.kernel}, S={args.stride}, OH={out_h}, OW={out_w}"
    )
    return OperatorWork("avg-pool", flops, direct, minimum, shape)


def format_count(value: float) -> str:
    for scale, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= scale:
            return f"{value / scale:.3f} {suffix}"
    return f"{value:.3f}"


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--traffic-model", choices=("direct", "minimum"), default="direct")
    parser.add_argument("--peak-tflops", type=float, help="device peak compute in TFLOP/s")
    parser.add_argument("--bandwidth-gbps", type=float, help="sustained/peak memory bandwidth in GB/s")
    parser.add_argument("--ms", type=float, help="measured kernel latency in milliseconds")
    parser.add_argument("--dram-read-bytes", type=float, default=0.0,
                        help="measured DRAM read bytes from profiler")
    parser.add_argument("--dram-write-bytes", type=float, default=0.0,
                        help="measured DRAM write bytes from profiler")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    add_common(common)
    subparsers = parser.add_subparsers(dest="operator", required=True)

    vector = subparsers.add_parser("vector-add", parents=[common])
    vector.add_argument("--n", type=positive, required=True)
    vector.set_defaults(calculator=vector_add)

    reduce = subparsers.add_parser("reduce", parents=[common])
    reduce.add_argument("--n", type=positive, required=True)
    reduce.set_defaults(calculator=reduce_sum)

    matrix = subparsers.add_parser("gemm", parents=[common])
    matrix.add_argument("--m", type=positive, required=True)
    matrix.add_argument("--n", type=positive, required=True)
    matrix.add_argument("--k", type=positive, required=True)
    matrix.set_defaults(calculator=gemm)

    pool = subparsers.add_parser("avg-pool", parents=[common])
    pool.add_argument("--batches", type=positive, required=True)
    pool.add_argument("--channels", type=positive, required=True)
    pool.add_argument("--height", type=positive, required=True)
    pool.add_argument("--width", type=positive, required=True)
    pool.add_argument("--kernel", type=positive, required=True)
    pool.add_argument("--stride", type=positive, required=True)
    pool.set_defaults(calculator=avg_pool)
    return parser


def report(work: OperatorWork, args: argparse.Namespace) -> None:
    measured_bytes = args.dram_read_bytes + args.dram_write_bytes
    selected_bytes = work.direct_bytes if args.traffic_model == "direct" else work.minimum_bytes
    if measured_bytes > 0:
        selected_bytes = measured_bytes
        selected_label = "measured DRAM"
    else:
        selected_label = args.traffic_model

    direct_ai = work.flops / work.direct_bytes
    minimum_ai = work.flops / work.minimum_bytes
    selected_ai = work.flops / selected_bytes

    print(f"operator:             {work.name}")
    print(f"shape:                {work.shape}")
    print(f"FLOPs:                {format_count(work.flops)}FLOP")
    print(f"direct traffic:       {format_count(work.direct_bytes)}B")
    print(f"minimum traffic:      {format_count(work.minimum_bytes)}B")
    print(f"direct AI:            {direct_ai:.4f} FLOP/B")
    print(f"minimum-traffic AI:   {minimum_ai:.4f} FLOP/B")
    print(f"selected AI ({selected_label}): {selected_ai:.4f} FLOP/B")

    roof_gflops = None
    if args.peak_tflops is not None or args.bandwidth_gbps is not None:
        if args.peak_tflops is None or args.bandwidth_gbps is None:
            raise ValueError("--peak-tflops and --bandwidth-gbps must be supplied together")
        if args.peak_tflops <= 0 or args.bandwidth_gbps <= 0:
            raise ValueError("peak compute and bandwidth must be positive")
        peak_gflops = args.peak_tflops * 1000.0
        ridge = peak_gflops / args.bandwidth_gbps
        bandwidth_roof = args.bandwidth_gbps * selected_ai
        roof_gflops = min(peak_gflops, bandwidth_roof)
        predicted_bound = "memory" if selected_ai < ridge else "compute"
        print(f"ridge point:          {ridge:.4f} FLOP/B")
        print(f"bandwidth roof:       {bandwidth_roof:.3f} GFLOP/s")
        print(f"selected roof:        {roof_gflops:.3f} GFLOP/s ({predicted_bound}-bound region)")

    if args.ms is not None:
        if args.ms <= 0:
            raise ValueError("--ms must be positive")
        achieved_gflops = work.flops / (args.ms * 1e6)
        achieved_gbps = selected_bytes / (args.ms * 1e6)
        print(f"achieved:             {achieved_gflops:.3f} GFLOP/s")
        print(f"effective bandwidth:  {achieved_gbps:.3f} GB/s ({selected_label} bytes)")
        if roof_gflops is not None:
            print(f"roof efficiency:      {100.0 * achieved_gflops / roof_gflops:.2f}%")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        work = args.calculator(args)
        report(work, args)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
