#!/usr/bin/env python3
"""Small decode-only simulator for static and continuous batching lessons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Simulation:
    timeline: list[list[int | None]]
    finished_at: dict[int, int]

    @property
    def idle_slots(self) -> int:
        return sum(value is None for step in self.timeline for value in step)


def validate(lengths: list[int], capacity: int) -> None:
    if capacity <= 0 or any(length <= 0 for length in lengths):
        raise ValueError("capacity and all output lengths must be positive")


def simulate_static(lengths: list[int], capacity: int) -> Simulation:
    validate(lengths, capacity)
    timeline: list[list[int | None]] = []
    finished: dict[int, int] = {}
    for start in range(0, len(lengths), capacity):
        batch = list(range(start, min(start + capacity, len(lengths))))
        duration = max(lengths[index] for index in batch)
        for local_step in range(1, duration + 1):
            row: list[int | None] = []
            for index in batch:
                if local_step <= lengths[index]:
                    row.append(index)
                    if local_step == lengths[index]:
                        finished[index] = len(timeline) + 1
                else:
                    row.append(None)
            row.extend([None] * (capacity - len(row)))
            timeline.append(row)
    return Simulation(timeline, finished)


def simulate_continuous(lengths: list[int], capacity: int) -> Simulation:
    validate(lengths, capacity)
    waiting = list(range(len(lengths)))
    active: list[list[int]] = []  # [request index, generated count]
    timeline: list[list[int | None]] = []
    finished: dict[int, int] = {}

    while waiting or active:
        while waiting and len(active) < capacity:
            active.append([waiting.pop(0), 0])
        row: list[int | None] = [item[0] for item in active]
        row.extend([None] * (capacity - len(row)))
        timeline.append(row)
        survivors: list[list[int]] = []
        for item in active:
            item[1] += 1
            if item[1] == lengths[item[0]]:
                finished[item[0]] = len(timeline)
            else:
                survivors.append(item)
        active = survivors
    return Simulation(timeline, finished)


def print_simulation(name: str, result: Simulation) -> None:
    print(f"{name}:")
    for step, row in enumerate(result.timeline, 1):
        slots = " ".join("-" if item is None else f"R{item}" for item in row)
        print(f"  iteration {step:02d}: {slots}")
    print(f"  iterations={len(result.timeline)}, idle_slots={result.idle_slots}")
    print(f"  finished_at={result.finished_at}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    try:
        print_simulation("static", simulate_static(args.lengths, args.batch))
        print_simulation("continuous", simulate_continuous(args.lengths, args.batch))
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
