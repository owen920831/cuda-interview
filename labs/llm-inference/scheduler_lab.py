#!/usr/bin/env python3
"""Implement static and iteration-level decode scheduling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    timeline: list[list[int | None]]
    finished_at: dict[int, int]


def simulate_static(lengths: list[int], capacity: int) -> Result:
    """A batch cannot admit a replacement until its longest request finishes."""
    # TODO: return one row per decode iteration and None for idle slots.
    raise NotImplementedError


def simulate_continuous(lengths: list[int], capacity: int) -> Result:
    """At an iteration boundary, finished requests are replaced from the queue."""
    # TODO: preserve FIFO admission and emit every slot in every iteration.
    raise NotImplementedError


def idle_slots(result: Result) -> int:
    return sum(item is None for row in result.timeline for item in row)


def run_tests() -> None:
    lengths = [2, 8, 3, 6, 1]
    static = simulate_static(lengths, 2)
    continuous = simulate_continuous(lengths, 2)
    assert set(static.finished_at) == set(range(len(lengths)))
    assert set(continuous.finished_at) == set(range(len(lengths)))
    assert idle_slots(continuous) < idle_slots(static)
    assert len(continuous.timeline) < len(static.timeline)
    print("Scheduler lab PASS")


if __name__ == "__main__":
    run_tests()
