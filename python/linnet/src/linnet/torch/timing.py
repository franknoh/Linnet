"""How long a call takes on a PyTorch device: the timer
`linnet.backends.Measured` prices regions with (docs/backends.md).

On a CUDA device a call is timed as a captured plan runs it: many calls
recorded in one CUDA graph and replayed, so that launching costs nothing,
where a CUDA graph may record it; otherwise between events. On the CPU, by
the clock. Each is the median of a few trials.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch

from ..backends import ExecutableMeta, Timer

_WARMUP = 3  # calls before timing: the first may compile
_CALLS = 20  # calls timed together, or recorded in one graph
_REPLAYS = 5
_TRIALS = 3


def timer_for(device: torch.device) -> Timer:
    """The timer for calls that run on `device`."""

    def timed(run: Callable[[], object], meta: ExecutableMeta) -> float:
        if device.type != "cuda":
            return _on_cpu(run)
        with torch.cuda.device(device):
            return _on_cuda(run, meta)

    return timed


def _on_cpu(run: Callable[[], object]) -> float:
    for _ in range(_WARMUP):
        run()

    def trial() -> float:
        start = time.perf_counter()
        for _ in range(_CALLS):
            run()
        return (time.perf_counter() - start) / _CALLS

    return statistics.median(trial() for _ in range(_TRIALS))


def _on_cuda(run: Callable[[], object], meta: ExecutableMeta) -> float:
    for _ in range(_WARMUP):
        run()
    torch.cuda.synchronize()
    # Recorded as a captured plan records it, state writes and all: a
    # region timed one way and its neighbours another would not compare.
    graph = _captured(run) if meta.capturable else None
    if graph is not None:
        replay = graph.replay
        return statistics.median(_elapsed(replay, _REPLAYS) for _ in range(_TRIALS)) / _CALLS
    return statistics.median(_elapsed(run, _CALLS) for _ in range(_TRIALS))


def _captured(run: Callable[[], object]) -> torch.cuda.CUDAGraph | None:
    """`_CALLS` calls of `run` recorded in one CUDA graph, or None where
    recording fails."""
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            for _ in range(_CALLS):
                run()
    except RuntimeError:
        torch.cuda.synchronize()
        return None
    return graph


def _elapsed(run: Callable[[], object], calls: int) -> float:
    """Seconds each of `calls` calls of `run` took on the device."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(calls):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000 / calls


__all__ = ["timer_for"]
