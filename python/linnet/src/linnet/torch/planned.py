"""A plan of PyTorch tensors replayed as one CUDA graph
(docs/design/execution.md): every backend's executables on PyTorch's stream,
captured together where each is capturable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from ..backends import Context, Plan
from ..compiler import PlanError

if TYPE_CHECKING:
    # The type `torch.cuda.graph_pool_handle` returns.
    from torch.cuda import _POOL_HANDLE  # pyright: ignore[reportPrivateUsage]


class Captured:
    """Runs `plan` with `context`: the first call as it is, on buffers of
    its own; the second captures the plan and replays it; every later one
    copies its inputs into those buffers and replays. Outputs are copies,
    which the next replay leaves alone."""

    def __init__(
        self,
        plan: Plan[torch.Tensor],
        context: Context[torch.Tensor],
        *,
        pool: _POOL_HANDLE | None = None,
        before_capture: Callable[[], None] | None = None,
    ) -> None:
        """`pool` is the CUDA graph memory pool to capture in (a model's, so
        its graphs and this one share it); `before_capture` runs outside the
        capture, just before it (to make this thread's cuBLAS handles, say)."""
        if not all(step.executable.meta.capturable for step in plan.steps):
            raise PlanError("a step of the plan cannot be captured into a CUDA graph")
        self._plan = plan
        self._context = context
        self._pool = pool
        self._before_capture = before_capture
        self._static: list[torch.Tensor] | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._outputs: list[torch.Tensor] = []

    def __call__(self, inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if self._static is None:
            self._static = [value.contiguous().clone() for value in inputs]
            return self._plan.run(self._static, self._context)
        for static, value in zip(self._static, inputs, strict=True):
            static.copy_(value)
        if self._graph is None:
            if self._before_capture is not None:
                self._before_capture()
            graph = torch.cuda.CUDAGraph()
            # Thread-local: another thread's CUDA calls leave the capture be.
            with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
                self._outputs = self._plan.run(self._static, self._context)
            self._graph = graph
        self._graph.replay()
        return [value.clone() for value in self._outputs]
