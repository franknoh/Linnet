"""cuBLASLt as a backend for PyTorch tensors (docs/design/execution.md):
every `linear` call without a bias whose weight is a parameter, as a native
product `linnet.runtime` launches on PyTorch's current stream, so a CUDA
graph PyTorch captures records it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence

import torch

from ..backends import (
    Capabilities,
    Context,
    Cost,
    ExecutableMeta,
    Graph,
    Options,
    Region,
    Target,
    TensorSpec,
)
from ..compiler import LinnetError, PlanError
from ..runtime import Executable, Runtime, TensorRef

_DTYPES = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}
_BYTES = {"f32": 4, "f16": 2, "bf16": 2}
_runtimes: dict[int, Runtime] = {}


def runtime_for(index: int) -> Runtime:
    """The runtime of CUDA device `index`, made once."""
    if index not in _runtimes:
        _runtimes[index] = Runtime("cuda", index)
    return _runtimes[index]


def tensor_ref(tensor: torch.Tensor) -> TensorRef:
    """`tensor` as the runtime sees it, without a copy."""
    strides = None if tensor.is_contiguous() else tuple(tensor.stride())
    kind = "cuda" if tensor.is_cuda else "cpu"
    return TensorRef(
        tensor.data_ptr(),
        tuple(tensor.shape),
        _DTYPES[tensor.dtype],
        (kind, tensor.device.index or 0),
        strides,
    )


class CublasLt:
    """Takes the `linear` calls it can run."""

    @property
    def name(self) -> str:
        return "cublaslt"

    @property
    def version(self) -> str:
        return "2"

    def probe(self, target: Target) -> Capabilities | None:
        if target.kind != "cuda":
            return None
        try:
            runtime_for(target.index)
        except LinnetError:
            return None
        return Capabilities(dtypes=frozenset(_DTYPES.values()))

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        """Each `linear` call it can run, and each set of them that read the
        same input (q, k and v; gate and up), which it joins into one."""
        siblings: dict[int, list[int]] = {}
        for call in graph.semantic_calls("std.nn.linear::linear"):
            if not _runs(graph, call.index):
                continue
            yield Region.of(graph, call.start, call.stop, call.index)
            x = call.arguments[0].value
            if call.stop - call.start == 1 and x is not None:
                siblings.setdefault(x, []).append(call.index)
        for calls in siblings.values():
            if len(calls) > 1:
                yield Region.of_steps(graph, [graph.calls[c].start for c in calls])

    def compile(self, region: Region, target: Target, options: Options) -> _Product:
        graph = region.graph
        calls = [region.call] if region.call is not None else _calls_of(region)
        x = graph.calls[calls[0]].arguments[0]
        weights = [graph.calls[c].arguments[1] for c in calls]
        rows = math.prod(x.shape[:-1])
        k = x.shape[-1]
        widths = [weight.shape[0] for weight in weights]
        executable = runtime_for(target.index).cublaslt_linear(rows, sum(widths), k, x.dtype)
        paths = [_parameter(graph, weight.value) for weight in weights]
        made = [graph.calls[c].results[0].value for c in calls]
        return _Product(region, executable, paths, widths, x.shape[:-1], made)

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        return None


def _parameter(graph: Graph, value: int | None) -> str:
    for path, made in graph.parameters.items():
        if made == value:
            return path
    raise PlanError("a `linear` call whose weight is not a parameter")


def _runs(graph: Graph, index: int) -> bool:
    """Whether cuBLASLt runs `linear` call `index`: no bias, a parameter for
    a weight, one of its dtypes throughout."""
    call = graph.calls[index]
    if len(call.arguments) != 3 or call.arguments[2].value is not None:
        return False
    x, weight = call.arguments[0], call.arguments[1]
    region = Region.of(graph, call.start, call.stop, call.index)
    return (
        x.dtype in _DTYPES.values()
        and weight.dtype == x.dtype
        and len(region.parameters) == 1
        and graph.parameters.get(region.parameters[0]) == weight.value
        and region.inputs == (x.value,)
    )


def _calls_of(region: Region) -> list[int]:
    """The `linear` calls whose steps a joined region holds, in order."""
    found: list[int] = []
    for step in region.steps:
        call = next(
            c
            for c in reversed(step.calls)
            if region.graph.calls[c].callee == "std.nn.linear::linear"
        )
        found.append(call)
    return found


class _Product:
    """One product by cuBLASLt of the input by the weights of one or more
    `linear` calls, joined (made once, on the first call), its output split
    back into each call's."""

    def __init__(
        self,
        region: Region,
        executable: Executable,
        weights: list[str],
        widths: list[int],
        rows: tuple[int, ...],
        made: list[int | None],
    ) -> None:
        values = region.graph.values
        self._executable = executable
        self._paths = weights
        self._widths = widths
        self._shape = (*rows, sum(widths))
        # Which part of the output each of the region's outputs is.
        self._parts = [made.index(value) for value in region.outputs]
        self._joined: torch.Tensor | None = None
        self._copied = len(weights) > 1
        # Joined weights are a copy of the model's, unless the context lays
        # them out side by side (`Context.join`).
        k = region.graph.values[region.inputs[0]].shape[-1]
        size = _BYTES[region.graph.values[region.inputs[0]].dtype]
        self._meta = ExecutableMeta(
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.inputs),
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.outputs),
            host="native",
            holds=sum(widths) * k * size if len(weights) > 1 else 0,
        )

    @property
    def meta(self) -> ExecutableMeta:
        return self._meta if self._copied else dataclasses.replace(self._meta, holds=0)

    def prepare(self, context: Context[torch.Tensor]) -> None:
        if len(self._paths) <= 1:
            return
        if self._joined is not None:
            # Again, after the weights changed: a copy is written over, in
            # place (a captured graph reads it); a joined view reads them.
            if self._copied:
                self._joined.copy_(torch.cat([context.parameters[p] for p in self._paths]))
            return
        joined = context.join(self._paths) if context.join is not None else None
        self._copied = joined is None
        if joined is None:
            joined = torch.cat([context.parameters[p] for p in self._paths]).contiguous()
        self._joined = joined

    def execute(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        context: Context[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        if len(self._paths) == 1:
            weight = context.parameters[self._paths[0]].contiguous()
        else:
            if self._joined is None:
                self.prepare(context)
            assert self._joined is not None
            weight = self._joined
        x = inputs[0].contiguous()
        out = torch.empty(self._shape, dtype=x.dtype, device=x.device)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        self._executable.execute([tensor_ref(x), tensor_ref(weight)], [tensor_ref(out)], stream)
        if len(self._widths) == 1:
            return [out]
        parts = torch.split(out, self._widths, dim=-1)
        return [parts[i] for i in self._parts]
