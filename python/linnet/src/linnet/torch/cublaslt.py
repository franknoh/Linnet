"""cuBLASLt as a backend for PyTorch tensors (docs/design/execution.md):
every `linear` call without a bias whose weight is a parameter, as a native
product `linnet.runtime` launches on PyTorch's current stream, so a CUDA
graph PyTorch captures records it.
"""

from __future__ import annotations

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
        return "1"

    def probe(self, target: Target) -> Capabilities | None:
        if target.kind != "cuda":
            return None
        try:
            runtime_for(target.index)
        except LinnetError:
            return None
        return Capabilities(dtypes=frozenset(_DTYPES.values()))

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        for call in graph.semantic_calls("std.nn.linear::linear"):
            if len(call.arguments) != 3 or call.arguments[2].value is not None:
                continue  # a bias
            x, weight = call.arguments[0], call.arguments[1]
            region = Region.of(graph, call.start, call.stop, call.index)
            if (
                x.dtype in _DTYPES.values()
                and weight.dtype == x.dtype
                and len(region.parameters) == 1
                and graph.parameters.get(region.parameters[0]) == weight.value
                and region.inputs == (x.value,)
            ):
                yield region

    def compile(self, region: Region, target: Target, options: Options) -> _Product:
        if region.call is None:
            raise PlanError("cuBLASLt takes a `linear` call")
        call = region.graph.calls[region.call]
        x, weight = call.arguments[0], call.arguments[1]
        rows = math.prod(x.shape[:-1])
        n, k = weight.shape
        executable = runtime_for(target.index).cublaslt_linear(rows, n, k, x.dtype)
        return _Product(region, executable, region.parameters[0], (*x.shape[:-1], n))

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        return None


class _Product:
    def __init__(
        self, region: Region, executable: Executable, weight: str, shape: tuple[int, ...]
    ) -> None:
        values = region.graph.values
        self._executable = executable
        self._weight = weight
        self._shape = shape
        self._meta = ExecutableMeta(
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.inputs),
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.outputs),
            host="native",
        )

    @property
    def meta(self) -> ExecutableMeta:
        return self._meta

    def prepare(self, context: Context[torch.Tensor]) -> None:
        return None

    def execute(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        context: Context[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        x = inputs[0].contiguous()
        weight = context.parameters[self._weight].contiguous()
        out = torch.empty(self._shape, dtype=x.dtype, device=x.device)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        self._executable.execute([tensor_ref(x), tensor_ref(weight)], [tensor_ref(out)], stream)
        return [out]
