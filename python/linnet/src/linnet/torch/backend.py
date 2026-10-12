"""The generated PyTorch source as a backend (docs/design/execution.md): the
fallback that runs every region no other backend takes. A whole graph runs
as one call of the model's entry; a smaller region as the source
`linnet graph --slices` prints for it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from .compiled import CompiledLinnetModule


def target_of(device: torch.device) -> Target:
    """`device` as a plan's target."""
    index = cast("int | None", device.index)
    if device.type != "cuda":
        return Target(device.type, index or 0)
    if index is None:
        index = torch.cuda.current_device()
    return Target(
        "cuda",
        index,
        torch.cuda.get_device_name(index),
        torch.cuda.get_device_capability(index),
        torch.cuda.mem_get_info(index)[1],
    )


class TorchSource:
    """Runs entry `entry` of `module`, with `generics` and `compile` as
    `run_entry` takes them."""

    def __init__(
        self,
        module: CompiledLinnetModule,
        entry: str,
        bindings: Mapping[str, str],
        generics: Mapping[str, int | str],
        compile: bool | str | None,
    ) -> None:
        self._module = module
        self._entry = entry
        self._bindings = dict(bindings)
        self._generics = dict(generics)
        self._compile = compile

    @property
    def name(self) -> str:
        return "torch-source"

    @property
    def version(self) -> str:
        return torch.__version__

    def probe(self, target: Target) -> Capabilities | None:
        if target.kind != self._module.interpreter.device.type:
            return None
        return Capabilities({"torch": torch.__version__})

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        return [Region.of(graph, 0, len(graph.steps))]

    def compile(self, region: Region, target: Target, options: Options) -> _Entry | _Slice:
        if region.whole:
            return _Entry(self, region)
        (function,) = self._module.slices(self._entry, self._bindings, [region])
        return _Slice(region, function)

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        return None

    def run(self, inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        result: object = self._module.run_entry(
            self._entry, list(inputs), self._generics, compile=self._compile
        )
        values = cast("tuple[object, ...]", result) if isinstance(result, tuple) else (result,)
        return [cast("torch.Tensor", value) for value in values]


class _Entry:
    """The whole graph: one call of the entry, which reads its parameters
    and states from the module and writes the states back there."""

    def __init__(self, backend: TorchSource, region: Region) -> None:
        values = region.graph.values

        def spec(value: int) -> TensorSpec:
            return TensorSpec(values[value].shape, values[value].dtype)

        self._backend = backend
        self._meta = ExecutableMeta(
            tuple(spec(v) for v in region.inputs),
            tuple(spec(v) for v in region.outputs),
            capturable=False,  # it captures CUDA graphs itself, when compiled to
            effects="state",
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
        return self._backend.run(inputs)


def _specs(region: Region) -> tuple[tuple[TensorSpec, ...], tuple[TensorSpec, ...]]:
    values = region.graph.values

    def spec(value: int) -> TensorSpec:
        return TensorSpec(values[value].shape, values[value].dtype)

    return tuple(spec(v) for v in region.inputs), tuple(spec(v) for v in region.outputs)


class _Slice:
    """A region as a module of its own, over the model's parameters and
    states by path."""

    def __init__(
        self,
        region: Region,
        function: Callable[[Sequence[torch.Tensor]], list[torch.Tensor]],
    ) -> None:
        self._function = function
        inputs, outputs = _specs(region)
        self._meta = ExecutableMeta(inputs, outputs, effects="state" if region.states else "pure")

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
        return self._function(inputs)
