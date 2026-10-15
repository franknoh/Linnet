"""DeepGEMM's FP8 product as a Linnet backend, a plugin from a package of
its own (docs/backends.md): a `std.quant::linear_fp8_blocks` call with
weight blocks of 128 by 128 that the export lets round its input to FP8
(`numerics="fast"`, where it takes `torch._scaled_mm(blocks)`), as
`deep_gemm.fp8_gemm_nt` on Hopper.

The input is rounded as Linnet's own FP8 path rounds it: each 128 inputs of
a row to FP8 with a scale of their own (`linnet.torch.kernels.fp8_groups`),
the rows padded to a multiple of four and the scales column by column, the
layout DeepGEMM reads. The two products differ only in the order they sum.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from typing import Protocol, cast

import torch

from linnet.backends import (
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
from linnet.compiler import PlanError

try:
    from linnet.torch import kernels
except ImportError:  # no Triton: nothing rounds the input
    kernels = None

_CALLEE = "std.quant::linear_fp8_blocks"
# The implementation the export takes where it may round the input to FP8.
_ROUNDS = "torch._scaled_mm(blocks)"
_BLOCK = 128


class _Gemm(Protocol):
    def __call__(
        self,
        a: tuple[torch.Tensor, torch.Tensor],
        b: tuple[torch.Tensor, torch.Tensor],
        d: torch.Tensor,
        *,
        disable_ue8m0_cast: bool = False,
    ) -> None: ...


def _gemm() -> _Gemm | None:
    """DeepGEMM's FP8 product, or None where it is not installed."""
    try:
        module = importlib.import_module("deep_gemm")
    except ImportError:
        return None
    return cast("_Gemm", module.fp8_gemm_nt)


class DeepGemm:
    """Takes the FP8 block-scaled `linear` calls it can run."""

    contract = 1  # the version of `linnet.backends` it was written for

    @property
    def name(self) -> str:
        return "deepgemm"

    @property
    def version(self) -> str:
        try:
            return str(importlib.import_module("deep_gemm").__version__)
        except ImportError:
            return "missing"

    def probe(self, target: Target) -> Capabilities | None:
        # Its SM90 kernels take f32 scales; SM100's take packed UE8M0 ones.
        if target.kind != "cuda" or target.capability[0] != 9:
            return None
        if kernels is None or _gemm() is None:
            return None
        return Capabilities({"deep_gemm": self.version}, frozenset(("bf16",)))

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        for call in graph.semantic_calls(_CALLEE):
            steps = graph.steps[call.start : call.stop]
            if (
                len(call.arguments) != 4
                or call.generics.get("Block") != _BLOCK
                or len(steps) != 1
                or steps[0].attrs.get("implementation") != _ROUNDS
            ):
                continue
            x, weight, scale, bias = call.arguments
            region = Region.of(graph, call.start, call.stop, call.index)
            persistent = set(graph.parameters.values())
            if (
                x.dtype == "bf16"
                and weight.dtype == "f8e4m3"
                and weight.value in persistent
                and scale.value in persistent
                and (bias.value is None or bias.value in persistent)
                and region.inputs == (x.value,)
            ):
                yield region

    def compile(self, region: Region, target: Target, options: Options) -> _BlockProduct:
        if region.call is None:
            raise PlanError("DeepGEMM takes a `linear_fp8_blocks` call")
        gemm = _gemm()
        if gemm is None or kernels is None:
            raise PlanError("DeepGEMM is not installed")
        graph = region.graph
        _, weight, scale, bias = graph.calls[region.call].arguments
        path = {value: path for path, value in graph.parameters.items()}
        return _BlockProduct(
            region,
            gemm,
            path[cast("int", weight.value)],
            path[cast("int", scale.value)],
            path[bias.value] if bias.value is not None else None,
            weight.shape[0],
        )

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        return None


class _BlockProduct:
    def __init__(
        self,
        region: Region,
        gemm: _Gemm,
        weight: str,
        scale: str,
        bias: str | None,
        width: int,
    ) -> None:
        values = region.graph.values
        self._gemm = gemm
        self._weight = weight
        self._scale_path = scale
        self._bias = bias
        self._width = width
        self._scale: torch.Tensor | None = None  # the scales in f32
        held = values[region.graph.parameters[scale]]
        self._meta = ExecutableMeta(
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.inputs),
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.outputs),
            holds=4 * held.shape[0] * held.shape[1],
        )

    @property
    def meta(self) -> ExecutableMeta:
        return self._meta

    def prepare(self, context: Context[torch.Tensor]) -> None:
        scale = context.parameters[self._scale_path].float()
        if self._scale is not None:
            self._scale.copy_(scale)  # again: into the tensor a captured graph reads
            return
        self._scale = scale.contiguous()

    def execute(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        context: Context[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        assert kernels is not None
        if self._scale is None:
            self.prepare(context)
        assert self._scale is not None
        (x,) = inputs
        flat = x.reshape(-1, x.shape[-1])
        rows = flat.shape[0]
        padded = -(-rows // 4) * 4  # the scales' columns 16-byte aligned
        q, scales = kernels.fp8_groups(flat, _BLOCK, padded)
        out = torch.empty((padded, self._width), dtype=x.dtype, device=x.device)
        weight = context.parameters[self._weight].contiguous()
        self._gemm((q, scales), (weight, self._scale), out, disable_ue8m0_cast=True)
        y = out[:rows]
        if self._bias is not None:
            y = y + context.parameters[self._bias]
        return [y.reshape(*x.shape[:-1], self._width)]


__all__ = ["DeepGemm"]
