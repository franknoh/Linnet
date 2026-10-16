"""vLLM's Marlin int4 product as a Linnet backend, a plugin from a package of
its own (docs/backends.md): a `std.quant::linear_int4_groups` call (an
`Int4GroupLinear`, the layer GPTQ and AWQ checkpoints import into) as
`gptq_marlin_gemm`, which multiplies straight from the 4-bit weights at
any number of rows.

Marlin reads a weight in tiles of its own. `prepare` lays the model's codes
out as a GPTQ checkpoint holds them (eight inputs to an `int32`, the first in
the low bits: Linnet's bytes, four at a time) and has vLLM repack those; the
scales and zero points it permutes as vLLM does for its own GPTQ and AWQ
layers. It keeps that copy, half a byte a weight. A weight is `(q - zero) *
scale` either way, so a GPTQ layer's zero of 8 and an AWQ layer's zero points
take the same path.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from types import ModuleType
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

_CALLEE = "std.quant::linear_int4_groups"
_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}
# Marlin's tiles: an output count a multiple of 64, an input count of 128,
# and groups of these sizes (or one group a row).
_GROUPS = (32, 64, 128)


class _Product(Protocol):
    def __call__(
        self,
        a: torch.Tensor,
        c: torch.Tensor | None,
        b_q_weight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        global_scale: torch.Tensor | None,
        b_zeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        b_q_type: object,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool = True,
        use_atomic_add: bool = False,
        use_fp32_reduce: bool = False,
        is_zp_float: bool = False,
    ) -> torch.Tensor: ...


class _Repack(Protocol):
    def __call__(
        self, b_q_weight: torch.Tensor, perm: torch.Tensor, size_k: int, size_n: int, num_bits: int
    ) -> torch.Tensor: ...


def _ops() -> ModuleType | None:
    try:
        return importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None


class Marlin:
    """Takes the group-wise int4 `linear` calls Marlin can run."""

    contract = 1  # the version of `linnet.backends` it was written for

    @property
    def name(self) -> str:
        return "marlin"

    @property
    def version(self) -> str:
        try:
            return str(importlib.import_module("vllm").__version__)
        except ImportError:
            return "missing"

    def probe(self, target: Target) -> Capabilities | None:
        if target.kind != "cuda" or target.capability < (8, 0) or _ops() is None:
            return None
        return Capabilities({"vllm": self.version}, frozenset(_DTYPES))

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        persistent = set(graph.parameters.values())
        for call in graph.semantic_calls(_CALLEE):
            if len(call.arguments) != 5:
                continue
            x, q, scale, zero, bias = call.arguments
            out_features, groups, group = q.shape
            region = Region.of(graph, call.start, call.stop, call.index)
            if (
                x.dtype in _DTYPES
                and scale.dtype == x.dtype
                and out_features % 64 == 0
                and (groups * group) % 128 == 0
                and (group in _GROUPS or groups == 1)
                and all(v.value in persistent for v in (q, scale, zero))
                and (bias.value is None or bias.value in persistent)
                and region.inputs == (x.value,)
            ):
                yield region

    def compile(self, region: Region, target: Target, options: Options) -> _Int4Product:
        module = _ops()
        if region.call is None or module is None:
            raise PlanError("marlin takes a `linear_int4_groups` call, and needs vLLM")
        graph = region.graph
        x, q, scale, zero, bias = graph.calls[region.call].arguments
        path = {value: path for path, value in graph.parameters.items()}
        out_features, groups, group = q.shape
        return _Int4Product(
            region,
            module,
            (
                path[cast("int", q.value)],
                path[cast("int", scale.value)],
                path[cast("int", zero.value)],
            ),
            path[bias.value] if bias.value is not None else None,
            (out_features, groups, group),
            _DTYPES[x.dtype],
        )

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        return None


def _scale_order(groups: int) -> torch.Tensor:
    """The order Marlin reads a row of scales (or zero points) in: 64 at a
    time, eight by eight, or by its own pattern with one group a row."""
    if groups > 1:
        return torch.tensor([i + 8 * j for i in range(8) for j in range(8)])
    pattern = [0, 1, 8, 9, 16, 17, 24, 25]
    return torch.tensor([2 * i + j for i in range(4) for j in pattern])


def _marlin_scales(scale: torch.Tensor) -> torch.Tensor:
    """`[Out, Groups]` scales as Marlin reads them: `[Groups, Out]`, each 64
    (or 32) in its order."""
    by_group = scale.t()
    order = _scale_order(by_group.shape[0]).to(scale.device)
    return by_group.reshape(-1, order.numel())[:, order].reshape(by_group.shape).contiguous()


def _marlin_zeros(zero: torch.Tensor) -> torch.Tensor:
    """`[Out, Groups]` zero points as Marlin reads them: in the scales' order
    (the many-group one), each eight interleaved and packed into an
    `int32`, the first in the low bits."""
    by_group = zero.t().to(torch.int32)
    groups, out_features = by_group.shape
    order = _scale_order(2).to(zero.device)
    permuted = by_group.reshape(-1, order.numel())[:, order]
    interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=zero.device)
    eights = permuted.reshape(-1, 8)[:, interleave].reshape(groups, out_features // 8, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=zero.device)
    return (eights << shifts).sum(-1, dtype=torch.int32).contiguous()


def _gptq_codes(packed: torch.Tensor, out_features: int) -> torch.Tensor:
    """Linnet's codes (`[Out, Groups, Group / 2]` bytes, the first of each
    pair low) as GPTQ packs them: `[In / 8, Out]` `int32`, eight inputs a
    word, the first in the low four bits. Four little-endian bytes are that
    word already."""
    return packed.reshape(out_features, -1).contiguous().view(torch.int32).t().contiguous()


class _Int4Product:
    def __init__(
        self,
        region: Region,
        ops: ModuleType,
        paths: tuple[str, str, str],
        bias: str | None,
        shape: tuple[int, int, int],
        dtype: torch.dtype,
    ) -> None:
        values = region.graph.values
        self._product = cast("_Product", ops.gptq_marlin_gemm)
        self._repack = cast("_Repack", ops.gptq_marlin_repack)
        self._type = importlib.import_module("vllm.scalar_type").scalar_types.uint4
        self._paths = paths
        self._bias = bias
        self._out, groups, group = shape
        self._in = groups * group
        self._dtype = dtype
        self._weight: torch.Tensor | None = None
        self._scales: torch.Tensor | None = None
        self._zeros: torch.Tensor | None = None
        self._workspace: torch.Tensor | None = None
        self._empty: torch.Tensor | None = None
        # The repacked codes, the scales, and the zero points packed eight
        # to a word.
        per_group = groups * self._out
        held = (self._out * self._in + per_group) // 2 + per_group * dtype.itemsize
        self._meta = ExecutableMeta(
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.inputs),
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.outputs),
            holds=held,
        )

    @property
    def meta(self) -> ExecutableMeta:
        return self._meta

    def prepare(self, context: Context[torch.Tensor]) -> None:
        codes, scale, zero = (context.parameters[path] for path in self._paths)
        empty = torch.empty(0, dtype=torch.int32, device=codes.device)
        weight = self._repack(_gptq_codes(codes, self._out), empty, self._in, self._out, 4)
        scales = _marlin_scales(scale.to(self._dtype))
        zeros = _marlin_zeros(zero)
        if self._weight is None or self._scales is None or self._zeros is None:
            self._weight, self._scales, self._zeros, self._empty = weight, scales, zeros, empty
            sms = torch.cuda.get_device_properties(codes.device).multi_processor_count
            self._workspace = torch.zeros(sms, dtype=torch.int32, device=codes.device)
            return
        # Again, after the weights changed: into the same tensors, which a
        # captured graph reads.
        self._weight.copy_(weight)
        self._scales.copy_(scales)
        self._zeros.copy_(zeros)

    def execute(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        context: Context[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        if self._weight is None:
            self.prepare(context)
        assert self._weight is not None and self._workspace is not None
        (x,) = inputs
        rows = x.reshape(-1, self._in)
        made = self._product(
            rows,
            None,
            self._weight,
            None,
            cast("torch.Tensor", self._scales),
            None,
            self._zeros,
            self._empty,
            self._empty,
            self._workspace,
            self._type,
            rows.shape[0],
            self._out,
            self._in,
            is_k_full=True,
            use_fp32_reduce=True,
        )
        if self._bias is not None:
            made = made + context.parameters[self._bias]
        return [made.reshape(*x.shape[:-1], self._out)]


__all__ = ["Marlin"]
