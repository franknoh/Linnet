"""FlashAttention's decode attention as a Linnet backend, a plugin from a
package of its own (docs/backends.md): a `grouped_attention` or
`grouped_attention_rows` call of one query per row over a KV cache, under a
mask that reaches a prefix of the cache, as `flash_attn_with_kvcache`.

It reads Linnet's cache `[rows, kv heads, length, head width]` without a
copy, as FlashAttention's `[rows, length, kv heads, head width]` with other
strides, and each row's length from the mask on the device
(`cache_seqlens`), so a CUDA graph replays it for any position. The scale
is the call's own, computed from constants (`Graph.scalar`).
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

_CALLEES = ("std.nn.attention::grouped_attention", "std.nn.attention::grouped_attention_rows")
_DTYPES = ("bf16", "f16")


class _Decode(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        causal: bool,
    ) -> torch.Tensor: ...


def _decode() -> _Decode | None:
    """FlashAttention's decode over a cache, or None where it is not installed."""
    try:
        module = importlib.import_module("flash_attn")
    except ImportError:
        return None
    return cast("_Decode", module.flash_attn_with_kvcache)


class FlashAttention:
    """Takes the decode attention calls it can run."""

    contract = 1  # the version of `linnet.backends` it was written for

    @property
    def name(self) -> str:
        return "flash-attn"

    @property
    def version(self) -> str:
        try:
            return str(importlib.import_module("flash_attn").__version__)
        except ImportError:
            return "missing"

    def probe(self, target: Target) -> Capabilities | None:
        # FlashAttention 2 runs on Ampere and later.
        if target.kind != "cuda" or target.capability < (8, 0) or _decode() is None:
            return None
        return Capabilities({"flash-attn": self.version}, frozenset(_DTYPES))

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        for call in graph.semantic_calls():
            if call.callee not in _CALLEES or len(call.arguments) != 5:
                continue
            query, key, value, scale, mask = call.arguments
            width = call.generics.get("D", 0)
            if (
                call.generics.get("Q") == 1
                and query.dtype in _DTYPES
                and key.dtype == query.dtype
                and value.dtype == query.dtype
                and 0 < width <= 256
                and width % 8 == 0
                and graph.scalar(scale.value) is not None
                and mask.value is not None
                and graph.reaches_prefix(mask.value)
            ):
                yield Region.of(graph, call.start, call.stop, call.index)

    def compile(self, region: Region, target: Target, options: Options) -> _Attention:
        if region.call is None:
            raise PlanError("FlashAttention takes an attention call")
        graph = region.graph
        call = graph.calls[region.call]
        scale = graph.scalar(call.arguments[3].value)
        decode = _decode()
        if scale is None or decode is None:
            raise PlanError("FlashAttention cannot run this call")
        # The scale, computed from constants, is no input: the region makes it.
        query, key, value, _, mask = call.arguments
        order = [region.inputs.index(a.value) for a in (query, key, value, mask)]
        return _Attention(region, decode, scale, order)

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        return None


class _Attention:
    def __init__(self, region: Region, decode: _Decode, scale: float, order: list[int]) -> None:
        values = region.graph.values
        self._decode = decode
        self._scale = scale
        self._order = order  # query, key, value, mask: where each is among the inputs
        self._meta = ExecutableMeta(
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.inputs),
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.outputs),
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
        query, key, value, mask = (inputs[i] for i in self._order)
        rows, length = query.shape[0], key.shape[2]
        # Each row's length: how many keys its mask reaches (a prefix).
        lengths = mask.reshape(-1, length).sum(-1, dtype=torch.int32).expand(rows)
        out = self._decode(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            cache_seqlens=lengths.contiguous(),
            softmax_scale=self._scale,
            causal=False,
        )
        return [out.transpose(1, 2)]


__all__ = ["FlashAttention"]
