"""FlashAttention as a Linnet backend, a plugin from a package of its own
(docs/backends.md). It takes four kinds of attention call, each read
without a copy and with every length found on the device, so a plan
captured as one CUDA graph replays it for any positions:

- decode: `grouped_attention` or `grouped_attention_rows`, one query a row
  over a KV cache `[rows, kv heads, length, width]`, under a mask that
  reaches a prefix of it (`Graph.reaches_prefix`), as
  `flash_attn_with_kvcache` with each row's length from the mask;
- packed prompts: `grouped_attention` over a pass of prompts packed one
  after another, each token seeing its own prompt up to itself
  (`Graph.segment_causal`), as `flash_attn_varlen_func`, the prompts'
  bounds from the segments. The mask's own steps are part of the region,
  so it is never built, unless the model computes something else between
  them and the call; then the bounds are read off the mask. Each prompt's
  positions rise one a token, as the serving engine packs them; its
  padding tokens, which nothing reads, differ;
- paged decode: `paged_attention`, one query a row over a pool of pages,
  as `flash_attn_with_kvcache` with the rows' page table;
- paged prompts: `paged_prefill_attention`, as `flash_attn_varlen_func`
  over the pages, the prompts' bounds from where the rows or the positions
  break.

FlashAttention 2 reads pages of a multiple of 256 positions: a model with
smaller pages (`PageSize`) keeps its paged calls.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence
from typing import cast

import torch

from linnet.backends import (
    Call,
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

_ATTENTION = "std.nn.attention::grouped_attention"
_ROWS = "std.nn.attention::grouped_attention_rows"
_PAGED = "std.nn.attention::paged_attention"
_PAGED_PROMPTS = "std.nn.attention::paged_prefill_attention"
_DTYPES = ("bf16", "f16")
_PAGE = 256  # FlashAttention 2 reads pages of a multiple of this

_Kernel = Callable[..., torch.Tensor]


def _kernel(name: str) -> _Kernel | None:
    """One of FlashAttention's functions, or None where it is not installed."""
    try:
        module = importlib.import_module("flash_attn")
    except ImportError:
        return None
    return cast("_Kernel", getattr(module, name))


class FlashAttention:
    """Takes the attention calls it can run."""

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
        if (
            target.kind != "cuda"
            or target.capability < (8, 0)
            or _kernel("flash_attn_func") is None
        ):
            return None
        return Capabilities({"flash-attn": self.version}, frozenset(_DTYPES))

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        for call in graph.semantic_calls():
            region = _region(graph, call)
            if region is not None:
                yield region

    def compile(self, region: Region, target: Target, options: Options) -> _Attention:
        if region.call is None:
            raise PlanError("FlashAttention takes an attention call")
        graph = region.graph
        call = graph.calls[region.call]
        paged = call.callee in (_PAGED, _PAGED_PROMPTS)
        scale = graph.scalar(call.arguments[-1 if paged else 3].value)
        if scale is None:
            raise PlanError("FlashAttention cannot run this call")
        where = region.inputs.index

        def at(i: int) -> int:
            return where(cast("int", call.arguments[i].value))

        if call.callee == _PAGED_PROMPTS:
            size = int(call.generics["Size"])
            pages = int(call.generics["Pages"])
            # q, key, value, table, rows, positions
            return _PagedPrompts(
                region, scale, [at(0), at(1), at(2), at(3), at(4), at(5)], size, pages
            )
        if call.callee == _PAGED:
            size = int(call.generics["Size"])
            pages = int(call.generics["Pages"])
            # q, key, value, table, positions
            return _PagedDecode(region, scale, [at(0), at(1), at(2), at(3), at(4)], size, pages)
        mask = call.arguments[4].value
        found = graph.segment_causal(cast("int", mask)) if mask is not None else None
        if found is not None:
            if found.segments in region.inputs:
                return _Packed(region, scale, [where(found.segments), at(0), at(1), at(2)])
            return _Packed(region, scale, [at(4), at(0), at(1), at(2)], by_mask=True)
        return _Decode(region, scale, [at(0), at(1), at(2), at(4)])

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        return None


def _region(graph: Graph, call: Call) -> Region | None:
    """The region FlashAttention runs for `call`, or None where it does not
    take it."""
    width = call.generics.get("D", 0)
    if not (isinstance(width, int) and 0 < width <= 256 and width % 8 == 0):
        return None
    if call.callee in (_PAGED, _PAGED_PROMPTS):
        size, length = call.generics.get("Size", 0), call.generics.get("N", 0)
        query, key, value = call.arguments[:3]
        if (
            not isinstance(size, int)
            or not isinstance(length, int)
            or size % _PAGE
            or length % size
            or query.dtype not in _DTYPES
            or key.dtype != query.dtype
            or value.dtype != query.dtype
            or graph.scalar(call.arguments[-1].value) is None
        ):
            return None
        return Region.of(graph, call.start, call.stop, call.index)
    if call.callee not in (_ATTENTION, _ROWS) or len(call.arguments) != 5:
        return None
    query, key, value, scale, mask = call.arguments
    if (
        query.dtype not in _DTYPES
        or key.dtype != query.dtype
        or value.dtype != query.dtype
        or graph.scalar(scale.value) is None
        or mask.value is None
    ):
        return None
    if call.generics.get("Q") == 1 and graph.reaches_prefix(mask.value):
        return Region.of(graph, call.start, call.stop, call.index)
    found = graph.segment_causal(mask.value) if call.callee == _ATTENTION else None
    if (
        found is None
        or call.generics.get("B") != 1
        or call.generics.get("Q") != call.generics.get("K")
    ):
        return None
    # From the mask's first step, so the mask is never built, where the
    # steps between build nothing else; else the call alone, the mask an
    # input.
    wide = Region.of(graph, found.start, call.stop, call.index)
    reads = {found.segments, *(a.value for a in (query, key, value))}
    made = {o.value for o in call.results}
    if reads <= set(wide.inputs) and set(wide.outputs) <= made:
        return wide
    return Region.of(graph, call.start, call.stop, call.index)


def _bounds(start: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each token's sequence, and the sequences' offsets (`cu_seqlens`), as
    many sequences as tokens, those past the last empty."""
    count = start.shape[0]
    ids = start.long().cumsum(0) - 1
    lengths = torch.zeros(count, dtype=torch.int32, device=start.device)
    lengths.scatter_add_(0, ids, torch.ones(count, dtype=torch.int32, device=start.device))
    offsets = torch.zeros(count + 1, dtype=torch.int32, device=start.device)
    offsets[1:] = lengths.cumsum(0)
    return ids, offsets


class _Attention:
    """An attention call FlashAttention runs: its inputs by where each
    argument stands among the region's."""

    def __init__(self, region: Region, scale: float, order: list[int]) -> None:
        values = region.graph.values
        self._scale = scale
        self._order = order
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
        return [self.attend(*(inputs[i] for i in self._order))]

    def attend(self, *arguments: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _Decode(_Attention):
    def attend(self, *arguments: torch.Tensor) -> torch.Tensor:
        query, key, value, mask = arguments
        rows, length = query.shape[0], key.shape[2]
        # Each row's length: how many keys its mask reaches (a prefix).
        lengths = mask.reshape(-1, length).sum(-1, dtype=torch.int32).expand(rows)
        decode = cast("_Kernel", _kernel("flash_attn_with_kvcache"))
        out = decode(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            cache_seqlens=lengths.contiguous(),
            softmax_scale=self._scale,
            causal=False,
        )
        return out.transpose(1, 2)


class _Packed(_Attention):
    def __init__(
        self, region: Region, scale: float, order: list[int], by_mask: bool = False
    ) -> None:
        super().__init__(region, scale, order)
        self._by_mask = by_mask  # bounds read off the mask rather than the segments

    def attend(self, *arguments: torch.Tensor) -> torch.Tensor:
        bounds, query, key, value = arguments
        tokens = query.shape[2]
        # A prompt starts where a token may not see the one before it.
        breaks = ~bounds.diagonal(-1) if self._by_mask else bounds[1:] != bounds[:-1]
        _, offsets = _bounds(_first(breaks))
        varlen = cast("_Kernel", _kernel("flash_attn_varlen_func"))
        out = varlen(
            query[0].transpose(0, 1),
            key[0].transpose(0, 1),
            value[0].transpose(0, 1),
            cu_seqlens_q=offsets,
            cu_seqlens_k=offsets,
            max_seqlen_q=tokens,
            max_seqlen_k=tokens,
            softmax_scale=self._scale,
            causal=True,
        )
        return out.transpose(0, 1)[None]


def _first(breaks: torch.Tensor) -> torch.Tensor:
    """Which tokens start a sequence: the first, and each where `breaks`
    (for the tokens after the first) holds."""
    return torch.cat([torch.ones(1, dtype=torch.bool, device=breaks.device), breaks])


def _pool(cache: torch.Tensor, size: int) -> torch.Tensor:
    """A pool `[1, kv heads, length, width]` as FlashAttention's pages
    `[pages, size, kv heads, width]`, a view."""
    if not cache.is_contiguous():
        raise PlanError("FlashAttention reads a contiguous pool of pages")
    _, heads, length, width = cache.shape
    return cache.as_strided(
        (length // size, size, heads, width),
        (size * width, width, length * width, 1),
        cache.storage_offset(),
    )


class _PagedDecode(_Attention):
    def __init__(
        self, region: Region, scale: float, order: list[int], size: int, pages: int
    ) -> None:
        super().__init__(region, scale, order)
        self._size = size
        self._reach = size * pages  # positions a row's table reaches

    def attend(self, *arguments: torch.Tensor) -> torch.Tensor:
        query, key, value, table, positions = arguments
        decode = cast("_Kernel", _kernel("flash_attn_with_kvcache"))
        out = decode(
            query.transpose(1, 2),
            _pool(key, self._size),
            _pool(value, self._size),
            cache_seqlens=(positions + 1).clamp(max=self._reach).to(torch.int32),
            block_table=table,
            softmax_scale=self._scale,
            causal=False,
        )
        return out.transpose(1, 2)


class _PagedPrompts(_Attention):
    def __init__(
        self, region: Region, scale: float, order: list[int], size: int, pages: int
    ) -> None:
        super().__init__(region, scale, order)
        self._size = size
        self._reach = size * pages

    def attend(self, *arguments: torch.Tensor) -> torch.Tensor:
        query, key, value, table, rows, positions = arguments
        tokens = rows.shape[0]
        # A prompt runs while its row stays and its positions rise one a token.
        breaks = (rows[1:] != rows[:-1]) | (positions[1:] != positions[:-1] + 1)
        ids, offsets = _bounds(_first(breaks))
        # Each prompt reaches its row up to its last position.
        reach = torch.zeros(tokens, dtype=torch.int64, device=rows.device)
        reach.scatter_reduce_(0, ids, positions.long() + 1, reduce="amax")
        keys = torch.zeros(tokens + 1, dtype=torch.int32, device=rows.device)
        keys[1:] = reach.clamp(max=self._reach).cumsum(0)
        first = offsets[:-1].clamp(max=tokens - 1).long()
        varlen = cast("_Kernel", _kernel("flash_attn_varlen_func"))
        out = varlen(
            query[0].transpose(0, 1),
            _pool(key, self._size),
            _pool(value, self._size),
            cu_seqlens_q=offsets,
            cu_seqlens_k=keys,
            max_seqlen_q=tokens,
            max_seqlen_k=self._reach,
            softmax_scale=self._scale,
            causal=True,
            block_table=table[rows[first].long()],
        )
        return out.transpose(0, 1)[None]


__all__ = ["FlashAttention"]
