"""FlashInfer's decode attention as a backend for PyTorch tensors
(docs/design/execution.md): a `grouped_attention` or
`grouped_attention_rows` call of one query per row over a KV cache, under a
mask that reaches a prefix of the cache, as
`flashinfer.decode.trtllm_batch_decode_with_kv_cache`. It reads each row's
length from the mask on the device, so a CUDA graph replays it for any
position.

The cache `[rows, kv heads, length, head width]` is read as FlashInfer's
paged cache in `HND` layout without a copy: pages of 128 positions (its
kernels take 16 to 128) are a strided view of it, a page apart every
`128 * width` elements and a head apart every `length * width`, and the
block table lists each row's pages.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from typing import Protocol, cast

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
from ..compiler import PlanError

_CALLEES = ("std.nn.attention::grouped_attention", "std.nn.attention::grouped_attention_rows")
_PAGED = "std.nn.attention::paged_attention"
_DTYPES = ("bf16", "f16")
_WORKSPACE_BYTES = 128 << 20


class _Decode(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        bmm1_scale: float = 1.0,
        *,
        kv_layout: str = "HND",
    ) -> torch.Tensor: ...


def _decode() -> _Decode | None:
    """FlashInfer's decode, or None where it is not installed."""
    try:
        module = importlib.import_module("flashinfer.decode")
    except ImportError:
        return None
    return cast("_Decode", module.trtllm_batch_decode_with_kv_cache)


class FlashInfer:
    """Takes the decode attention calls it can run."""

    @property
    def name(self) -> str:
        return "flashinfer"

    @property
    def version(self) -> str:
        try:
            return str(importlib.import_module("flashinfer").__version__)
        except ImportError:
            return "missing"

    def probe(self, target: Target) -> Capabilities | None:
        # Its decode kernels (xqa, trtllm-gen) need Hopper or later.
        if target.kind != "cuda" or target.capability < (9, 0) or _decode() is None:
            return None
        return Capabilities({"flashinfer": self.version}, frozenset(_DTYPES))

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        for call in graph.semantic_calls(_PAGED):
            query, key, value = call.arguments[:3]
            size = call.generics.get("Size", 0)
            if (
                query.dtype in _DTYPES
                and key.dtype == query.dtype
                and value.dtype == query.dtype
                and call.generics.get("D") in (64, 128, 256)
                and size in (16, 32, 64, 128)
                and graph.scalar(call.arguments[-1].value) is not None
            ):
                yield Region.of(graph, call.start, call.stop, call.index)
        for call in graph.semantic_calls():
            if call.callee not in _CALLEES or len(call.arguments) != 5:
                continue
            query, key, value, scale, mask = call.arguments
            if (
                call.generics.get("Q") != 1
                or query.dtype not in _DTYPES
                or key.dtype != query.dtype
                or value.dtype != query.dtype
                or call.generics.get("D") not in (64, 128, 256)
                or _page(call.generics.get("K", 0)) is None
                or scale.value is None
                or mask.value is None
                or graph.scalar(scale.value) is None
                or not graph.reaches_prefix(mask.value)
            ):
                continue
            yield Region.of(graph, call.start, call.stop, call.index)

    def compile(self, region: Region, target: Target, options: Options) -> _Attention:
        if region.call is None:
            raise PlanError("FlashInfer takes an attention call")
        graph = region.graph
        call = graph.calls[region.call]
        if call.callee == _PAGED:
            scale = graph.scalar(call.arguments[-1].value)
            decode = _decode()
            if scale is None or decode is None:
                raise PlanError("FlashInfer cannot run this call")
            order = [region.inputs.index(cast("int", a.value)) for a in call.arguments[:5]]
            size, pages = int(call.generics["Size"]), int(call.generics["Pages"])
            return _PagedDecode(region, decode, scale, order, size, pages)
        scale = graph.scalar(call.arguments[3].value)
        decode = _decode()
        if scale is None or decode is None:
            raise PlanError("FlashInfer cannot run this call")
        # The scale, computed from constants, is no input: the region makes it.
        query, key, value, _, mask = call.arguments
        order = [region.inputs.index(a.value) for a in (query, key, value, mask)]
        return _Attention(region, decode, scale, order)

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        return None


def _page(length: int) -> int | None:
    """The page size, of those FlashInfer's decode kernels take, a cache of
    `length` positions splits into: the largest."""
    return next((size for size in (128, 64, 32, 16) if length % size == 0), None)


_workspaces: dict[int, torch.Tensor] = {}


def _workspace(device: torch.device) -> torch.Tensor:
    index = device.index or 0
    if index not in _workspaces:
        _workspaces[index] = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    return _workspaces[index]


class _Attention:
    def __init__(self, region: Region, decode: _Decode, scale: float, order: list[int]) -> None:
        values = region.graph.values
        self._decode = decode
        self._scale = scale
        self._order = order  # query, key, value, mask: where each is among the inputs
        self._blocks: torch.Tensor | None = None
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
        rows, heads, _, width = query.shape
        kv_heads, length = key.shape[1], key.shape[2]
        page = _page(length)
        assert page is not None
        per_row = length // page
        # Page p of the view starts p * page positions into the cache; row b's
        # pages are b * kv_heads * per_row onwards, each head a cache apart.
        pages = (rows - 1) * kv_heads * per_row + per_row
        size = (pages, kv_heads, page, width)
        strides = (page * width, length * width, width, 1)
        if self._blocks is None:
            first = torch.arange(rows, device=query.device)[:, None] * kv_heads * per_row
            self._blocks = (first + torch.arange(per_row, device=query.device)).int()
        # Each row's length: how many keys its mask reaches (a prefix).
        lengths = mask.reshape(-1, length).sum(-1, dtype=torch.int32).expand(rows)
        out = self._decode(
            query.reshape(rows, heads, width),
            (torch.as_strided(key, size, strides), torch.as_strided(value, size, strides)),
            _workspace(query.device),
            self._blocks,
            lengths.contiguous().view(torch.uint32),
            length,
            self._scale,
            kv_layout="HND",
        )
        return [out.reshape(rows, heads, 1, width)]


class _PagedDecode(_Attention):
    """`paged_attention`: one query a row over a pool of pages, each row's
    pages its row of the table. The pool `[1, kv heads, length, width]` is
    FlashInfer's `HND` pages as it is, a page apart every `size * width`
    elements."""

    def __init__(
        self,
        region: Region,
        decode: _Decode,
        scale: float,
        order: list[int],
        size: int,
        pages: int,
    ) -> None:
        super().__init__(region, decode, scale, order)
        self._size = size
        self._reach = size * pages  # positions a row's table reaches

    def execute(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        context: Context[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        query, key, value, table, positions = (inputs[i] for i in self._order)
        rows, heads, _, width = query.shape
        _, kv_heads, length, _ = key.shape
        size = (length // self._size, kv_heads, self._size, width)
        strides = (self._size * width, length * width, width, 1)
        lengths = (positions + 1).clamp(max=self._reach).to(torch.int32)
        out = self._decode(
            query.reshape(rows, heads, width),
            (
                torch.as_strided(key, size, strides, key.storage_offset()),
                torch.as_strided(value, size, strides, value.storage_offset()),
            ),
            _workspace(query.device),
            table,
            lengths.view(torch.uint32),
            self._reach,
            self._scale,
            kv_layout="HND",
        )
        return [out.reshape(rows, heads, 1, width)]
