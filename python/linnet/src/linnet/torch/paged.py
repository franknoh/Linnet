"""`std.nn.attention::paged_attention` for generated PyTorch: one query a
row over a pool of key/value pages, each row's pages listed in its row of a
table (paged serving, `linnet.serve`).

Compiled for CUDA, FlexAttention reads each row's pages where they lie in the
pool: the pages before the row's position whole, and the page holding its
position up to it. A row's query heads that share a key/value head go in as
that head's queries, so the kernel reads each page once for all of them.
Otherwise each row's positions are gathered out of the pool and attended
over as two products around a softmax.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any

import torch


def attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    table: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
    size: int,
) -> torch.Tensor:
    """`query` [rows, heads, 1, width] over the pool `key`/`value` [1,
    kv_heads, positions, width]: row `b` sees its positions up to
    `positions[b]`, position `s` at offset `s % size` of page `table[b, s //
    size]`."""
    width = query.shape[3]
    pool = key.shape[2]
    if (
        torch.compiler.is_compiling()
        and query.is_cuda
        and not _split(query)
        and pool % size == 0
        and size % 16 == 0
        and 16 <= width <= 256
    ):
        return _flex(query, key, value, table, positions, scale, size)
    return _gathered(query, key, value, table, positions, scale, size)


def _flex(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    table: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
    size: int,
) -> torch.Tensor:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

    from .flex import fresh_copy

    rows, heads, _, width = query.shape
    kv_heads, pool = key.shape[1], key.shape[2]
    group = heads // kv_heads
    pages = table.shape[1]
    at = positions.long()
    page = at // size
    # The pages before a row's position are seen whole; the one holding it,
    # up to it. Both lists are as long: the decoding kernel wraps its place
    # in either by the length of the first (PyTorch 2.14).
    whole = page.to(torch.int32).reshape(rows, 1, 1)
    whole_pages = table.to(torch.int32).reshape(rows, 1, 1, pages)
    current = torch.cat([table.gather(1, page[:, None]), torch.zeros_like(table[:, 1:])], dim=1).to(
        torch.int32
    )
    current = current.reshape(rows, 1, 1, pages)
    partial = torch.ones(rows, 1, 1, dtype=torch.int32, device=query.device)

    def mask_mod(b: Any, h: Any, q: Any, kv: Any) -> Any:
        return kv % size <= positions[b] % size

    block_mask = BlockMask.from_kv_blocks(
        partial,
        current,
        whole,
        whole_pages,
        BLOCK_SIZE=(128, size),
        mask_mod=mask_mod,
        seq_lengths=(group, pool),
        compute_q_blocks=False,
    )
    # A copy Inductor cannot fold back into a view of the projection.
    grouped = fresh_copy(query.reshape(rows, kv_heads, group, width))
    out = flex_attention(grouped, key, value, block_mask=block_mask, scale=scale)
    assert isinstance(out, torch.Tensor)
    return out.reshape(rows, heads, 1, width)


def _gathered(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    table: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
    size: int,
) -> torch.Tensor:
    rows, heads, _, width = query.shape
    kv_heads = key.shape[1]
    pages = table.shape[1]
    logical = torch.arange(pages * size, device=query.device)
    at = table.long()[:, logical // size] * size + logical % size  # [rows, pages * size]
    keys = key[0][:, at].transpose(0, 1)  # [rows, kv_heads, pages * size, width]
    values = value[0][:, at].transpose(0, 1)
    grouped = query.reshape(rows, kv_heads, heads // kv_heads, width)
    scores = torch.matmul(grouped, keys.transpose(-1, -2)).float() * scale
    seen = logical[None, :] <= positions.long()[:, None]
    scores = scores.masked_fill(~seen[:, None, None, :], -1e30)
    weights = torch.softmax(scores, dim=-1).to(value.dtype)
    return torch.matmul(weights, values).reshape(rows, heads, 1, width)


def _split(value: torch.Tensor) -> bool:
    """Whether `value` is split from the outside (a DTensor), which
    FlexAttention does not take."""
    from torch.distributed.tensor import DTensor

    return isinstance(value, DTensor)


__all__ = ["attend"]
