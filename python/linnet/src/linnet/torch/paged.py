"""`std.nn.attention::paged_attention` and `paged_prefill_attention` for
generated PyTorch: queries over a pool of key/value pages, each row's pages
listed in its row of a table (paged serving, `linnet.serve`).

Compiled for CUDA, FlexAttention reads the pages where they lie in the pool.
`attend` takes one query a row: the pages before the row's position whole,
and the page holding its position up to it. A row's query heads that share a
key/value head go in as that head's queries, so the kernel reads each page
once for all of them. `prefill` takes prompt tokens packed end to end, each
of some row: every block of 128 tokens reads the pages its rows see once
for all its tokens, each token masked to its own row up to its position.
Otherwise each row's positions are gathered out of the pool and attended
over as two products around a softmax.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


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

    def mask_mod(
        b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
    ) -> torch.Tensor:
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


def prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    table: torch.Tensor,
    rows: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
    size: int,
) -> torch.Tensor:
    """`query` [1, heads, tokens, width] over the pool `key`/`value` [1,
    kv_heads, positions, width]: token `p` sees row `rows[p]`'s positions up
    to `positions[p]`, position `s` at offset `s % size` of page
    `table[rows[p], s // size]`."""
    width = query.shape[3]
    pool = key.shape[2]
    if (
        torch.compiler.is_compiling()
        and query.is_cuda
        and not _split(query)
        and pool % size == 0
        and size >= 16
        and size & (size - 1) == 0
        and 16 <= width <= 256
    ):
        return _flex_prefill(query, key, value, table, rows, positions, scale, size)
    each = query.permute(2, 1, 0, 3)  # a token a row
    seen = _gathered(each, key, value, table[rows.long()], positions, scale, size)
    return seen.permute(2, 1, 0, 3)


# Tokens a block of the prompt pass's attention: each block reads the pages
# its tokens' rows see.
_TOKEN_BLOCK = 128


def _flex_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    table: torch.Tensor,
    rows: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
    size: int,
) -> torch.Tensor:
    from torch.nn.attention.flex_attention import flex_attention

    from .flex import fresh_copy

    mask = _prefill_mask(table, rows, positions, query.shape[2], key.shape[2], size)
    out = flex_attention(
        fresh_copy(query), key, value, block_mask=mask, scale=scale, enable_gqa=True
    )
    assert isinstance(out, torch.Tensor)
    return out


def _prefill_mask(
    table: torch.Tensor,
    rows: torch.Tensor,
    positions: torch.Tensor,
    tokens: int,
    pool: int,
    size: int,
) -> BlockMask:
    """Which pool positions each token sees, block by block. Every layer of
    a pass makes the same mask from the same inputs; the compiled graph
    keeps one (common subexpressions are merged)."""
    from torch.nn.attention.flex_attention import BlockMask

    from .flex import fresh_copy

    device = table.device
    count, pages = table.shape
    pool_pages = pool // size
    owner = rows.long()
    at = positions.long()
    # Each row's place for every page of the pool (`pages` where the row
    # does not list it; unused places list page 0, past every position the
    # row has): a token sees a pool position when its page comes early
    # enough in its row.
    places = torch.arange(pages, dtype=torch.int32, device=device).expand(count, pages)
    first = torch.full((count, pool_pages), pages, dtype=torch.int32, device=device)
    first = first.scatter_reduce(1, table.long(), places, reduce="amin", include_self=True)
    # The pages each block of tokens reads: its rows' up to the latest
    # position of each in the block. A block of one row's tokens reads the
    # pages wholly before its first token whole, with no mask to apply.
    blocks = -(-tokens // _TOKEN_BLOCK)
    block = torch.arange(tokens, device=device) // _TOKEN_BLOCK
    key = block * count + owner
    latest = torch.full((blocks * count,), -1, dtype=torch.int64, device=device)
    latest = latest.scatter_reduce(0, key, at, reduce="amax", include_self=True)
    earliest = torch.full((blocks * count,), pages * size, dtype=torch.int64, device=device)
    earliest = earliest.scatter_reduce(0, key, at, reduce="amin", include_self=True)
    latest, earliest = latest.reshape(blocks, count, 1), earliest.reshape(blocks, count, 1)
    alone = ((latest >= 0).sum(1, keepdim=True) == 1).reshape(blocks, 1, 1)
    starts = torch.arange(pages, device=device) * size
    listed = table.long().expand(blocks, count, pages).reshape(blocks, count * pages)

    def pool_pages_of(seen: torch.Tensor) -> torch.Tensor:
        reads = torch.zeros(blocks, pool_pages, dtype=torch.int32, device=device)
        flat = seen.to(torch.int32).reshape(blocks, count * pages)
        return reads.scatter_reduce(1, listed, flat, reduce="amax", include_self=True)

    reads = pool_pages_of(starts <= latest)
    whole = pool_pages_of((starts + size <= earliest) & (latest >= 0) & alone)
    partial = reads - whole

    def lists(live: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        number = live.sum(-1).to(torch.int32).reshape(1, 1, blocks)
        order = torch.argsort(live, dim=-1, descending=True, stable=True).to(torch.int32)
        return number, order.reshape(1, 1, blocks, pool_pages)

    # Inductor misreads a tensor the mask reads when the same graph computes
    # it (PyTorch 2.14: wrong numbers, or a lowering error): each goes
    # through a copy it cannot see into.
    first, owner, at = fresh_copy(first), fresh_copy(owner), fresh_copy(at)

    def mask_mod(
        b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
    ) -> torch.Tensor:
        return first[owner[q], kv // size] * size + kv % size <= at[q]

    return BlockMask.from_kv_blocks(
        *lists(partial),
        *lists(whole),
        BLOCK_SIZE=(_TOKEN_BLOCK, size),
        mask_mod=mask_mod,
        seq_lengths=(tokens, pool),
    )


def _split(value: torch.Tensor) -> bool:
    """Whether `value` is split from the outside (a DTensor), which
    FlexAttention does not take."""
    from torch.distributed.tensor import DTensor

    return isinstance(value, DTensor)


__all__ = ["attend", "prefill"]
