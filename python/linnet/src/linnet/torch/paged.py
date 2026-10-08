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

from typing import TYPE_CHECKING, TypeAlias

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
    blocks: PrefillBlocks | None = None,
) -> torch.Tensor:
    """`query` [1, heads, tokens, width] over the pool `key`/`value` [1,
    kv_heads, positions, width]: token `p` sees row `rows[p]`'s positions up
    to `positions[p]`, position `s` at offset `s % size` of page
    `table[rows[p], s // size]`. `blocks` is `prefill_blocks` of the same
    pass, made once for all its layers (made here without)."""
    width = query.shape[3]
    if torch.compiler.is_compiling() and not _split(query) and 16 <= width <= 256:
        if blocks is None:
            blocks = prefill_blocks(table, rows, positions, key.shape[2], size)
        if blocks is not None and query.is_cuda:
            return _flex_prefill(query, key, value, blocks, scale, size)
    each = query.permute(2, 1, 0, 3)  # a token a row
    seen = _gathered(each, key, value, table[rows.long()], positions, scale, size)
    return seen.permute(2, 1, 0, 3)


# Tokens a block of the prompt pass's attention: each block reads the pages
# its tokens' rows see.
_TOKEN_BLOCK = 128

# What a pass's FlexAttention reads, the same for every layer: for each block
# of tokens, which of its rows read each page of the pool (two words of bits,
# a bit a row), each token's bit and position, each page's place in the rows
# that list it, and the pages each block reads in part and whole, as
# FlexAttention lists them.
PrefillBlocks: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def prefill_blocks(
    table: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor, pool: int, size: int
) -> PrefillBlocks | None:
    """`prefill`'s FlexAttention inputs for a pass's tokens over a pool of
    `pool` positions, or None where FlexAttention does not run (off CUDA, a
    pass that is not whole blocks, or pages it cannot take whole)."""
    if (
        not table.is_cuda
        or rows.shape[0] % _TOKEN_BLOCK
        or pool % size
        or size < 16
        or size & (size - 1)
    ):
        return None
    return _blocks(table, rows, positions, pool, size)


def _flex_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blocks: PrefillBlocks,
    scale: float,
    size: int,
) -> torch.Tensor:
    """FlexAttention with each block of tokens a sequence of the batch: the
    mask then reads what it needs of a block by the block's number, which is
    the same for a whole tile, rather than by each token's row. The blocks
    are a view of the query, which starts where its storage does (a pass's
    prompts come before its steps), so Inductor reads it right uncopied."""
    from torch.nn.attention.flex_attention import flex_attention

    _, heads, tokens, width = query.shape
    count = tokens // _TOKEN_BLOCK
    by_token = query.transpose(1, 2).reshape(count, _TOKEN_BLOCK, heads, width)
    grouped = by_token.transpose(1, 2)
    out = flex_attention(
        grouped,
        key.expand(count, -1, -1, -1),
        value.expand(count, -1, -1, -1),
        block_mask=_mask(blocks, key.shape[2], size),
        scale=scale,
        enable_gqa=True,
    )
    assert isinstance(out, torch.Tensor)
    return out.transpose(1, 2).reshape(1, tokens, heads, width).transpose(1, 2)


def _blocks(
    table: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor, pool: int, size: int
) -> PrefillBlocks:
    from .flex import fresh_copy

    device = table.device
    count, pages = table.shape
    tokens = rows.shape[0]
    pool_pages = pool // size
    blocks = tokens // _TOKEN_BLOCK
    owner = rows.long()
    at = positions.long()
    key = torch.arange(tokens, device=device) // _TOKEN_BLOCK * count + owner
    latest = torch.full((blocks * count,), -1, dtype=torch.int64, device=device)
    latest = latest.scatter_reduce(0, key, at, reduce="amax", include_self=True)
    earliest = torch.full((blocks * count,), pages * size, dtype=torch.int64, device=device)
    earliest = earliest.scatter_reduce(0, key, at, reduce="amin", include_self=True)
    latest, earliest = latest.reshape(blocks, count, 1), earliest.reshape(blocks, count, 1)
    present = latest >= 0
    # Each block's rows numbered in row order, each its bit in one of two
    # words: a block of 128 tokens has 128 rows at most.
    numbers = present.long().cumsum(1) - 1
    ordinal = numbers.reshape(-1)[key]
    one = torch.ones((), dtype=torch.int64, device=device)
    low = torch.where(numbers < 64, one << numbers.clamp(0, 63), 0)
    high = torch.where(numbers >= 64, one << (numbers - 64).clamp(0, 63), 0)
    # A page holds the same positions in every row that lists it (rows share
    # the pages of a common start): its place, where page 0's is moot.
    places = torch.arange(pages, device=device).expand(count, pages).reshape(-1)
    logical = torch.full((pool_pages,), pages, dtype=torch.int64, device=device)
    logical = logical.scatter_reduce(0, table.long().reshape(-1), places, reduce="amin")
    starts = torch.arange(pages, device=device) * size
    listed = table.long().expand(blocks, count, pages).reshape(blocks, count * pages)
    seen = starts <= latest  # [blocks, rows, places]: the row reads the place's page

    def words(bits: torch.Tensor) -> torch.Tensor:
        """Each block's bits of the rows reading each page: distinct bits
        summed, which is their union."""
        each = torch.where(seen, bits, 0).reshape(blocks, count * pages)
        out = torch.zeros(blocks, pool_pages, dtype=torch.int64, device=device)
        return out.scatter_add(1, listed, each)

    low_words, high_words = words(low), words(high)
    reads = (low_words != 0) | (high_words != 0)
    # A block of one row's tokens reads the pages wholly before its first
    # token whole, with no mask to apply.
    alone = (present.sum(1, keepdim=True) == 1).reshape(blocks, 1, 1)
    before = (starts + size <= earliest) & present & alone
    whole = torch.zeros(blocks, pool_pages, dtype=torch.int32, device=device)
    whole = whole.scatter_reduce(
        1,
        listed,
        before.to(torch.int32).reshape(blocks, count * pages),
        reduce="amax",
        include_self=True,
    )
    partial = reads.to(torch.int32) - whole

    # Inductor misreads what FlexAttention reads when the same graph computes
    # it (PyTorch 2.14: wrong numbers, or a lowering error): each goes
    # through a copy it cannot see into.
    def lists(live: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        number = live.sum(-1).to(torch.int32).reshape(blocks, 1, 1)
        order = torch.argsort(live, dim=-1, descending=True, stable=True).to(torch.int32)
        return fresh_copy(number), fresh_copy(order.reshape(blocks, 1, 1, pool_pages))

    return (
        fresh_copy(low_words),
        fresh_copy(high_words),
        fresh_copy(ordinal),
        fresh_copy(logical),
        fresh_copy(at),
        *lists(partial),
        *lists(whole),
    )


def _mask(blocks: PrefillBlocks, pool: int, size: int) -> BlockMask:
    """The block mask of `blocks`, a block of tokens a sequence: which pool
    positions each token sees."""
    from torch.nn.attention.flex_attention import BlockMask

    low, high, ordinal, logical, at, count, order, whole_count, whole_order = blocks

    def mask_mod(
        b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
    ) -> torch.Tensor:
        token = b * _TOKEN_BLOCK + q
        page = kv // size
        number = ordinal[token]
        word = torch.where(number < 64, low[b, page], high[b, page])
        reads = ((word >> (number % 64)) & 1) == 1
        return reads & (logical[page] * size + kv % size <= at[token])

    return BlockMask.from_kv_blocks(
        count,
        order,
        whole_count,
        whole_order,
        BLOCK_SIZE=(_TOKEN_BLOCK, size),
        mask_mod=mask_mod,
        seq_lengths=(_TOKEN_BLOCK, pool),
        compute_q_blocks=False,
    )


def _split(value: torch.Tensor) -> bool:
    """Whether `value` is split from the outside (a DTensor), which
    FlexAttention does not take."""
    from torch.distributed.tensor import DTensor

    return isinstance(value, DTensor)


__all__ = ["PrefillBlocks", "attend", "prefill", "prefill_blocks"]
