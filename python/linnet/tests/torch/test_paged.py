"""`linnet.torch.paged.prefill`: prompt tokens over a pool of pages. Its
FlexAttention path lists, for each block of 128 tokens, the pages it reads in
part and whole; these tests hold the lists and the mask to what each token
sees, and FlexAttention's numbers (eager, on the CPU) to the gathered path's."""

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import random

import pytest
import torch

from linnet.torch import paged

SIZE = 16


def _pool() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Four rows of up to twelve pages in a pool of 32: rows 1 and 2 share
    row 0's first two pages, row 3 has none (it ended). Tokens: a chunk of
    row 0 that fills the first block, chunks of rows 1 and 2 at their own
    positions, then padding as the engine packs it."""
    table = torch.zeros(4, 12, dtype=torch.int32)
    table[0, :11] = torch.tensor([5, 9, 3, 17, 22, 8, 12, 19, 25, 26, 28])
    table[1, :6] = torch.tensor([5, 9, 11, 30, 2, 31])
    table[2, :4] = torch.tensor([5, 9, 14, 27])
    rows = [0] * 128 + [1] * 50 + [2] * 30
    positions = [*range(40, 168), *range(32, 82), *range(20, 50)]
    pad = 256 - len(rows)
    return (
        table,
        torch.tensor(rows + [0] * pad, dtype=torch.int32),
        torch.tensor(positions + [0] * pad, dtype=torch.int32),
    )


def _seen(table: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """[tokens, pool]: which pool positions each token sees."""
    seen = torch.zeros(rows.shape[0], 32 * SIZE, dtype=torch.bool)
    for token, (row, at) in enumerate(zip(rows.tolist(), positions.tolist(), strict=True)):
        for position in range(at + 1):
            page = int(table[row, position // SIZE])
            seen[token, page * SIZE + position % SIZE] = True
    return seen


def test_blocks_list_what_each_token_sees() -> None:
    table, rows, positions = _pool()
    seen = _seen(table, rows, positions)
    blocks = paged._blocks(table, rows, positions, 32 * SIZE, SIZE)
    mask = paged._mask(blocks, 32 * SIZE, SIZE)
    partial_count, partial = mask.kv_num_blocks, mask.kv_indices
    whole_count, whole = mask.full_kv_num_blocks, mask.full_kv_indices
    assert whole_count is not None and whole is not None
    assert mask.mask_mod is not None
    queries = torch.arange(128)
    keys = torch.arange(32 * SIZE)
    zero = torch.zeros((), dtype=torch.int64)
    for block in range(2):
        # A block of tokens is a sequence of the batch.
        number = torch.tensor(block)
        modded = mask.mask_mod(number, zero, queries[:, None], keys[None, :])
        tokens = slice(block * 128, (block + 1) * 128)
        parts = set(partial[block, 0, 0, : int(partial_count[block, 0, 0])].tolist())
        wholes = set(whole[block, 0, 0, : int(whole_count[block, 0, 0])].tolist())
        assert not parts & wholes
        reached = {int(k) // SIZE for k in seen[tokens].any(0).nonzero().flatten()}
        assert reached == parts | wholes
        for page in wholes:
            assert bool(seen[tokens, page * SIZE : (page + 1) * SIZE].all())
        for page in parts:
            keys_of = slice(page * SIZE, (page + 1) * SIZE)
            assert torch.equal(modded[:, keys_of], seen[tokens, keys_of])
    # The first block, row 0's alone, reads the two pages before position 40
    # whole; the second has three rows' tokens and reads nothing whole.
    assert set(whole[0, 0, 0, :2].tolist()) == {5, 9}
    assert int(whole_count[0, 0, 0]) == 2 and int(whole_count[1, 0, 0]) == 0


def test_flex_attention_gives_the_gathered_numbers() -> None:
    table, rows, positions = _pool()
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(1, 4, 256, 16, generator=generator)
    key = torch.randn(1, 2, 32 * SIZE, 16, generator=generator)
    value = torch.randn(1, 2, 32 * SIZE, 16, generator=generator)
    blocks = paged._blocks(table, rows, positions, 32 * SIZE, SIZE)
    flexed = paged._flex_prefill(query, key, value, blocks, 0.25, SIZE)
    gathered = paged.prefill(query, key, value, table, rows, positions, 0.25, SIZE)
    torch.testing.assert_close(flexed, gathered, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(("heads", "kv_heads", "size"), [(32, 4, 64), (32, 32, 64), (32, 4, 16)])
def test_compiled_flex_attention_gives_the_gathered_numbers(
    heads: int, kv_heads: int, size: int
) -> None:
    """Compiled for CUDA, the block lists and what the mask reads are built in
    the pass's own graph: still the gathered path's numbers."""
    rng = random.Random(0)
    rows, pages, tokens = 16, 512 // size, 512
    pool_pages = rows * pages + 1
    free = list(range(1, pool_pages))
    rng.shuffle(free)
    table = torch.zeros(rows, pages, dtype=torch.int32)
    lengths: list[int] = []
    for row in range(rows):
        count = rng.randint(1, pages)
        listed = [free.pop() for _ in range(count)]
        if row > 0 and count > 2:
            listed[:2] = table[0, :2].tolist()  # a common start
        table[row, :count] = torch.tensor(listed)
        lengths.append(count * size)
    owners: list[int] = []
    positions: list[int] = []
    while len(owners) < tokens:
        row = rng.randrange(rows)
        start = rng.randrange(lengths[row])
        count = min(rng.randint(1, 300), lengths[row] - start, tokens - len(owners))
        owners += [row] * count
        positions += range(start, start + count)
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(1, heads, tokens, 64, generator=generator).cuda()
    key = torch.randn(1, kv_heads, pool_pages * size, 64, generator=generator).cuda()
    value = torch.randn(1, kv_heads, pool_pages * size, 64, generator=generator).cuda()
    inputs = (
        query,
        key,
        value,
        table.cuda(),
        torch.tensor(owners, dtype=torch.int32).cuda(),
        torch.tensor(positions, dtype=torch.int32).cuda(),
    )

    def attend(*args: torch.Tensor) -> torch.Tensor:
        query, key, value, table, rows, positions = args
        return paged.prefill(query, key, value, table, rows, positions, 0.125, size)

    torch._dynamo.reset()
    compiled = torch.compile(attend)
    torch.testing.assert_close(compiled(*inputs), attend(*inputs), atol=1e-4, rtol=1e-4)
