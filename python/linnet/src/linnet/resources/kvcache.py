"""Key/value caches: which states hold one, and how a backend lays it out.

A Linnet model declares its caches as `state` members, shaped by the model:
`Tensor[Batch, KvHeads, MaxSeq, HeadDim; T]` per layer for grouped-query
attention, one key/value head for multi-query attention, as many as query
heads for multi-head attention, and whatever a layer declares when layers
differ. Their bytes are therefore exact from the manifest, with no
assumption about the attention variant. A state is a cache when a method
writes it with an operation from `std.nn.cache`.

How a backend stores a cache is a layout (`KVLayout`): Linnet's own runtime
keeps each one contiguous, as declared; a serving engine that pages it
(`PagedLayout`) rounds the tokens up to whole blocks and keeps spare ones.
The graph is the same either way.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .. import dtypes, ir
from . import expr as ex
from .graph import Confidence, MemoryObject


def _block_paths(program: ir.Program) -> dict[str, str]:
    """The block at each path of the hierarchy, `[*]` for array elements:
    `{"": "Model", "layers[*]": "DecoderLayer", ...}`."""
    paths: dict[str, str] = {}

    def visit(path: str, name: str) -> None:
        paths[path] = name
        block = program.blocks.get(name)
        if block is None:
            return
        for member in block.members:
            if member.kind != "sub":
                continue
            member_type = member.type
            suffix = ""
            if isinstance(member_type, ir.OptionalType):
                member_type = member_type.inner
            if isinstance(member_type, ir.ArrayType):
                member_type = member_type.element
                suffix = "[*]"
            if isinstance(member_type, ir.NamedType):
                child = f"{path}.{member.name}{suffix}" if path else f"{member.name}{suffix}"
                visit(child, member_type.name)

    visit("", program.root.name)
    return paths


def kv_state_paths(program: ir.Program) -> frozenset[str]:
    """The manifest paths of the states written through `std.nn.cache`."""
    written: set[tuple[str, str]] = set()
    for function in program.functions.values():
        if function.block is None:
            continue
        producers: dict[int, ir.Op] = {}
        for op in function.body.walk():
            for result in op.results:
                producers[result.id] = op
        for op in function.body.walk():
            if op.kind != "state.write" or len(op.operands) < 2:
                continue
            producer = producers.get(op.operands[1])
            if producer is None or producer.kind not in ("call", "semantic.call"):
                continue
            if str(producer.attrs.get("callee", "")).startswith("std.nn.cache::"):
                written.add((function.block, str(op.attrs["name"])))
    blocks = _block_paths(program)
    found: set[str] = set()
    for entry in program.manifest:
        if entry.kind != "state":
            continue
        parent, _, leaf = entry.path.rpartition(".")
        if (blocks.get(parent, ""), leaf) in written:
            found.add(entry.path)
    return frozenset(found)


@dataclass(frozen=True, slots=True)
class CacheBytes:
    nbytes: int
    confidence: Confidence
    note: str = ""


class KVLayout(Protocol):
    """How a backend stores one cache tensor."""

    @property
    def name(self) -> str: ...

    def nbytes(
        self, cache: MemoryObject, env: Mapping[str, int], dtype: str | None
    ) -> CacheBytes: ...


class ContiguousLayout:
    """The cache as declared: one contiguous tensor, every position
    allocated up front. Linnet's runtimes store caches this way."""

    name = "contiguous"

    def nbytes(self, cache: MemoryObject, env: Mapping[str, int], dtype: str | None) -> CacheBytes:
        if dtype is None or dtype == cache.dtype:
            return CacheBytes(ex.evaluate(cache.nbytes, env), Confidence.EXACT)
        elements = ex.evaluate(ex.product(cache.shape), env)
        return CacheBytes(
            elements * dtypes.dtype(dtype).element_bytes,
            Confidence.MODELED,
            f"stored as {dtype} rather than the declared {cache.dtype}",
        )


@dataclass(frozen=True, slots=True)
class PagedLayout:
    """Positions in blocks of `block_tokens`, as paging serving engines
    store them: the cache's token axes (`token_axes`, by default the batch
    and position axes of a `[Batch, Heads, Seq, Dim]` cache) are allocated
    in whole blocks, `reserve_blocks` spare ones are kept, and each block is
    rounded up to `alignment` bytes."""

    block_tokens: int = 16
    reserve_blocks: int = 0
    alignment: int = 1
    token_axes: tuple[int, ...] = (0, 2)
    name: str = "paged"

    def nbytes(self, cache: MemoryObject, env: Mapping[str, int], dtype: str | None) -> CacheBytes:
        shape = [ex.evaluate(d, env) for d in cache.shape]
        tokens = math.prod(shape[a] for a in self.token_axes)
        per_token = math.prod(s for i, s in enumerate(shape) if i not in self.token_axes)
        element = dtypes.dtype(dtype or cache.dtype).element_bytes
        block = per_token * self.block_tokens * element
        block = -(-block // self.alignment) * self.alignment
        blocks = -(-tokens // self.block_tokens) + self.reserve_blocks
        return CacheBytes(
            blocks * block,
            Confidence.MODELED,
            f"{blocks} blocks of {self.block_tokens} positions",
        )


def bytes_per_token(
    caches: list[MemoryObject], env: Mapping[str, int], axes: tuple[int, ...] = (0, 2)
) -> int:
    """Cache bytes one more position of one more sequence costs: every
    cache's size divided by its token axes."""
    total = 0
    for cache in caches:
        shape = [ex.evaluate(d, env) for d in cache.shape]
        if len(shape) <= max(axes):
            continue
        tokens = math.prod(shape[a] for a in axes)
        total += ex.evaluate(cache.nbytes, env) // max(tokens, 1)
    return total
