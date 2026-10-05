"""The resource analysis IR: every tensor an entry touches as a memory object,
every operation as a step, and the liveness that decides peak memory.

`linnet.resources.trace` builds a `TensorGraph` from a compiled program.
Sizes stay symbolic (`linnet.resources.expr`); the structure (which object
is live at which step) is fixed by the program, so the peak for any binding
of the free symbols is one sweep over the steps, not a re-analysis.

A view or an in-place result names its storage owner rather than owning
bytes, so views, aliases, tied parameters and in-place outputs are never
counted twice: liveness is tracked per storage owner, and an owner stays
live until the last use of any object that shares it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from . import expr as ex


class Category(StrEnum):
    """What a block of memory is for."""

    PARAMETER = "parameters"
    BUFFER = "buffers"
    STATE = "state"
    KV_CACHE = "kv_cache"
    INPUT = "inputs"
    ACTIVATION = "activations"
    WORKSPACE = "workspace"
    GRADIENT = "gradients"
    TEMPORARY = "backward_temporaries"
    MASTER = "master_weights"
    OPTIMIZER = "optimizer_states"
    COMMUNICATION = "communication"
    RUNTIME = "runtime"


class Confidence(StrEnum):
    """How a number was obtained.

    `exact`: from the program and its bindings alone. `backend-modeled`: from
    a backend resource model's statement of what an implementation
    allocates. `estimated`: a typical value for something the runtime decides
    (a CUDA context). `unknown`: nothing models it; the number is absent, not
    zero."""

    EXACT = "exact"
    MODELED = "backend-modeled"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


CONFIDENCE_ORDER = (
    Confidence.EXACT,
    Confidence.MODELED,
    Confidence.ESTIMATED,
    Confidence.UNKNOWN,
)


def weakest(confidences: Iterable[Confidence]) -> Confidence:
    """The least certain of several confidences."""
    return max(confidences, key=CONFIDENCE_ORDER.index, default=Confidence.EXACT)


@dataclass(frozen=True, slots=True)
class MemoryObject:
    """A tensor's storage, or a view of another object's.

    `storage` is the id of the object that owns the bytes: itself, or the
    object a view, alias or in-place result shares. `producer` is the step
    that creates it, None for what exists before the entry runs (inputs,
    parameters, buffers, state). `device` is the logical device or shard
    that owns it."""

    id: int
    name: str
    category: Category
    shape: tuple[ex.Expr, ...]
    dtype: str
    nbytes: ex.Expr
    producer: int | None
    storage: int
    persistent: bool = False
    path: str | None = None
    device: str = "device:0"

    @property
    def owns_storage(self) -> bool:
        return self.storage == self.id


@dataclass(frozen=True, slots=True)
class Step:
    """One operation as the backend runs it.

    `kind` is the Core IR operation, `implementation` the native function a
    library call lowers to (None when its canonical body was traced
    instead). `scope` is the block path whose method runs it, such as
    `layers.3.attention`; `flops` counts its arithmetic."""

    index: int
    kind: str
    label: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    scope: str
    flops: ex.Expr
    implementation: str | None = None


@dataclass(frozen=True, slots=True)
class TensorGraph:
    """An entry as memory objects and steps, in execution order."""

    entry: str
    objects: tuple[MemoryObject, ...]
    steps: tuple[Step, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]

    def object(self, id: int) -> MemoryObject:
        return self.objects[id]

    def storage_owners(self) -> tuple[MemoryObject, ...]:
        return tuple(o for o in self.objects if o.owns_storage)

    def by_category(self, category: Category) -> tuple[MemoryObject, ...]:
        return tuple(o for o in self.objects if o.owns_storage and o.category == category)


@dataclass(frozen=True, slots=True)
class Lifetime:
    """The steps an object's storage is live through, both ends included."""

    first: int
    last: int


def lifetimes(graph: TensorGraph) -> dict[int, Lifetime]:
    """The lifetime of every transient storage owner: from the step that
    creates it to the last step that reads it or any view of it. Entry
    inputs and outputs live throughout, since the caller holds them.
    Persistent objects (parameters, buffers, state) are not listed: they are
    live throughout too."""
    end = max(len(graph.steps) - 1, 0)
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for obj in graph.objects:
        owner = graph.objects[obj.storage]
        if owner.persistent:
            continue
        start = 0 if obj.producer is None else obj.producer
        first[owner.id] = min(first.get(owner.id, start), start)
        last[owner.id] = max(last.get(owner.id, start), start)
    for step in graph.steps:
        for id in step.inputs:
            owner = graph.objects[id].storage
            if owner in last:
                last[owner] = max(last[owner], step.index)
    for id in (*graph.inputs, *graph.outputs):
        owner = graph.objects[id].storage
        if owner in last:
            last[owner] = end
            if id in graph.inputs:
                first[owner] = 0
    return {id: Lifetime(first[id], last[id]) for id in first}


@dataclass(frozen=True, slots=True)
class Peak:
    """The most transient memory live at once, and where."""

    nbytes: int
    step: int
    live: tuple[int, ...]


def sizes(graph: TensorGraph, env: Mapping[str, int]) -> dict[int, int]:
    """Every storage owner's size in bytes under `env`."""
    return {o.id: ex.evaluate(o.nbytes, env) for o in graph.objects if o.owns_storage}


def peak(
    graph: TensorGraph,
    env: Mapping[str, int],
    spans: Mapping[int, Lifetime] | None = None,
    extra: Mapping[int, int] | None = None,
) -> Peak:
    """The largest sum of live transient storage over the steps.

    `extra` adds bytes live during one step only (a backend workspace),
    keyed by step index."""
    spans = lifetimes(graph) if spans is None else spans
    count = max(len(graph.steps), 1)
    nbytes = sizes(graph, env)
    delta = np.zeros(count + 1, dtype=np.int64)
    for id, span in spans.items():
        delta[span.first] += nbytes[id]
        delta[span.last + 1] -= nbytes[id]
    live = np.cumsum(delta[:count])
    if extra:
        for step, size in extra.items():
            live[step] += size
    at = int(np.argmax(live)) if count else 0
    total = int(live[at]) if count else 0
    owners = tuple(sorted(id for id, s in spans.items() if s.first <= at <= s.last))
    return Peak(total, at, owners)


def peak_expr(graph: TensorGraph, spans: Mapping[int, Lifetime] | None = None) -> ex.Expr:
    """The peak as a formula of the free symbols: the max, over the steps
    where something is created, of the bytes live there. Identical live sets
    appear once."""
    spans = lifetimes(graph) if spans is None else spans
    births = sorted({s.first for s in spans.values()})
    seen: set[frozenset[int]] = set()
    candidates: list[ex.Expr] = []
    for at in births:
        live = frozenset(id for id, s in spans.items() if s.first <= at <= s.last)
        if live in seen or not live:
            continue
        seen.add(live)
        candidates.append(ex.total(graph.objects[id].nbytes for id in sorted(live)))
    return ex.maximum(*candidates) if candidates else ex.ZERO


@dataclass(frozen=True, slots=True)
class BufferPlan:
    """Transient storage packed into one arena: each owner gets an offset no
    live-overlapping owner shares. `naive` is every owner in its own buffer;
    `planned` the arena; `lower_bound` the peak, which no plan beats."""

    naive: int
    planned: int
    lower_bound: int
    offsets: Mapping[int, int]


def plan_buffers(
    graph: TensorGraph,
    env: Mapping[str, int],
    spans: Mapping[int, Lifetime] | None = None,
    alignment: int = 1,
) -> BufferPlan:
    """Greedy by size: the largest owners are placed first, each at the
    lowest offset that no owner live at the same time occupies. Storage of
    tensors that died is reused by later ones, as a static allocator would."""
    spans = lifetimes(graph) if spans is None else spans
    nbytes = sizes(graph, env)
    order = sorted(spans, key=lambda id: (-nbytes[id], spans[id].first, id))
    first = np.array([spans[id].first for id in order], dtype=np.int64)
    last = np.array([spans[id].last for id in order], dtype=np.int64)
    size = np.array([_align(nbytes[id], alignment) for id in order], dtype=np.int64)
    offset = np.zeros(len(order), dtype=np.int64)
    for i in range(len(order)):
        if size[i] == 0:
            continue
        overlap = (first[:i] <= last[i]) & (last[:i] >= first[i]) & (size[:i] > 0)
        taken = sorted(zip(offset[:i][overlap].tolist(), size[:i][overlap].tolist(), strict=True))
        at = 0
        for start, length in taken:
            if at + int(size[i]) <= start:
                break
            at = max(at, start + length)
        offset[i] = at
    planned = int(np.max(offset + size)) if len(order) else 0
    return BufferPlan(
        naive=int(size.sum()),
        planned=planned,
        lower_bound=peak(graph, env, spans).nbytes,
        offsets={id: int(offset[i]) for i, id in enumerate(order)},
    )


def _align(size: int, alignment: int) -> int:
    return -(-size // alignment) * alignment if alignment > 1 else size
