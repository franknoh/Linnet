"""Model parallelism: which axis of each weight tensor parallelism splits
across devices, and which blocks pipeline parallelism puts on each stage.

A model runs tensor-parallel when its large weights are split across the
devices of a mesh, each device computing its share and the framework adding
the collectives the split needs (XLA's partitioner for `linnet.jax`, DTensor
for `linnet.torch`). Any split computes the same numbers; the split only
decides how much crosses between devices. The defaults follow the usual
layout for decoders: projections into heads or into the feed-forward width
are split by output (each device computes some heads, some of the hidden
units), projections back out of them by input (each device holds the rows
its heads produce, and the partial sums are added), and everything else --
embeddings, norms, the output head -- is copied to every device.

Rules are glob patterns over parameter paths, mapped to the axis to split or
`None` for a copy; the first match wins, and a path no rule matches is
copied. An axis a weight's extent does not divide evenly is not split.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping, Sequence

# Linear weights are [out, in]: axis 0 splits the output, axis 1 the input.
DEFAULT_RULES: dict[str, int | None] = {
    "*q_proj.weight": 0,
    "*k_proj.weight": 0,
    "*v_proj.weight": 0,
    "*qkv.weight": 0,
    "*gate_proj.weight": 0,
    "*up_proj.weight": 0,
    "*gate_up.weight": 0,
    "*q_proj.bias": 0,
    "*k_proj.bias": 0,
    "*v_proj.bias": 0,
    "*qkv.bias": 0,
    "*o_proj.weight": 1,
    "*down_proj.weight": 1,
    "*mlp.down.weight": 1,
    "*mlp.up.weight": 0,
    "*mlp.gate.weight": 0,
}

# A KV cache is [rows, heads, positions, width]: split by heads, as the
# key and value projections that fill it are.
STATE_AXIS = 1

# The largest all-reduce `linnet.torch.collectives` sums with its one-shot
# kernel; past it NCCL is faster. Each process holds a buffer of twice this
# (`ONE_SHOT_BYTES / 2` f32 slots), which its peers read.
ONE_SHOT_BYTES = 64 * 1024


def split_axis(
    path: str,
    shape: Sequence[int],
    devices: int,
    rules: Mapping[str, int | None] | None = None,
) -> int | None:
    """The axis of the weight at `path` to split over `devices`, or `None`
    to copy it to each."""
    table = DEFAULT_RULES if rules is None else rules
    for pattern, axis in table.items():
        if fnmatch.fnmatchcase(path, pattern):
            if axis is None or axis >= len(shape) or shape[axis] % devices != 0:
                return None
            return axis
    return None


def state_axis(shape: Sequence[int], devices: int) -> int | None:
    """The axis of a state member (a KV cache) to split, or `None`."""
    if len(shape) == 4 and shape[STATE_AXIS] % devices == 0:
        return STATE_AXIS
    return None


def units(paths: Sequence[str]) -> tuple[list[str], list[str]]:
    """The units of a model with parameters at `paths`, as sharding (FSDP)
    gathers them and pipelining places them: each element of the root's
    block arrays (`layers.0`, `layers.1`, ...) and each other block the root
    holds; and the array elements alone, which are worth recomputing.
    Parameters of the root block itself belong to none."""
    every: list[str] = []
    listed: list[str] = []
    for path in paths:
        parts = path.split(".")
        if len(parts) < 2:
            continue
        unit = f"{parts[0]}.{parts[1]}" if parts[1].isdigit() and len(parts) > 2 else parts[0]
        if unit not in every:
            every.append(unit)
            if unit != parts[0]:
                listed.append(unit)
    return every, listed


def pipeline_stages(sizes: Sequence[int], stages: int) -> list[int]:
    """The stage of each unit when units of `sizes` bytes, in execution
    order, are split into `stages` contiguous non-empty stages with the
    largest stage as small as it can be: what `linnet.torch.pipeline` runs
    and `linnet memory --pipeline-parallel` predicts."""
    if stages < 1:
        raise ValueError("a pipeline has at least one stage")
    if len(sizes) < stages:
        raise ValueError(f"{len(sizes)} blocks cannot fill {stages} stages")

    def greedy(limit: int) -> list[int] | None:
        assigned: list[int] = []
        stage, load = 0, 0
        for size in sizes:
            if load and load + size > limit:
                stage, load = stage + 1, 0
            if stage >= stages:
                return None
            assigned.append(stage)
            load += size
        return assigned

    low, high = max(sizes, default=0), max(sum(sizes), 1)
    while low < high:
        middle = (low + high) // 2
        if greedy(middle) is None:
            low = middle + 1
        else:
            high = middle
    assigned = greedy(low)
    assert assigned is not None
    # Fewer stages than asked: split the last stage that has more than one
    # unit, again and again, keeping the order.
    while assigned[-1] + 1 < stages:
        counts = [assigned.count(k) for k in range(assigned[-1] + 1)]
        widest = max(k for k, count in enumerate(counts) if count > 1)
        at = len(assigned) - 1 - assigned[::-1].index(widest)
        assigned = [a + 1 if i >= at else a for i, a in enumerate(assigned)]
    return assigned


def stage_starts(units: Sequence[str], assigned: Sequence[int]) -> list[str]:
    """The unit each stage after the first starts at."""
    return [units[i] for i in range(1, len(units)) if assigned[i] != assigned[i - 1]]


def assign_stages(units: Sequence[str], starts: Sequence[str]) -> list[int]:
    """Each unit's stage when stages after the first start at `starts`."""
    positions = [units.index(start) if start in units else -1 for start in starts]
    for start, position in zip(starts, positions, strict=True):
        if position < 0:
            raise ValueError(f"`{start}` is not a block of the model ({', '.join(units)})")
    if positions != sorted(set(positions)) or (positions and positions[0] == 0):
        raise ValueError("stage starts must be distinct blocks after the first, in order")
    return [sum(1 for p in positions if p <= i) for i in range(len(units))]


def stage_of_path(units: Sequence[str], assigned: Sequence[int], path: str) -> int | None:
    """The stage of the block holding `path` (a parameter, a state, or a
    block), or None when no block holds it."""
    for unit, stage in zip(units, assigned, strict=True):
        if path == unit or path.startswith(unit + "."):
            return stage
    return None


__all__ = [
    "DEFAULT_RULES",
    "ONE_SHOT_BYTES",
    "assign_stages",
    "pipeline_stages",
    "split_axis",
    "stage_of_path",
    "stage_starts",
    "state_axis",
    "units",
]
