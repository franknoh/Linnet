"""A pipeline's stages in the memory analysis: the split
`linnet.torch.pipeline` runs, each stage's part of the graph, and what
crosses between stages.

A step that reads a parameter (or a value computed from one) runs on that
parameter's stage, or the latest of its operands'; a step of the inputs
alone runs on every stage that reads its result. So only values computed
from weights cross, as `linnet.torch.stages` splits the generated source.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..parallel import stage_of_path
from .graph import TensorGraph, restrict


@dataclass(frozen=True, slots=True)
class StagePart:
    """One stage: its blocks, its part of the graph (renumbered), and the
    objects (by their ids in the whole graph) it receives and sends."""

    stage: int
    units: tuple[str, ...]
    graph: TensorGraph
    receives: tuple[int, ...]
    sends: tuple[int, ...]


def split_graph(
    graph: TensorGraph, units: Sequence[str], assigned: Sequence[int]
) -> list[StagePart]:
    """`graph` as the stages that run it when unit `units[i]` is on stage
    `assigned[i]`."""
    stages = max(assigned, default=0) + 1
    last = stages - 1
    objects = graph.objects
    anchored: dict[int, int] = {}  # storage owner -> the stage that makes it
    placed: list[set[int]] = [set() for _ in graph.steps]
    bound: list[bool] = [False] * len(graph.steps)
    for step in graph.steps:
        anchors: list[int] = []
        for id in step.inputs:
            owner = objects[objects[id].storage]
            if owner.persistent and owner.path is not None:
                held = stage_of_path(units, assigned, owner.path)
                if held is not None:
                    anchors.append(held)
            elif owner.id in anchored:
                anchors.append(anchored[owner.id])
        if anchors:
            stage = max(anchors)
            placed[step.index] = {stage}
            bound[step.index] = True
            for out in step.outputs:
                anchored[objects[out].storage] = stage

    readers: dict[int, list[int]] = {}
    for step in graph.steps:
        for id in step.inputs:
            readers.setdefault(objects[id].storage, []).append(step.index)
    results = {objects[i].storage for i in graph.outputs}
    # Values of the inputs alone go to every stage that reads them, latest
    # step first so that its readers are placed.
    for step in reversed(graph.steps):
        if bound[step.index]:
            continue
        wanted: set[int] = set()
        for out in step.outputs:
            owner = objects[out].storage
            for reader in readers.get(owner, []):
                wanted |= placed[reader]
            if owner in results:
                wanted.add(last)
        placed[step.index] = wanted

    reach: dict[int, int] = {}  # anchored owner -> the latest stage that reads it
    for owner, made in anchored.items():
        stages_reading = [s for r in readers.get(owner, []) for s in placed[r]]
        if owner in results:
            stages_reading.append(last)
        reach[owner] = max([made, *stages_reading])

    parts: list[StagePart] = []
    entry_inputs = {objects[i].storage for i in graph.inputs}
    for stage in range(stages):
        steps = [step.index for step in graph.steps if stage in placed[step.index]]
        receives = sorted(o for o, made in anchored.items() if made < stage <= reach[o])
        sends = (
            sorted(objects[i].storage for i in graph.outputs)
            if stage == last
            else sorted(o for o, made in anchored.items() if made <= stage < reach[o])
        )
        read = {objects[i].storage for s in steps for i in graph.steps[s].inputs}
        inputs = [i for i in graph.inputs if objects[i].storage in read & entry_inputs]
        parts.append(
            StagePart(
                stage,
                tuple(u for u, s in zip(units, assigned, strict=True) if s == stage),
                restrict(graph, steps, [*inputs, *receives], sends),
                tuple(receives),
                tuple(sends),
            )
        )
    return parts


__all__ = ["StagePart", "split_graph"]
