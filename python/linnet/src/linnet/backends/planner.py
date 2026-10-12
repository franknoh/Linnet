"""From a graph to an execution plan: which backend runs which range of
steps (docs/design/execution.md).

Partitioning and costing sit behind separate interfaces. A cost model
prices each region a backend proposes, and the same steps on the fallback;
a region that saves too little of the fallback's time is dropped. `Greedy`
takes the regions that save the most first, then, where the costs are not
known, the largest; the fallback backend runs every step no region took. A
plan runs its executables in order, each on the values the ones before it
made.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

from ..compiler import PlanError
from .contract import Backend, Context, Cost, Executable, Options, Region, Target
from .graph import Graph

T = TypeVar("T")

# Steps that open a span whose inside cannot run apart from its outside,
# and the step that closes each.
_SPANS = {
    "begin_while": "end_while",
    "begin_counted": "end_counted",
    "begin_custom_gradient": "end_custom_gradient",
    "begin_remat": "end_remat",
}


def _depths(graph: Graph) -> list[int]:
    """How many loops, ops with a `grad` and recomputed blocks are open at
    each step."""
    depth = 0
    out: list[int] = []
    for step in graph.steps:
        if step.op in _SPANS.values():
            depth -= 1
        out.append(depth)
        if step.op in _SPANS:
            depth += 1
    return out


def runs_apart(graph: Graph, region: Region) -> bool:
    """Whether `region` can run on its own: a range `splittable` says so of,
    or steps none of which lies inside a loop, an op's `grad` or a
    recomputed block."""
    if not region.members:
        return splittable(graph, region.start, region.stop)
    depths = _depths(graph)
    return all(depths[i] == 0 and graph.steps[i].op not in _SPANS for i in region.members)


def splittable(graph: Graph, start: int, stop: int) -> bool:
    """Whether `[start, stop)` can run on its own: every loop, op with a
    `grad` and recomputed block it touches lies wholly inside it."""
    open_spans: list[str] = []
    for step in graph.steps[start:stop]:
        if step.op in _SPANS:
            open_spans.append(_SPANS[step.op])
        elif step.op in _SPANS.values():
            if not open_spans or open_spans[-1] != step.op:
                return False
            open_spans.pop()
    return not open_spans


@dataclass(frozen=True, slots=True)
class Proposal(Generic[T]):
    """A region a backend offered to run: its cost, and the seconds it saves
    against the fallback, where both are known. `rank` is the backend's
    place among the planner's."""

    backend: Backend[T]
    region: Region
    cost: Cost | None = None
    saving: float | None = None
    rank: int = 0


class CostModel(Protocol[T]):
    def costs(
        self, asked: Sequence[tuple[Backend[T], Region]], target: Target, options: Options
    ) -> list[Cost | None]:
        """Each region's cost on its backend, None where unknown."""
        ...


class Estimates(Generic[T]):
    """A region's cost as its backend estimates it."""

    def costs(
        self, asked: Sequence[tuple[Backend[T], Region]], target: Target, options: Options
    ) -> list[Cost | None]:
        return [backend.estimate_cost(region, target) for backend, region in asked]


class Partitioner(Protocol[T]):
    def partition(
        self, graph: Graph, proposals: Sequence[Proposal[T]], memory: int | None
    ) -> list[Proposal[T]]:
        """The proposals taken: none overlapping, in step order, together
        keeping at most `memory` bytes (None: any)."""
        ...


class Greedy(Generic[T]):
    """The regions that save the most first; then, of those whose saving is
    not known, the largest, and of two the same size, the cheaper. A region
    that would keep more bytes than the budget has left is passed over."""

    def partition(
        self, graph: Graph, proposals: Sequence[Proposal[T]], memory: int | None = None
    ) -> list[Proposal[T]]:
        def order(p: Proposal[T]) -> tuple[int, float, float, int, int]:
            seconds = p.cost.seconds if p.cost is not None else float("inf")
            if p.saving is not None:
                return (0, -p.saving, seconds, p.rank, p.region.start)
            return (1, -len(p.region.indices), seconds, p.rank, p.region.start)

        taken: list[Proposal[T]] = []
        used: set[int] = set()
        left = memory
        for proposal in sorted(proposals, key=order):
            steps = set(proposal.region.indices)
            held = proposal.cost.bytes if proposal.cost is not None else 0
            if steps & used or (left is not None and held > left):
                continue
            taken.append(proposal)
            used |= steps
            if left is not None:
                left -= held
        return sorted(taken, key=lambda p: p.region.start)


def runs_of(region: Region) -> list[Region]:
    """The region as the fallback would run it: its ranges of steps, each a
    region of its own (one, unless its steps lie apart)."""
    if not region.members:
        return [region]
    runs: list[Region] = []
    first = previous = region.members[0]
    for i in (*region.members[1:], None):
        if i is not None and i == previous + 1:
            previous = i
            continue
        runs.append(Region.of(region.graph, first, previous + 1))
        if i is not None:
            first = previous = i
    return runs


@dataclass(frozen=True, slots=True)
class PlanStep(Generic[T]):
    """A region, the backend that runs it and the executable it made; the
    seconds it saves against the fallback, where known."""

    backend: str
    region: Region
    executable: Executable[T]
    saving: float | None = None


@dataclass(frozen=True, slots=True)
class Plan(Generic[T]):
    """Executables in step order, each over a region of `graph`."""

    graph: Graph
    target: Target
    steps: tuple[PlanStep[T], ...]

    def prepare(self, context: Context[T]) -> None:
        for step in self.steps:
            step.executable.prepare(context)

    def run(self, inputs: Sequence[T], context: Context[T]) -> list[T]:
        """The entry's results for `inputs`."""
        if len(inputs) != len(self.graph.inputs):
            raise PlanError(f"the plan takes {len(self.graph.inputs)} inputs, got {len(inputs)}")
        values: dict[int, T] = dict(zip(self.graph.inputs, inputs, strict=True))
        for step in self.steps:
            arguments = [values[value] for value in step.region.inputs]
            made = step.executable.execute(arguments, [], context)
            values.update(zip(step.region.outputs, made, strict=True))
        return [values[value] for value in self.graph.outputs]


def _check_order(graph: Graph, regions: Sequence[Region]) -> None:
    """Each region, in the order they run, reads only what the entry takes
    or an earlier region made."""
    made = set(graph.inputs)
    for region in regions:
        missing = set(region.inputs) - made
        if missing:
            raise PlanError(
                f"the region of steps {region.start} to {region.stop} reads values "
                f"{sorted(missing)} that no earlier region makes"
            )
        made.update(region.outputs)


class Planner(Generic[T]):
    """Plans a graph over `backends`, with `fallback` running whatever no
    region took. A backend whose `probe` declines the target is skipped."""

    def __init__(
        self,
        backends: Sequence[Backend[T]],
        fallback: Backend[T],
        partitioner: Partitioner[T] | None = None,
        cost_model: CostModel[T] | None = None,
    ) -> None:
        self.backends = list(backends)
        self.fallback = fallback
        self.partitioner: Partitioner[T] = partitioner or Greedy()
        self.cost_model: CostModel[T] = cost_model or Estimates()

    def proposals(
        self, graph: Graph, target: Target, options: Options | None = None
    ) -> list[Proposal[T]]:
        """Each region a backend that runs on `target` matched, priced with
        the same steps on the fallback. One that saves less than
        `options.margin` of the fallback's time is left out."""
        options = options or Options()
        offered: list[tuple[int, Backend[T], Region]] = []
        for rank, backend in enumerate(self.backends):
            if backend.probe(target) is None:
                continue
            offered += [
                (rank, backend, region)
                for region in backend.match(graph, target)
                if runs_apart(graph, region)
            ]
        # The fallback's runs of each region, each priced once.
        parts = [runs_of(region) for _, _, region in offered]
        runs: dict[tuple[int, int], Region] = {}
        for part in parts:
            for run in part:
                runs.setdefault((run.start, run.stop), run)
        asked: list[tuple[Backend[T], Region]] = [(b, r) for _, b, r in offered]
        asked += [(self.fallback, run) for run in runs.values()]
        priced = self.cost_model.costs(asked, target, options) if asked else []
        fallback = dict(zip(runs, priced[len(offered) :], strict=True))
        found: list[Proposal[T]] = []
        for (rank, backend, region), part, cost in zip(
            offered, parts, priced[: len(offered)], strict=True
        ):
            spans = [fallback[(run.start, run.stop)] for run in part]
            if cost is None or any(span is None for span in spans):
                found.append(Proposal(backend, region, cost, None, rank))
                continue
            instead = sum(span.seconds for span in spans if span is not None)
            if not math.isfinite(instead):
                # The fallback failed to run it alone: no saving is known.
                found.append(Proposal(backend, region, cost, None, rank))
                continue
            saving = instead - cost.seconds
            if saving > options.margin * instead:
                found.append(Proposal(backend, region, cost, saving, rank))
        return found

    def plan(self, graph: Graph, target: Target, options: Options | None = None) -> Plan[T]:
        if self.fallback.probe(target) is None:
            raise PlanError(f"the fallback backend `{self.fallback.name}` does not run on {target}")
        options = options or Options()
        taken = self.partitioner.partition(
            graph, self.proposals(graph, target, options), options.memory
        )
        regions: list[tuple[Backend[T], Region]] = [(p.backend, p.region) for p in taken]
        savings = {id(p.region): p.saving for p in taken}
        # The fallback runs each run of steps no region took.
        used = {i for p in taken for i in p.region.indices}
        start = 0
        for i in range(len(graph.steps) + 1):
            if i == len(graph.steps) or i in used:
                regions += self._fill(graph, start, i)
                start = i + 1
        regions.sort(key=lambda pair: pair[1].start)
        _check_order(graph, [region for _, region in regions])
        # A backend that compiles several regions at once (`compile_regions`)
        # gets all of its own together.
        compiled: dict[int, Executable[T]] = {}
        mine: dict[int, list[int]] = {}
        for i, (backend, _) in enumerate(regions):
            mine.setdefault(id(backend), []).append(i)
        for indices in mine.values():
            backend = regions[indices[0]][0]
            together = cast(
                "Callable[[list[Region], Target, Options], Sequence[Executable[T]]] | None",
                getattr(backend, "compile_regions", None),
            )
            if together is not None and len(indices) > 1:
                made = together([regions[i][1] for i in indices], target, options)
                compiled.update(zip(indices, made, strict=True))
                continue
            for i in indices:
                compiled[i] = backend.compile(regions[i][1], target, options)
        steps = tuple(
            PlanStep(backend.name, region, compiled[i], savings.get(id(region)))
            for i, (backend, region) in enumerate(regions)
        )
        return Plan(graph, target, steps)

    def _fill(self, graph: Graph, start: int, stop: int) -> Iterable[tuple[Backend[T], Region]]:
        """The fallback's region over the steps `[start, stop)`, if they
        compute anything."""
        if start >= stop:
            return []
        if not splittable(graph, start, stop):
            raise PlanError(f"steps {start} to {stop} cut through a loop or an op's gradient")
        region = Region.of(graph, start, stop)
        computes = any(step.results for step in region.steps)
        return [(self.fallback, region)] if computes or region.whole else []
