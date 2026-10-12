"""From a graph to an execution plan: which backend runs which range of
steps (docs/design/execution.md).

Partitioning and costing sit behind separate interfaces. `Greedy` takes the
largest proposed regions first, the cheaper of two the same size; the
fallback backend runs every step no region took. A plan runs its
executables in order, each on the values the ones before it made.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from ..compiler import PlanError
from .contract import Backend, Context, Executable, Options, Region, Target
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
    """A region a backend offered to run, at a cost (seconds, or a rank)."""

    backend: Backend[T]
    region: Region
    cost: float


class CostModel(Protocol[T]):
    def cost(self, backend: Backend[T], region: Region, target: Target) -> float: ...


class Estimates(Generic[T]):
    """A region's cost as its backend estimates it; a backend without an
    estimate ranks by the order it was given in."""

    def __init__(self, backends: Sequence[Backend[T]]) -> None:
        self._rank = {id(backend): i for i, backend in enumerate(backends)}

    def cost(self, backend: Backend[T], region: Region, target: Target) -> float:
        estimate = backend.estimate_cost(region, target)
        if estimate is not None:
            return estimate.seconds
        return float(self._rank.get(id(backend), len(self._rank)))


class Partitioner(Protocol[T]):
    def partition(self, graph: Graph, proposals: Sequence[Proposal[T]]) -> list[Proposal[T]]:
        """The proposals taken: none overlapping, in step order."""
        ...


class Greedy(Generic[T]):
    """The largest regions first; of two the same size, the cheaper."""

    def partition(self, graph: Graph, proposals: Sequence[Proposal[T]]) -> list[Proposal[T]]:
        taken: list[Proposal[T]] = []
        ordered = sorted(
            proposals, key=lambda p: (-(p.region.stop - p.region.start), p.cost, p.region.start)
        )
        for proposal in ordered:
            region = proposal.region
            if all(region.stop <= t.region.start or t.region.stop <= region.start for t in taken):
                taken.append(proposal)
        return sorted(taken, key=lambda p: p.region.start)


@dataclass(frozen=True, slots=True)
class PlanStep(Generic[T]):
    backend: str
    region: Region
    executable: Executable[T]


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
        self.cost_model: CostModel[T] = cost_model or Estimates([*self.backends, fallback])

    def proposals(self, graph: Graph, target: Target) -> list[Proposal[T]]:
        found: list[Proposal[T]] = []
        for backend in self.backends:
            if backend.probe(target) is None:
                continue
            for region in backend.match(graph, target):
                if splittable(graph, region.start, region.stop):
                    found.append(
                        Proposal(backend, region, self.cost_model.cost(backend, region, target))
                    )
        return found

    def plan(self, graph: Graph, target: Target, options: Options | None = None) -> Plan[T]:
        if self.fallback.probe(target) is None:
            raise PlanError(f"the fallback backend `{self.fallback.name}` does not run on {target}")
        options = options or Options()
        taken = self.partitioner.partition(graph, self.proposals(graph, target))
        regions: list[tuple[Backend[T], Region]] = []
        at = 0
        for proposal in taken:
            regions += self._fill(graph, at, proposal.region.start)
            regions.append((proposal.backend, proposal.region))
            at = proposal.region.stop
        regions += self._fill(graph, at, len(graph.steps))
        steps = tuple(
            PlanStep(backend.name, region, backend.compile(region, target, options))
            for backend, region in regions
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
