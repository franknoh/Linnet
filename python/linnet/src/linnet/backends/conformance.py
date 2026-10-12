"""Whether a backend computes what Linnet's own code computes
(docs/backends.md).

`conform` plans a graph with the backend and the fallback, and runs the
plan with the fallback's executable for every region, so each region sees
the values Linnet's own code makes. At each region the backend took, it
also runs the backend's executable on the same inputs and measures how far
its outputs are from the fallback's. A region that writes state runs once,
by the fallback, and is reported as not compared.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from .contract import Backend, Context, Executable, Options, Region, Target
from .graph import Graph
from .planner import Planner

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Finding(Generic[T]):
    """One region the backend took: how far its outputs fell from the
    fallback's (the largest of `difference`'s measures over them), whether
    that is within the tolerance, and what went wrong if it failed."""

    region: Region
    difference: float
    passed: bool
    error: str | None = None


def conform(
    graph: Graph,
    target: Target,
    backend: Backend[T],
    fallback: Backend[T],
    inputs: Sequence[T],
    context: Context[T],
    difference: Callable[[T, T], float],
    tolerance: float,
    options: Options | None = None,
) -> list[Finding[T]]:
    """Each region `backend` takes in `graph`, run by it and by `fallback` on
    the same inputs; `difference(got, expected)` measures one output."""
    options = options or Options()
    plan = Planner([backend], fallback).plan(graph, target, options)
    taken = [i for i, step in enumerate(plan.steps) if step.backend == backend.name]
    regions = [plan.steps[i].region for i in taken]
    # The fallback's executables for the same regions, together where it can.
    together = cast(
        "Callable[[list[Region], Target, Options], Sequence[Executable[T]]] | None",
        getattr(fallback, "compile_regions", None),
    )
    if together is not None and regions:
        made = together(regions, target, options)
    else:
        made = [fallback.compile(region, target, options) for region in regions]
    reference = dict(zip(taken, made, strict=True))
    values: dict[int, T] = dict(zip(graph.inputs, inputs, strict=True))
    findings: list[Finding[T]] = []
    for i, step in enumerate(plan.steps):
        arguments = [values[value] for value in step.region.inputs]
        expected = (reference.get(i) or step.executable).execute(arguments, [], context)
        values.update(zip(step.region.outputs, expected, strict=True))
        if i not in reference:
            continue
        if step.executable.meta.effects == "state" or step.region.next_states:
            findings.append(Finding(step.region, 0.0, True, "writes state: not compared"))
            continue
        try:
            got = step.executable.execute(arguments, [], context)
        except Exception as error:  # the finding reports it
            findings.append(Finding(step.region, float("inf"), False, repr(error)))
            continue
        if len(got) != len(expected):
            findings.append(
                Finding(
                    step.region,
                    float("inf"),
                    False,
                    f"made {len(got)} outputs, the fallback {len(expected)}",
                )
            )
            continue
        worst = max((difference(a, b) for a, b in zip(got, expected, strict=True)), default=0.0)
        findings.append(Finding(step.region, worst, worst <= tolerance))
    return findings
