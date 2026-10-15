"""Backends that run regions of a Linnet graph, and the planner that picks
them (docs/design/execution.md, docs/backends.md).

A backend registers itself with `register`, or from another package through
the `linnet.backends` entry point group: each entry point names a function
that takes nothing and returns a backend. A backend written for another
version of the contract (its `contract`, `CONTRACT` when it has none) is
left out, and `skipped` says why; so is one whose entry point fails to load.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from importlib import metadata
from typing import cast

from .contract import (
    Backend,
    Capabilities,
    Context,
    Cost,
    Executable,
    ExecutableMeta,
    Options,
    Region,
    Target,
    TensorSpec,
)
from .costs import Benchmarks, Measured, Timer
from .graph import Call, Graph, Operand, SegmentMask, Step, Value
from .planner import (
    CostModel,
    Estimates,
    Greedy,
    Partitioner,
    Plan,
    Planner,
    PlanStep,
    Proposal,
    runs_apart,
    runs_of,
    splittable,
)

# The version of the contract (`Backend`, `Executable`, `Region`) this
# package implements. A backend states the one it was written for as
# `contract`; one without it is taken as written for this one.
CONTRACT = 1

_registered: list[Backend[object]] = []
_skipped: dict[str, str] = {}


def register(backend: Backend[object]) -> bool:
    """Makes `backend` one `available` lists, unless it was written for
    another version of the contract; whether it did."""
    wanted: object = getattr(backend, "contract", CONTRACT)
    if wanted != CONTRACT:
        _skipped[backend.name] = f"written for contract {wanted}, this is {CONTRACT}"
        return False
    if all(b.name != backend.name for b in _registered):
        _registered.append(backend)
    return True


@functools.cache
def _load_entry_points() -> None:
    for point in metadata.entry_points(group="linnet.backends"):
        try:
            make = cast("Callable[[], Backend[object]]", point.load())
            register(make())
        except Exception as error:  # a broken plugin is left out, not fatal
            _skipped[point.name] = f"{point.value} failed to load: {error}"


def available() -> list[Backend[object]]:
    """Every backend registered here or by an installed package's
    `linnet.backends` entry point, in the order they came."""
    _load_entry_points()
    return list(_registered)


def skipped() -> dict[str, str]:
    """The backends left out, by name, and why."""
    _load_entry_points()
    return dict(_skipped)


__all__ = [
    "CONTRACT",
    "Backend",
    "Benchmarks",
    "Call",
    "Capabilities",
    "Context",
    "Cost",
    "CostModel",
    "Estimates",
    "Executable",
    "ExecutableMeta",
    "Graph",
    "Greedy",
    "Measured",
    "Operand",
    "Options",
    "Partitioner",
    "Plan",
    "PlanStep",
    "Planner",
    "Proposal",
    "Region",
    "SegmentMask",
    "Step",
    "Target",
    "TensorSpec",
    "Timer",
    "Value",
    "available",
    "register",
    "runs_apart",
    "runs_of",
    "skipped",
    "splittable",
]
