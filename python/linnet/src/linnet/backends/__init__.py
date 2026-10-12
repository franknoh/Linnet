"""Backends that run regions of a Linnet graph, and the planner that picks
them (docs/design/execution.md).

A backend registers itself with `register`, or from another package through
the `linnet.backends` entry point group: each entry point names a function
that takes nothing and returns a backend.
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
from .graph import Call, Graph, Operand, Step, Value
from .planner import Estimates, Greedy, Plan, Planner, PlanStep, Proposal, splittable

_registered: list[Backend[object]] = []


def register(backend: Backend[object]) -> None:
    """Makes `backend` one `available` lists."""
    if all(b.name != backend.name for b in _registered):
        _registered.append(backend)


@functools.cache
def _load_entry_points() -> None:
    for point in metadata.entry_points(group="linnet.backends"):
        make = cast("Callable[[], Backend[object]]", point.load())
        register(make())


def available() -> list[Backend[object]]:
    """Every backend registered here or by an installed package's
    `linnet.backends` entry point, in the order they came."""
    _load_entry_points()
    return list(_registered)


__all__ = [
    "Backend",
    "Call",
    "Capabilities",
    "Context",
    "Cost",
    "Estimates",
    "Executable",
    "ExecutableMeta",
    "Graph",
    "Greedy",
    "Operand",
    "Options",
    "Plan",
    "PlanStep",
    "Planner",
    "Proposal",
    "Region",
    "Step",
    "Target",
    "TensorSpec",
    "Value",
    "available",
    "register",
    "splittable",
]
