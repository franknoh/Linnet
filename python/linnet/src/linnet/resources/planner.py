"""Constraint solving over memory models: the largest batch, context or
cache that fits a device, and the feasibility of candidate configurations.

Memory never decreases as the batch, the sequence length or the cache
grows (every size is a sum of products of non-negative dimensions), so the
largest value that fits is found by doubling until it does not and then
bisecting: about two dozen analyses, each one sweep over the traced steps.

Throughput is not modeled yet. `ExecutionPlanner.plan` says so rather than
ranking configurations by a number nothing measured.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Literal

from .analysis import MemoryModel, Source
from .config import ExecutionConfig
from .result import MemoryAnalysisResult

Target = Literal["batch", "context", "kv-cache"]
_ROLE: dict[str, str] = {"batch": "batch", "context": "context", "kv-cache": "cache"}


@dataclass(frozen=True, slots=True)
class ResourceConstraint:
    """A device's memory, less a safety margin: `reserve` bytes and
    `reserve_percent` of the device."""

    device_memory: int
    reserve: int = 0
    reserve_percent: float = 0.0

    @property
    def budget(self) -> int:
        return int(
            self.device_memory - self.reserve - self.device_memory * self.reserve_percent / 100
        )

    def fits(self, result: MemoryAnalysisResult) -> bool:
        return result.expected_peak <= self.budget


@dataclass(frozen=True, slots=True)
class FitResult:
    """The largest value of `target` that fits, the analysis there, and the
    budget left. `value` is None when not even 1 fits; `limited_by` says
    whether memory or the model's own `where` clauses stopped the search."""

    target: str
    value: int | None
    result: MemoryAnalysisResult | None
    budget: int
    limited_by: Literal["memory", "model", "search"]

    @property
    def headroom(self) -> int | None:
        return None if self.result is None else self.budget - self.result.expected_peak


def fit(
    model: MemoryModel,
    constraint: ResourceConstraint,
    target: Target,
    upper: int = 1 << 24,
) -> FitResult:
    """The largest `target` (batch, context or kv-cache) whose expected peak
    fits the constraint, the other roles held at the model's configuration.
    `model` must have been built with the target's role free."""
    role = _ROLE[target]
    if role not in model.free:
        raise ValueError(f"build the model with `{role}` free to fit it")

    def at(value: int) -> MemoryAnalysisResult | None:
        values: dict[str, int | None] = {"batch": None, "context": None, "cache": None}
        values[role] = value
        if role == "context" and model.config.cache is None:
            values["cache"] = value
        env = model.env(values["batch"], values["context"], values["cache"])
        if not model.satisfied(env):
            return None
        return model.analyze(values["batch"], values["context"], values["cache"])

    def ok(value: int) -> tuple[bool, MemoryAnalysisResult | None, bool]:
        result = at(value)
        if result is None:
            return False, None, True
        return constraint.fits(result), result, False

    fits, result, blocked = ok(1)
    if not fits:
        return FitResult(target, None, result, constraint.budget, "model" if blocked else "memory")
    low, best = 1, result
    high = 2
    stopped: Literal["memory", "model", "search"] = "search"
    while high <= upper:
        fits, result, blocked = ok(high)
        if not fits:
            stopped = "model" if blocked else "memory"
            break
        low, best = high, result
        high *= 2
    else:
        return FitResult(target, low, best, constraint.budget, "search")
    # `low` fits and `high` does not.
    while high - low > 1:
        middle = (low + high) // 2
        fits, result, blocked = ok(middle)
        if fits:
            low, best = middle, result
        else:
            high = middle
            stopped = "model" if blocked else "memory"
    return FitResult(target, low, best, constraint.budget, stopped)


Builder = Callable[[Source, ExecutionConfig, tuple[str, ...]], MemoryModel]


def _build(source: Source, config: ExecutionConfig, free: tuple[str, ...]) -> MemoryModel:
    return MemoryModel(source, config, free)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A configuration and whether it fits."""

    config: ExecutionConfig
    result: MemoryAnalysisResult
    fits: bool


class ExecutionPlanner:
    """Configurations of one model checked against a device.

    Each configuration traces its entry once (`MemoryModel`), so a planner
    can compare backends, numerics, dtypes, checkpoint policies or sharding
    across candidates and maximize a size within each."""

    def __init__(self, source: Source, build: Builder | None = None) -> None:
        self.source = source
        self.build: Builder = build or _build

    def evaluate(self, config: ExecutionConfig) -> MemoryAnalysisResult:
        return self.build(self.source, config, ()).analyze()

    def feasible(
        self, configs: Iterable[ExecutionConfig], constraint: ResourceConstraint
    ) -> list[Candidate]:
        out: list[Candidate] = []
        for config in configs:
            result = self.evaluate(config)
            out.append(Candidate(config, result, constraint.fits(result)))
        return out

    def maximize(
        self, config: ExecutionConfig, constraint: ResourceConstraint, target: Target
    ) -> FitResult:
        role = _ROLE[target]
        if role == "batch" and config.batch is None:
            config = replace(config, batch=1)
        if role == "context" and config.context is None:
            config = replace(config, context=1)
        if role == "cache" and config.cache is None:
            config = replace(config, cache=1)
        return fit(self.build(self.source, config, (role,)), constraint, target)

    def plan(
        self, config: ExecutionConfig, constraint: ResourceConstraint, objective: str
    ) -> FitResult:
        """Not yet: throughput needs a performance model, which Linnet does
        not have. Memory-feasible configurations come from `feasible` and
        `maximize`."""
        raise NotImplementedError(
            f"no performance model to {objective} with yet; use `maximize` for the largest "
            "batch, context or cache that fits"
        )
