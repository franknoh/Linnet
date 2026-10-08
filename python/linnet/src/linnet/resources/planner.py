"""Constraint solving over memory models: the largest batch, context or
cache that fits a device, and the feasibility of candidate configurations.

Memory never decreases as the batch, the sequence length or the cache
grows (every size is a sum of products of non-negative dimensions), so the
largest value that fits is found by doubling until it does not and then
bisecting: about two dozen analyses, each one sweep over the traced steps.

`ExecutionPlanner.plan` ranks layouts over several devices by the step
time `linnet.resources.performance` predicts from the device's measured
rates.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .. import nest
from ..compiler import LinnetError
from .analysis import MemoryModel, entry_roles
from .config import ExecutionConfig
from .performance import DeviceSpec, Throughput
from .result import MemoryAnalysisResult
from .training import CheckpointPolicy

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
    step: int = 1,
) -> FitResult:
    """The largest `target` (batch, context or kv-cache), a multiple of
    `step`, whose expected peak fits the constraint, the other roles held
    at the model's configuration. `model` must have been built with the
    target's role free."""
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

    def ok(multiple: int) -> tuple[bool, MemoryAnalysisResult | None, bool]:
        result = at(multiple * step)
        if result is None:
            return False, None, True
        return constraint.fits(result), result, False

    fits, result, blocked = ok(1)
    if not fits:
        return FitResult(target, None, result, constraint.budget, "model" if blocked else "memory")
    low, best = 1, result
    high = 2
    stopped: Literal["memory", "model", "search"] = "search"
    while high * step <= upper:
        fits, result, blocked = ok(high)
        if not fits:
            stopped = "model" if blocked else "memory"
            break
        low, best = high, result
        high *= 2
    else:
        return FitResult(target, low * step, best, constraint.budget, "search")
    # `low` fits and `high` does not.
    while high - low > 1:
        middle = (low + high) // 2
        fits, result, blocked = ok(middle)
        if fits:
            low, best = middle, result
        else:
            high = middle
            stopped = "model" if blocked else "memory"
    return FitResult(target, low * step, best, constraint.budget, stopped)


# A model for a configuration, with the roles given left free.
Builder = Callable[[ExecutionConfig, tuple[str, ...]], MemoryModel]


class ExecutionPlanner:
    """Configurations of one model checked against a device.

    Each configuration traces its entry once (`MemoryModel`), so a planner
    can compare backends, numerics, dtypes, checkpoint policies or sharding
    across candidates and maximize a size within each."""

    def __init__(
        self,
        model: str | Path | nest.Card,
        *,
        root: str | None = None,
        std_root: str | Path | None = None,
        build: Builder | None = None,
    ) -> None:
        """`model` as `MemoryModel` takes it; `build` makes each candidate's
        `MemoryModel` instead (another backend model, say)."""

        def default(config: ExecutionConfig, free: tuple[str, ...]) -> MemoryModel:
            return MemoryModel(model, config, root=root, std_root=std_root, free=free)

        self.model = model
        self.root = root
        self.std_root = std_root
        self.build: Builder = build or default

    def evaluate(self, config: ExecutionConfig) -> MemoryAnalysisResult:
        return self.build(config, ()).analyze()

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
        return fit(self.build(config, (role,)), constraint, target)

    def plan(
        self,
        config: ExecutionConfig,
        constraint: ResourceConstraint,
        device: DeviceSpec,
        devices: int = 1,
    ) -> ThroughputPlan:
        """The layouts of `devices` devices ranked by the predicted throughput
        each reaches (`linnet.resources.performance`): tensor- and
        pipeline-parallel degrees (powers of two), micro-batches, in
        training block checkpointing or none, the remaining devices as
        replicas. Within each layout the batch (or, for an entry without
        one, the sequence length) grows to the largest that fits, unless
        the configuration fixes it. A layout the analysis refuses or that
        does not fit is listed with why."""
        target: Target | None = None
        roles = entry_roles(self.model, config, root=self.root, std_root=self.std_root)
        if config.batch is None and "batch" in roles:
            target, config = "batch", replace(config, batch=1)
        elif config.context is None and "context" in roles:
            target, config = "context", replace(config, context=1)
        role = None if target is None else _ROLE[target]
        candidates: list[ThroughputCandidate] = []
        for tensor in _powers(devices):
            for stages in _powers(devices // tensor):
                replicas = devices // (tensor * stages)
                # As many micro-batches as stages at least: the variants set them.
                layout = replace(
                    config, tensor_parallel=tensor, pipeline_parallel=stages, microbatches=stages
                )
                try:
                    model = self.build(layout, () if role is None else (role,))
                except LinnetError as error:
                    candidates.append(
                        ThroughputCandidate(layout, replicas, None, None, None, str(error))
                    )
                    continue
                for variant in _variants(layout, stages):
                    candidates.append(
                        _evaluate(model, variant, constraint, device, replicas, target)
                    )
        candidates.sort(
            key=lambda c: -1.0 if c.throughput is None else c.throughput.tokens_per_second,
            reverse=True,
        )
        return ThroughputPlan(device, devices, target, tuple(candidates))


@dataclass(frozen=True, slots=True)
class ThroughputCandidate:
    """One layout: its configuration, the replicas of it the devices hold,
    the size it runs at, its analysis and predicted throughput; or why not."""

    config: ExecutionConfig
    replicas: int
    size: int | None
    result: MemoryAnalysisResult | None
    throughput: Throughput | None
    refused: str | None = None


@dataclass(frozen=True, slots=True)
class ThroughputPlan:
    """Layouts of a model over `devices` devices, the fastest first."""

    device: DeviceSpec
    devices: int
    target: Target | None
    candidates: tuple[ThroughputCandidate, ...]

    @property
    def best(self) -> ThroughputCandidate | None:
        first = self.candidates[0] if self.candidates else None
        return first if first is not None and first.throughput is not None else None


def _powers(limit: int) -> list[int]:
    """1, 2, 4, ... up to `limit`, those dividing it."""
    found: list[int] = []
    value = 1
    while value <= limit:
        if limit % value == 0:
            found.append(value)
        value *= 2
    return found


def _variants(config: ExecutionConfig, stages: int) -> list[ExecutionConfig]:
    """A layout's micro-batch counts and, in training, checkpoint policies."""
    counts = [1] if stages == 1 else [stages, 2 * stages, 4 * stages]
    policies: list[CheckpointPolicy | None] = (
        [None] if config.training is None else [CheckpointPolicy(), CheckpointPolicy("blocks")]
    )
    found: list[ExecutionConfig] = []
    for count in counts:
        for policy in policies:
            training = (
                config.training
                if policy is None or config.training is None
                else replace(config.training, checkpoint=policy)
            )
            found.append(replace(config, microbatches=count, training=training))
    return found


def _evaluate(
    model: MemoryModel,
    config: ExecutionConfig,
    constraint: ResourceConstraint,
    device: DeviceSpec,
    replicas: int,
    target: Target | None,
) -> ThroughputCandidate:
    """`config` on the trace `model` made (micro-batches and checkpointing
    change the analysis, not the trace)."""
    variant = copy.copy(model)
    variant.config = config
    try:
        if target is None:
            result = variant.analyze()
            if not constraint.fits(result):
                return ThroughputCandidate(config, replicas, None, result, None, "does not fit")
            return ThroughputCandidate(
                config, replicas, None, result, variant.throughput(device, replicas=replicas)
            )
        found = fit(variant, constraint, target, step=config.microbatches)
        if found.value is None or found.result is None:
            why = "the model's `where` clauses" if found.limited_by == "model" else "memory"
            return ThroughputCandidate(
                config, replicas, None, found.result, None, f"{why}: none fits"
            )
        # The largest size that fits is not always the fastest: attention's
        # cost per position grows with the length. Try the powers of two
        # below it as well.
        sizes = {found.value}
        size = config.microbatches
        while size < found.value:
            sizes.add(size)
            size *= 2
        best: tuple[int, Throughput] | None = None
        for size in sorted(sizes):
            values = {_ROLE[target]: size}
            if _ROLE[target] == "context" and config.cache is None:
                values["cache"] = size
            predicted = variant.throughput(device, replicas=replicas, **values)
            if best is None or predicted.tokens_per_second > best[1].tokens_per_second:
                best = (size, predicted)
        assert best is not None
        size, predicted = best
        result = found.result
        if size != found.value:
            values = {_ROLE[target]: size}
            if _ROLE[target] == "context" and config.cache is None:
                values["cache"] = size
            result = variant.analyze(**values)
        return ThroughputCandidate(config, replicas, size, result, predicted)
    except LinnetError as error:
        return ThroughputCandidate(config, replicas, None, None, None, str(error))
