"""What an analysis reports: components with their confidence, the peaks,
and everything left unknown, as values a program reads rather than text it
parses. `to_dict` is the JSON document `linnet memory --json` prints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..sizes import format_bytes
from .graph import Category, Confidence, weakest


@dataclass(frozen=True, slots=True)
class MemoryComponent:
    """One line of the breakdown. `nbytes` is None when unknown; `formula`
    is the bytes as a function of the free symbols, when it has one."""

    name: str
    category: Category
    nbytes: int | None
    confidence: Confidence
    formula: str | None = None
    note: str = ""
    device: str = "device:0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "bytes": self.nbytes,
            "confidence": self.confidence.value,
            "formula": self.formula,
            "note": self.note,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class Allocation:
    """Transient storage three ways: a buffer per tensor (`naive`), packed
    by a static planner that reuses dead tensors' storage (`planned`), and
    the live peak no plan can beat (`lower_bound`)."""

    naive: int
    planned: int
    lower_bound: int


@dataclass(frozen=True, slots=True)
class MemoryAnalysisResult:
    """A model's memory under one configuration, per device.

    `graph_peak` is what the program's own tensors need at their peak:
    parameters, buffers, state, caches and the largest set of live
    activations (in training, also gradients, master weights and optimizer
    states). `expected_peak` adds what the backend model and the runtime
    estimates account for. Components whose size is unknown are listed in
    `unknown` and counted in neither."""

    configuration: Mapping[str, Any]
    components: tuple[MemoryComponent, ...]
    graph_peak: int
    expected_peak: int
    peak_at: str
    symbols: Mapping[str, int]
    unknown: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    formulas: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    allocation: Allocation | None = None
    recompute: float | None = None

    @property
    def confidence(self) -> Confidence:
        """The weakest confidence among the components counted."""
        return weakest(c.confidence for c in self.components if c.nbytes is not None)

    def component(self, name: str) -> MemoryComponent:
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(name)

    def total(self, category: Category) -> int:
        return sum(c.nbytes or 0 for c in self.components if c.category == category)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "configuration": dict(self.configuration),
            "symbols": dict(self.symbols),
            "components": [c.to_dict() for c in self.components],
            "peak": {
                "graph": self.graph_peak,
                "expected": self.expected_peak,
                "at": self.peak_at,
                "confidence": self.confidence.value,
            },
            "allocation": None
            if self.allocation is None
            else {
                "naive": self.allocation.naive,
                "planned": self.allocation.planned,
                "lower_bound": self.allocation.lower_bound,
            },
            "recompute": self.recompute,
            "formulas": dict(self.formulas),
            "unknown": list(self.unknown),
            "warnings": list(self.warnings),
            "assumptions": list(self.assumptions),
        }


def format_result(result: MemoryAnalysisResult) -> str:
    """The breakdown as `linnet memory` prints it."""
    width = max([len(c.name) for c in result.components] + [20])
    lines: list[str] = []
    for component in result.components:
        if component.nbytes is None:
            continue
        lines.append(
            f"{component.name:<{width}}  {format_bytes(component.nbytes):>12}  "
            f"[{component.confidence.value}]"
        )
    lines.append("-" * (width + 16))
    lines.append(f"{'Graph peak':<{width}}  {format_bytes(result.graph_peak):>12}")
    lines.append(
        f"{'Expected peak':<{width}}  {format_bytes(result.expected_peak):>12}  "
        f"[{result.confidence.value}]"
    )
    lines.append(f"\nPeak at {result.peak_at}.")
    if result.allocation is not None:
        a = result.allocation
        lines.append(
            f"Transient storage: {format_bytes(a.naive)} with a buffer per tensor, "
            f"{format_bytes(a.planned)} planned, {format_bytes(a.lower_bound)} live at most."
        )
    if result.recompute is not None:
        lines.append(f"Recomputation: +{result.recompute * 100:.1f}% compute (estimated).")
    if result.unknown:
        lines.append("\nNot included (unknown):")
        lines.extend(f"  - {item}" for item in result.unknown)
    if result.warnings:
        lines.append("\nWarnings:")
        lines.extend(f"  - {item}" for item in result.warnings)
    return "\n".join(lines)
