"""Costs measured by running each region (docs/design/execution.md).

`Measured` prices a region by timing its executable on real values: a run
of the graph by a reference backend, cut at the boundary of every region to
price, gives each region its inputs. Regions of one kind (`Region.kind`:
each layer's) are timed once, and `Benchmarks` keeps the times by device,
backend and kind, so that the next plan reads them instead.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Generic, TypeVar, cast

from ..compiler import PlanError, write_atomically
from ..home import home
from .contract import Backend, Context, Cost, Executable, ExecutableMeta, Options, Region, Target
from .planner import runs_of, splittable

T = TypeVar("T")

# Seconds one call of a function takes, given its executable's metadata
# (whether a CUDA graph may record it, say).
Timer = Callable[[Callable[[], object], ExecutableMeta], float]


class Benchmarks:
    """Measured costs by key, kept in the JSON file at `path` (None: in
    memory only)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._costs: dict[str, Cost] = self._read()
        self._measured: dict[str, Cost] = {}

    @classmethod
    def default(cls) -> Benchmarks:
        """The costs measured on this machine, in `$LINNET_HOME`."""
        return cls(home() / "benchmarks.json")

    def get(self, key: str) -> Cost | None:
        return self._costs.get(key)

    def put(self, key: str, cost: Cost) -> None:
        self._costs[key] = cost
        self._measured[key] = cost

    def __len__(self) -> int:
        return len(self._costs)

    def save(self) -> None:
        """Writes what was measured here over what the file holds now, which
        another process may have added to."""
        if self.path is None or not self._measured:
            return
        merged = {**self._read(), **self._measured}
        document = {key: [cost.seconds, cost.bytes] for key, cost in merged.items()}
        write_atomically(self.path, json.dumps(document, indent=0, sort_keys=True) + "\n")
        self._costs.update(merged)

    def _read(self) -> dict[str, Cost]:
        if self.path is None or not self.path.exists():
            return {}
        try:
            loaded: object = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}  # a damaged file: measured again
        if not isinstance(loaded, dict):
            return {}
        costs: dict[str, Cost] = {}
        for key, value in cast("dict[str, object]", loaded).items():
            if isinstance(value, list):
                seconds, held = cast("list[object]", value)[:2]
                if isinstance(seconds, int | float) and isinstance(held, int):
                    costs[key] = Cost(float(seconds), held)
        return costs


def key_of(backend: Backend[T], region: Region, target: Target, options: Options) -> str:
    """Where `Benchmarks` keeps `region`'s cost on `backend`."""
    device = f"{target.kind}:{target.name}:{target.capability[0]}.{target.capability[1]}"
    return "|".join(
        (
            device,
            f"processes={target.processes}",
            backend.name,
            backend.version,
            options.numerics,
            region.kind,
        )
    )


class Measured(Generic[T]):
    """Costs measured by timing (`time`) each kind of region on its backend,
    on the values `reference` gives the region's inputs when it runs the
    graph on `inputs` with `context`. A region that fails to compile or run
    costs infinitely much."""

    def __init__(
        self,
        inputs: Sequence[T],
        context: Context[T],
        reference: Backend[T],
        time: Timer,
        benchmarks: Benchmarks | None = None,
    ) -> None:
        self.inputs = list(inputs)
        self.context = context
        self.reference = reference
        self.time = time
        self.benchmarks = benchmarks if benchmarks is not None else Benchmarks()
        self.failures: dict[str, str] = {}

    def costs(
        self, asked: Sequence[tuple[Backend[T], Region]], target: Target, options: Options
    ) -> list[Cost | None]:
        keys = [key_of(backend, region, target, options) for backend, region in asked]
        missing: dict[str, tuple[Backend[T], Region]] = {}
        for key, pair in zip(keys, asked, strict=True):
            if self.benchmarks.get(key) is None and key not in self.failures:
                missing.setdefault(key, pair)
        if missing:
            self._measure(missing, target, options)
            self.benchmarks.save()
        return [
            Cost(math.inf) if key in self.failures else self.benchmarks.get(key) for key in keys
        ]

    def _measure(
        self, missing: dict[str, tuple[Backend[T], Region]], target: Target, options: Options
    ) -> None:
        values = self._values([region for _, region in missing.values()], target, options)
        by_backend: dict[int, list[str]] = {}
        for key, (backend, _) in missing.items():
            by_backend.setdefault(id(backend), []).append(key)
        for keys in by_backend.values():
            backend = missing[keys[0]][0]
            regions = [missing[key][1] for key in keys]
            for key, made in zip(keys, _compile(backend, regions, target, options), strict=True):
                if isinstance(made, str):
                    self.failures[key] = made
                    continue
                region = missing[key][1]
                try:
                    made.prepare(self.context)
                    arguments = [values[value] for value in region.inputs]

                    def run(
                        executable: Executable[T] = made, arguments: list[T] = arguments
                    ) -> object:
                        return executable.execute(arguments, [], self.context)

                    seconds = self.time(run, made.meta)
                except Exception as error:  # the cost says it failed
                    self.failures[key] = repr(error)
                    continue
                self.benchmarks.put(key, Cost(seconds, made.meta.holds))

    def _values(self, regions: Sequence[Region], target: Target, options: Options) -> dict[int, T]:
        """Every value a region of `regions` reads, as the reference makes
        it: the graph run in ranges cut at each region's boundaries."""
        graph = regions[0].graph
        cuts = {0, len(graph.steps)}
        for region in regions:
            for run in runs_of(region):
                cuts |= {run.start, run.stop}
        ordered = sorted(cuts)
        ranges: list[Region] = []
        for start, stop in itertools.pairwise(ordered):
            # Every region runs apart, so no cut falls inside a loop.
            if not splittable(graph, start, stop):
                raise PlanError(f"steps {start} to {stop} cut through a loop or an op's gradient")
            region = Region.of(graph, start, stop)
            if region.whole or any(step.results for step in region.steps):
                ranges.append(region)
        values: dict[int, T] = dict(zip(graph.inputs, self.inputs, strict=True))
        for region, made in zip(
            ranges, _compile(self.reference, ranges, target, options), strict=True
        ):
            if isinstance(made, str):
                where = f"steps {region.start} to {region.stop}"
                raise PlanError(f"`{self.reference.name}` cannot run {where}: {made}")
            outputs = made.execute([values[value] for value in region.inputs], [], self.context)
            values.update(zip(region.outputs, outputs, strict=True))
        return values


def _compile(
    backend: Backend[T], regions: Sequence[Region], target: Target, options: Options
) -> list[Executable[T] | str]:
    """Each region compiled by `backend`, together where it can, or why it
    could not be."""
    together = cast(
        "Callable[[list[Region], Target, Options], Sequence[Executable[T]]] | None",
        getattr(backend, "compile_regions", None),
    )
    if together is not None and len(regions) > 1:
        try:
            return list(together(list(regions), target, options))
        except Exception:  # each on its own below, to find which fails
            pass
    made: list[Executable[T] | str] = []
    for region in regions:
        try:
            made.append(backend.compile(region, target, options))
        except Exception as error:  # the cost says it failed
            made.append(repr(error))
    return made


__all__ = ["Benchmarks", "Measured", "Timer", "key_of"]
