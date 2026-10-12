"""The contract between the planner and the backends that run graph regions
(docs/design/execution.md).

A backend takes a region of any size, from one step to the whole graph,
through the same four methods; only the regions its `match` returns differ.
`T` is the tensor type a backend's executables take and return, such as
`torch.Tensor`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeVar

from .graph import SOURCES, STRUCTURE, Graph, Step

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Target:
    """Where a plan runs: one device, and the processes a split model spans."""

    kind: str  # `cuda`, `cpu`, ...
    index: int = 0
    name: str = ""
    capability: tuple[int, int] = (0, 0)  # a CUDA device's compute capability
    memory: int = 0  # bytes
    processes: int = 1


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a backend runs on a target: its libraries' versions, and the
    dtypes it takes (None: every one)."""

    libraries: Mapping[str, str] = field(default_factory=dict[str, str])
    dtypes: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class Cost:
    """A region's time on a backend, and the bytes it moves."""

    seconds: float
    bytes: int = 0


@dataclass(frozen=True, slots=True)
class Options:
    """How to compile: the numerics the program allows."""

    numerics: str = "equivalent"


@dataclass(frozen=True, slots=True)
class Region:
    """The steps `[start, stop)` of a graph, which one backend runs, and the
    values that cross its boundary. Parameters and states are not inputs:
    they persist between calls, and an executable reads them by path."""

    graph: Graph
    start: int
    stop: int
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    parameters: tuple[str, ...]
    states: tuple[str, ...]
    call: int | None = None  # the call whose range it is, if any

    @property
    def steps(self) -> Sequence[Step]:
        return self.graph.steps[self.start : self.stop]

    @property
    def whole(self) -> bool:
        return self.start == 0 and self.stop == len(self.graph.steps)

    @classmethod
    def of(cls, graph: Graph, start: int, stop: int, call: int | None = None) -> Region:
        """The range's region: its inputs are the values its steps read that
        steps before it made, in order of first use; its outputs the values
        it makes that later steps read, or that the entry returns."""
        persistent = {value: path for path, value in graph.parameters.items()}
        state = {value: path for path, value in graph.states.items()}
        made: set[int] = set()
        inputs: dict[int, None] = {}
        parameters: dict[str, None] = {}
        states: dict[str, None] = {}
        for step in graph.steps[start:stop]:
            if step.op in SOURCES:
                continue  # what the entry takes, which no region makes
            if step.op not in STRUCTURE:
                for operand in step.operands:
                    value = operand.value
                    if value is None or value in made:
                        continue
                    if value in persistent:
                        parameters[persistent[value]] = None
                    elif value in state:
                        states[state[value]] = None
                    else:
                        inputs[value] = None
            made.update(step.results)
        read_after: set[int] = {
            operand.value
            for step in graph.steps[stop:]
            if step.op not in STRUCTURE or step.op == "finish"
            for operand in step.operands
            if operand.value is not None
        }
        outputs = tuple(
            value
            for step in graph.steps[start:stop]
            if step.op not in SOURCES
            for value in step.results
            if value in read_after
        )
        if start == 0 and stop == len(graph.steps):
            inputs = dict.fromkeys(graph.inputs)
            outputs = graph.outputs
        return cls(
            graph,
            start,
            stop,
            tuple(inputs),
            outputs,
            tuple(parameters),
            tuple(states),
            call,
        )


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """A tensor an executable takes or returns."""

    shape: tuple[int, ...]
    dtype: str
    layout: str = "contiguous"  # or strides, or a layout the backend names
    device: int = 0  # a slot of the placement


@dataclass(frozen=True, slots=True)
class ExecutableMeta:
    """What the planner and the runtime need to know about an executable."""

    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    # Whether `execute` makes its outputs and returns them; otherwise it
    # writes the ones it is given.
    allocates: bool = True
    aliases: Mapping[int, int] = field(default_factory=dict[int, int])  # output -> input
    workspace: int = 0  # scratch bytes
    capturable: bool = True  # a CUDA graph may record it
    stream: Literal["caller", "own"] = "caller"
    host: Literal["native", "python"] = "python"
    effects: Literal["pure", "collective", "random", "state"] = "pure"


@dataclass(slots=True)
class Context(Generic[T]):
    """What an executable runs with: the target, the stream (None: the
    library's current one), and the persistent values by path."""

    target: Target
    parameters: Mapping[str, T] = field(default_factory=dict[str, T])
    states: MutableMapping[str, T] = field(default_factory=dict[str, T])
    stream: int | None = None


class Executable(Protocol[T]):
    """One region, compiled. `execute` enqueues the work and returns; it
    never waits for the device."""

    @property
    def meta(self) -> ExecutableMeta: ...

    def prepare(self, context: Context[T]) -> None:
        """Once per set of weights: convert them into the layout the kernels
        read, and keep what it made."""
        ...

    def execute(
        self, inputs: Sequence[T], outputs: Sequence[T], context: Context[T]
    ) -> Sequence[T]:
        """The region's outputs, in `Region.outputs` order: the ones it was
        given, written, or ones it made (`meta.allocates`)."""
        ...


class Backend(Protocol[T]):
    """A library, compiler or runtime that runs graph regions."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def probe(self, target: Target) -> Capabilities | None:
        """What it runs on `target`; None where it does not run there."""
        ...

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        """The regions of `graph` it can run: a call by its callee, a block's
        call, or any range of steps."""
        ...

    def compile(self, region: Region, target: Target, options: Options) -> Executable[T]: ...

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        """The region's time here, or None to have the planner measure it."""
        ...
