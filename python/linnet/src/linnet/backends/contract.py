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
    """Steps of a graph that one backend runs, and the values that cross its
    boundary: the range `[start, stop)`, or with `members` just those steps
    of it, which run together where the first one stands (sibling products
    that read the same input, say). Parameters and states are not inputs:
    they persist between calls, and an executable reads them by path."""

    graph: Graph
    start: int
    stop: int
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    parameters: tuple[str, ...]
    states: tuple[str, ...]
    # The states it assigns: each one's value after the call, by path.
    next_states: tuple[tuple[str, int], ...] = ()
    call: int | None = None  # the call whose range it is, if any
    members: tuple[int, ...] = ()  # its steps, where not the whole range

    @property
    def indices(self) -> Sequence[int]:
        return self.members or range(self.start, self.stop)

    @property
    def steps(self) -> Sequence[Step]:
        return [self.graph.steps[i] for i in self.indices]

    @property
    def whole(self) -> bool:
        return not self.members and self.start == 0 and self.stop == len(self.graph.steps)

    @classmethod
    def of(cls, graph: Graph, start: int, stop: int, call: int | None = None) -> Region:
        """The range's region: its inputs are the values its steps read that
        steps before it made, in order of first use; its outputs the values
        it makes that later steps read, or that the entry returns. A
        constant crosses no boundary: each region that reads one makes it."""
        return cls.of_steps(graph, range(start, stop), call)

    @classmethod
    def of_steps(cls, graph: Graph, indices: Iterable[int], call: int | None = None) -> Region:
        """The region of these steps, run where the first of them stands:
        its outputs are what the graph's other steps after that read."""
        members = tuple(sorted(indices))
        start, stop = members[0], members[-1] + 1
        persistent = {value: path for path, value in graph.parameters.items()}
        state = {value: path for path, value in graph.states.items()}
        constants = {value for step in graph.find("constant") for value in step.results}
        made: set[int] = set()
        inputs: dict[int, None] = {}
        parameters: dict[str, None] = {}
        states: dict[str, None] = {}
        for step in (graph.steps[i] for i in members):
            if step.op in SOURCES:
                continue  # what the entry takes, which no region makes
            if step.op not in STRUCTURE:
                for operand in step.operands:
                    value = operand.value
                    if value is None or value in made or value in constants:
                        continue
                    if value in persistent:
                        parameters[persistent[value]] = None
                    elif value in state:
                        states[state[value]] = None
                    else:
                        inputs[value] = None
            made.update(step.results)
        # What the other steps after it read, and what the entry returns
        # wherever its `finish` falls.
        mine = set(members)
        read_after: set[int] = {
            operand.value
            for i in range(start, len(graph.steps))
            if i not in mine and graph.steps[i].op not in STRUCTURE
            for operand in graph.steps[i].operands
            if operand.value is not None
        }
        read_after.update(graph.outputs)
        outputs = tuple(
            value
            for step in (graph.steps[i] for i in members)
            if step.op not in SOURCES and step.op != "constant"
            for value in step.results
            if value in read_after
        )
        contiguous = len(members) == stop - start
        if contiguous and start == 0 and stop == len(graph.steps):
            inputs = dict.fromkeys(graph.inputs)
            outputs = graph.outputs
        next_states = tuple(
            (path, value) for path, value in graph.next_states.items() if value in made
        )
        return cls(
            graph,
            start,
            stop,
            tuple(inputs),
            outputs,
            tuple(parameters),
            tuple(states),
            next_states,
            call,
            () if contiguous else members,
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
    """A library, compiler or runtime that runs graph regions. One that
    compiles several regions better together than apart may also offer
    `compile_regions(regions, target, options)`, which the planner then
    calls once with all of its regions."""

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
