"""An entry's graph as `linnet graph` prints it (docs/graph-format.md).

The export's calls to a target, flattened into `steps` in order. The calls
of functions, ops and block methods around them become `calls`, each the
range of steps it made. A range of steps in order is convex: no value leaves
it and comes back, so any range can run on a backend of its own.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from ..compiler import PlanError
from ..ir import JsonValue

# Steps that stand for what the entry takes: its inputs, parameters and
# states before the call.
SOURCES = frozenset(("input", "parameter", "state"))

# Steps that make no tensor operation: placement and loop markers, a
# remat block's ends, `finish`.
STRUCTURE = frozenset(
    (
        "enable_placement",
        "set_slot",
        "release",
        "begin_remat",
        "end_remat",
        "while_initial_condition",
        "while_trailing_condition",
        "finish",
    )
)

# Steps that compute from their operands alone: a value they make from
# constants alone is made again by each region that reads it, as a
# constant is, when it has at most `_MAX_CONSTANT_ELEMENTS` elements.
_COMPUTING = frozenset(("elementwise", "convert", "reshape", "broadcast"))
_MAX_CONSTANT_ELEMENTS = 64


@dataclass(frozen=True, slots=True)
class Value:
    """A tensor a step made; `name` is the target's."""

    id: int
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class Operand:
    """A value as a step or call took it, with the shape and dtype it viewed
    it with; `value` is None for an absent optional."""

    value: int | None
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class Callback:
    """A run of a callback the target made from a step: an op's backward
    pass (`pullback`) or a kernel's fallback (`body`), kept as recorded."""

    role: str
    arguments: tuple[int, ...]
    nodes: tuple[JsonValue, ...]
    results: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class Step:
    """One call the export made to its target. `calls` are the indices of
    the calls around it, outermost first."""

    index: int
    op: str
    attrs: Mapping[str, JsonValue]
    operands: tuple[Operand, ...]
    results: tuple[int, ...]
    declined: bool
    callbacks: tuple[Callback, ...]
    calls: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Call:
    """A call of a function, op (`semantic`) or block method (with its
    block's `path`), and the range of steps `[start, stop)` it made."""

    index: int
    callee: str
    semantic: bool
    path: str
    generics: Mapping[str, int]
    arguments: tuple[Operand, ...]
    results: tuple[Operand, ...]
    start: int
    stop: int
    parent: int | None


@dataclass(frozen=True, slots=True)
class Graph:
    """The steps and calls of one entry, and the values they made."""

    target: str
    values: tuple[Value, ...]
    steps: tuple[Step, ...]
    calls: tuple[Call, ...]
    # What the properties below computed once.
    _memo: dict[str, Mapping[int, Step]] = field(
        default_factory=dict[str, Mapping[int, Step]], compare=False, repr=False
    )

    @classmethod
    def parse(cls, text: str) -> Graph:
        document = _object(json.loads(text))
        if document.get("version") != 1:
            raise PlanError(f"unsupported graph version {document.get('version')!r}")
        values = tuple(
            Value(
                _int(value["id"]),
                str(value["name"]),
                _dims(value["shape"]),
                str(value["dtype"]),
            )
            for value in (_object(v) for v in _list(document["values"]))
        )
        reader = _Reader(values)
        reader.nodes(_list(document["nodes"]), ())
        return cls(str(document["target"]), values, tuple(reader.steps), tuple(reader.calls))

    @property
    def constants(self) -> Mapping[int, Step]:
        """The values no region takes from another but makes again, by the
        step that makes each: constants, and small values computed from
        constants alone (a scale computed from a width, say)."""
        if "constants" not in self._memo:
            found: dict[int, Step] = {}
            for step in self.steps:
                constant = step.op == "constant" or (
                    step.op in _COMPUTING
                    and bool(step.operands)
                    and not step.callbacks
                    and all(o.value is not None and o.value in found for o in step.operands)
                    and all(
                        math.prod(self.values[v].shape) <= _MAX_CONSTANT_ELEMENTS
                        for v in step.results
                    )
                )
                if constant:
                    found.update(dict.fromkeys(step.results, step))
            self._memo["constants"] = MappingProxyType(found)
        return self._memo["constants"]

    def find(self, op: str) -> Iterator[Step]:
        """The steps of kind `op`, in order."""
        return (step for step in self.steps if step.op == op)

    @property
    def inputs(self) -> tuple[int, ...]:
        """The entry's inputs, in order."""
        return tuple(step.results[0] for step in self.find("input"))

    @property
    def parameters(self) -> Mapping[str, int]:
        """Each parameter's value, by path, in the order the target took them."""
        return {str(step.attrs["path"]): step.results[0] for step in self.find("parameter")}

    @property
    def states(self) -> Mapping[str, int]:
        """Each state's value before the call, by path."""
        return {str(step.attrs["path"]): step.results[0] for step in self.find("state")}

    @property
    def finish(self) -> Step:
        found = next(self.find("finish"), None)
        if found is None:
            raise PlanError("malformed graph: no `finish` step")
        return found

    @property
    def outputs(self) -> tuple[int, ...]:
        """The entry's results."""
        count = _int(self.finish.attrs["returns"])
        return tuple(_present(operand) for operand in self.finish.operands[:count])

    @property
    def next_states(self) -> Mapping[str, int]:
        """Each state the entry assigns, its value after the call, by path."""
        finish = self.finish
        count = _int(finish.attrs["returns"])
        paths = [str(path) for path in _list(finish.attrs["states"])]
        return {
            path: _present(operand)
            for path, operand in zip(paths, finish.operands[count:], strict=True)
        }

    def semantic_calls(self, callee: str | None = None) -> Iterator[Call]:
        """The calls of ops (of `callee`, when given), outermost first."""
        return (
            call
            for call in self.calls
            if call.semantic and (callee is None or call.callee == callee)
        )

    @property
    def producers(self) -> Mapping[int, Step]:
        """The step that made each value, by value."""
        if "producers" not in self._memo:
            made = {value: step for step in self.steps for value in step.results}
            self._memo["producers"] = MappingProxyType(made)
        return self._memo["producers"]

    def scalar(self, value: int | None) -> float | None:
        """The number a value computed from constants alone holds, such as an
        attention's scale, or None where it reads anything else."""
        step = self.producers.get(value) if value is not None else None
        if step is None:
            return None
        if step.op == "constant":
            literal = step.attrs.get("literal")
            number = literal.get("value") if isinstance(literal, dict) else None
            return float(number) if isinstance(number, int | float) else None
        if step.op in ("convert", "reshape", "broadcast") and len(step.operands) == 1:
            return self.scalar(step.operands[0].value)
        if step.op != "elementwise":
            return None
        numbers = [self.scalar(operand.value) for operand in step.operands]
        if any(n is None for n in numbers):
            return None
        kind = str(step.attrs.get("kind"))
        if len(numbers) == 1 and kind in _UNARY:
            return _UNARY[kind](cast("float", numbers[0]))
        if len(numbers) == 2 and kind in _BINARY:
            return _BINARY[kind](cast("float", numbers[0]), cast("float", numbers[1]))
        return None

    def reaches_prefix(self, mask: int) -> bool:
        """Whether an attention mask reaches a prefix of the keys: positions
        compared with something below or at them (`iota <= position`),
        maybe reshaped or broadcast."""
        step = self.producers.get(mask)
        while step is not None and step.op in ("reshape", "broadcast") and len(step.operands) == 1:
            inner = step.operands[0].value
            step = self.producers.get(inner) if inner is not None else None
        if step is None or step.op != "compare" or step.attrs.get("kind") not in ("le", "lt"):
            return False
        keys, limit = step.operands
        return self._from_iota(keys.value) and not self._from_iota(limit.value)

    def segment_causal(self, mask: int) -> SegmentMask | None:
        """The segments and positions of an attention mask that lets each
        query see the keys of its own segment at its position or before
        (`segments[i] == segments[j] && positions[j] <= positions[i]`, as a
        packed pass of prompts builds it), with the first step that builds
        it; None for any other mask."""
        step = self.producers.get(mask)
        if step is None or step.op != "elementwise" or step.attrs.get("kind") != "and":
            return None
        if len(step.operands) != 2:
            return None
        compares = [
            self.producers.get(o.value) if o.value is not None else None for o in step.operands
        ]
        if any(c is None or c.op != "compare" for c in compares):
            return None
        same, before = sorted(
            cast("list[Step]", compares), key=lambda c: str(c.attrs.get("kind")) != "eq"
        )
        if same.attrs.get("kind") != "eq" or before.attrs.get("kind") != "le":
            return None
        a, b = (o.value for o in same.operands)
        segments = self._spread(a, b)
        if segments is None:
            segments = self._spread(b, a)  # `==` either way round
        # `positions[j] <= positions[i]`: the keys' along the second axis.
        positions = self._spread(before.operands[1].value, before.operands[0].value)
        if segments is None or positions is None or self._from_iota(positions):
            return None
        built = [step, *cast("list[Step]", compares)]
        built += [
            self.producers[o.value]
            for c in cast("list[Step]", compares)
            for o in c.operands
            if o.value is not None
        ]
        first = min(self.steps.index(s) for s in built)
        return SegmentMask(segments, positions, first)

    def _spread(self, rows: int | None, columns: int | None) -> int | None:
        """The one value `rows` broadcasts along the first axis and
        `columns` along the second, or None."""
        made = [self.producers.get(v) if v is not None else None for v in (rows, columns)]
        if any(m is None or m.op != "broadcast" or len(m.operands) != 1 for m in made):
            return None
        row, column = cast("list[Step]", made)
        if row.attrs.get("dims") != [0] or column.attrs.get("dims") != [1]:
            return None
        source = row.operands[0].value
        return source if source is not None and source == column.operands[0].value else None

    def _from_iota(self, value: int | None) -> bool:
        """Whether `value` is positions 0, 1, ... (an `iota`, converted,
        reshaped or broadcast)."""
        while value is not None and value in self.producers:
            step = self.producers[value]
            if step.op == "iota":
                return True
            if step.op not in ("convert", "reshape", "broadcast") or len(step.operands) != 1:
                return False
            value = step.operands[0].value
        return False


@dataclass(frozen=True, slots=True)
class SegmentMask:
    """What `Graph.segment_causal` found: the values of the segments and of
    the positions the mask compares, and the first step that builds it."""

    segments: int
    positions: int
    start: int


# The elementwise operations `Graph.scalar` evaluates.
_UNARY: dict[str, Callable[[float], float]] = {
    "neg": lambda a: -a,
    "sqrt": math.sqrt,
    "rsqrt": lambda a: 1.0 / math.sqrt(a),
    "exp": math.exp,
    "log": math.log,
}
_BINARY: dict[str, Callable[[float, float], float]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
}


def _present(operand: Operand) -> int:
    if operand.value is None:
        raise PlanError("malformed graph: an absent value among an entry's results")
    return operand.value


class _Reader:
    """Flattens the nested nodes of a graph document into steps and calls."""

    def __init__(self, values: Sequence[Value]) -> None:
        self.values = values
        self.steps: list[Step] = []
        self.calls: list[Call] = []

    def operand(self, data: JsonValue) -> Operand:
        if data is None:
            return Operand(None, (), "")
        if isinstance(data, dict):
            return Operand(_int(data["value"]), _dims(data["shape"]), str(data["dtype"]))
        value = self.values[_int(data)]
        return Operand(value.id, value.shape, value.dtype)

    def nodes(self, nodes: Sequence[JsonValue], around: tuple[int, ...]) -> None:
        for node in (_object(n) for n in nodes):
            if node["op"] == "call":
                self.call(node, around)
            else:
                self.step(node, around)

    def call(self, node: Mapping[str, JsonValue], around: tuple[int, ...]) -> None:
        index = len(self.calls)
        # A placeholder until the call's steps are read: the range ends where
        # they do.
        self.calls.append(Call(index, "", False, "", {}, (), (), 0, 0, None))
        start = len(self.steps)
        self.nodes(_list(node["body"]), (*around, index))
        generics = _object(node["generics"])
        self.calls[index] = Call(
            index,
            str(node["callee"]),
            _int(node["semantic"]) != 0,
            str(node["path"]),
            MappingProxyType({key: _int(value) for key, value in generics.items()}),
            tuple(self.operand(a) for a in _list(node["arguments"])),
            tuple(self.operand(r) for r in _list(node["results"])),
            start,
            len(self.steps),
            around[-1] if around else None,
        )

    def step(self, node: Mapping[str, JsonValue], around: tuple[int, ...]) -> None:
        fields = ("op", "operands", "results", "declined", "regions")
        callbacks = tuple(
            Callback(
                str(region["role"]),
                tuple(_int(a) for a in _list(region["arguments"])),
                tuple(_list(region["nodes"])),
                tuple(None if r is None else _int(r) for r in _list(region["results"])),
            )
            for region in (_object(r) for r in _list(node.get("regions", [])))
        )
        self.steps.append(
            Step(
                len(self.steps),
                str(node["op"]),
                MappingProxyType({k: v for k, v in node.items() if k not in fields}),
                tuple(self.operand(o) for o in _list(node["operands"])),
                tuple(_int(r) for r in _list(node["results"])),
                node.get("declined") is True,
                callbacks,
                around,
            )
        )


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise PlanError(f"malformed graph: expected an object, found {type(value).__name__}")
    return value


def _list(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise PlanError(f"malformed graph: expected a list, found {type(value).__name__}")
    return value


def _int(value: JsonValue) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PlanError(f"malformed graph: expected an integer, found {value!r}")
    return value


def _dims(value: JsonValue) -> tuple[int, ...]:
    return tuple(_int(d) for d in _list(value))
