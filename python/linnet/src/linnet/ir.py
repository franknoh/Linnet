"""The compiled program: what `linnet plan` prints (see
`docs/plan-format.md`), read by `Program.from_json` into frozen dataclasses
with exhaustive unions for dimensions and types. The runtimes walk it (the
PyTorch interpreter and modules, the JAX and ONNX loaders), and so do the
diagrams, the model cards and the resource analysis, none of them indexing
JSON. Dimensions stay symbolic until `Bindings` give the generics values;
`format_type` prints them the way the language spells them
(`Tensor[B, S, H; bf16]`). An operation's attributes stay as the plan
spells them; `parse_shape`, `parse_dim` and `parse_substitution` read them.
"""

from __future__ import annotations

import fnmatch
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, TypeVar, cast

from .compiler import LinnetError, PlanError
from .dtypes import CLASSES, DTYPES

T = TypeVar("T")

# A value as `json.loads` returns it: what the plan and an operation's attributes hold.
JsonValue: TypeAlias = "bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None"

# ---------------------------------------------------------------- dimensions


@dataclass(frozen=True, slots=True)
class DimSymbol:
    """A dimension generic such as `B`."""

    id: int
    name: str


@dataclass(frozen=True, slots=True)
class PackSize:
    """The number of elements in a shape pack (`*S`) seen as one dimension."""

    id: int
    name: str


@dataclass(frozen=True, slots=True)
class DimExpr:
    """Arithmetic over dimensions."""

    op: Literal["add", "mul", "floordiv", "mod", "min", "max"]
    args: tuple[Dim, ...]


Dim = int | DimSymbol | PackSize | DimExpr


@dataclass(frozen=True, slots=True)
class Pack:
    """A shape pack standing for any number of axes."""

    id: int
    name: str


Unit = Dim | Pack
Shape = tuple[Unit, ...]


@dataclass(frozen=True, slots=True)
class DTypeVar:
    """A dtype generic such as `T`."""

    id: int
    name: str


DType = str | DTypeVar

# --------------------------------------------------------------------- types


@dataclass(frozen=True, slots=True)
class ScalarType:
    dtype: DType


@dataclass(frozen=True, slots=True)
class TensorType:
    shape: Shape
    dtype: DType


@dataclass(frozen=True, slots=True)
class TupleType:
    elements: tuple[Type, ...]


@dataclass(frozen=True, slots=True)
class OptionalType:
    inner: Type


@dataclass(frozen=True, slots=True)
class ArrayType:
    element: Type
    length: Dim


@dataclass(frozen=True, slots=True)
class DimArg:
    dim: Dim


@dataclass(frozen=True, slots=True)
class ShapeArg:
    shape: Shape


@dataclass(frozen=True, slots=True)
class DTypeArg:
    dtype: DType


GenericArg = DimArg | ShapeArg | DTypeArg


@dataclass(frozen=True, slots=True)
class NamedType:
    """A block, struct, or enum instantiated with generic arguments."""

    kind: Literal["block", "struct", "enum"]
    name: str
    module: str
    args: tuple[GenericArg, ...]


@dataclass(frozen=True, slots=True)
class ShapeType:
    shape: Shape


@dataclass(frozen=True, slots=True)
class UnitType:
    pass


Type = (
    ScalarType
    | TensorType
    | TupleType
    | OptionalType
    | ArrayType
    | NamedType
    | ShapeType
    | UnitType
)

# --------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class Generic:
    name: str
    kind: Literal["dim", "shape", "dtype"]
    id: int
    default: GenericArg | None = None
    dtype_class: Literal["float", "integer", "numeric", "any"] | None = None


@dataclass(frozen=True, slots=True)
class Constraint:
    relation: Literal["==", "!=", "<", "<=", ">", ">="]
    lhs: Dim
    rhs: Dim


@dataclass(frozen=True, slots=True)
class Member:
    name: str
    kind: Literal["param", "buffer", "state", "sub"]
    type: Type


@dataclass(frozen=True, slots=True)
class Block:
    name: str
    module: str
    pub: bool
    generics: tuple[Generic, ...]
    constraints: tuple[Constraint, ...]
    members: tuple[Member, ...]

    def member(self, name: str) -> Member:
        for member in self.members:
            if member.name == name:
                return member
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One parameter, buffer, or state of the instantiated hierarchy."""

    path: str
    kind: Literal["param", "buffer", "state"]
    dtype: DType
    shape: Shape
    repeat: tuple[Dim, ...]
    optional: bool


@dataclass(frozen=True, slots=True)
class Substitution:
    """What a call binds the callee's generics to (the `substitution` attribute)."""

    dims: Mapping[int, Dim]
    packs: Mapping[int, Shape]
    dtypes: Mapping[int, DType]


@dataclass(frozen=True, slots=True)
class Value:
    id: int
    name: str
    type: Type


@dataclass(frozen=True, slots=True)
class Op:
    kind: str
    operands: tuple[int, ...]
    results: tuple[Value, ...]
    attrs: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))
    regions: tuple[Region, ...] = ()


@dataclass(frozen=True, slots=True)
class Region:
    args: tuple[Value, ...]
    ops: tuple[Op, ...]

    def walk(self) -> Iterator[Op]:
        """Every operation, nested regions included, in program order."""
        for op in self.ops:
            yield op
            for region in op.regions:
                yield from region.walk()


@dataclass(frozen=True, slots=True)
class Function:
    name: str
    kind: Literal["fn", "op", "entry", "kernel"]
    block: str | None
    pub: bool
    generics: tuple[Generic, ...]
    constraints: tuple[Constraint, ...]
    results: tuple[Type, ...]
    states: tuple[str, ...]
    body: Region
    # An op's `grad`: its parameters, result and result gradient in, the
    # gradient of each parameter that `takes_gradient` out.
    gradient: Region | None = None
    # A kernel: its body's last `outputs` arguments are the results it writes,
    # and `grid` yields its launch grid. An op's kernel, by name.
    outputs: int = 0
    grid: Region | None = None
    kernel: str | None = None

    @property
    def short_name(self) -> str:
        """`forward` for `llama::Model.forward`, `linear` for `std.nn.linear::linear`."""
        tail = self.name.rsplit("::", 1)[-1]
        return tail.rsplit(".", 1)[-1]

    @property
    def params(self) -> tuple[Value, ...]:
        """The declared parameters: the body's arguments without a leading `self`."""
        args = self.body.args
        if args and args[0].name == "self" and isinstance(args[0].type, NamedType):
            return args[1:]
        return args


@dataclass(frozen=True, slots=True)
class Constant:
    name: str
    pub: bool
    type: Type | None
    contextual: bool
    body: Region


@dataclass(frozen=True, slots=True)
class Root:
    name: str
    generics: tuple[Generic, ...]
    constraints: tuple[Constraint, ...]


@dataclass(frozen=True, slots=True)
class Program:
    """A compiled root block with every block, function, and constant it uses."""

    version: int
    module: str
    root: Root
    manifest: tuple[ManifestEntry, ...]
    blocks: Mapping[str, Block]
    functions: Mapping[str, Function]
    constants: tuple[Constant, ...]
    text: str = field(default="", repr=False, compare=False)  # the plan as `linnet plan` printed it

    @staticmethod
    def from_json(text: str) -> Program:
        document: JsonValue = json.loads(text)
        if not isinstance(document, dict):
            raise LinnetError("a plan is a JSON object")
        return _Reader().program(document, text)

    @property
    def root_block(self) -> Block:
        return self.blocks[self.root.name]

    def entries(self, block: str | None = None) -> tuple[Function, ...]:
        """The entries of a block (the root by default), in declaration order."""
        name = self.root.name if block is None else block
        return tuple(f for f in self.functions.values() if f.kind == "entry" and f.block == name)

    def methods(self, block: str) -> tuple[Function, ...]:
        return tuple(f for f in self.functions.values() if f.block == block)

    def entry(self, name: str | None = None, *, prefer: str | None = None) -> Function:
        """The root entry called `name`; without one, the only entry, or
        else the one called `prefer`."""
        entries = {e.short_name: e for e in self.entries()}
        return entries[choose_entry(self.root.name, list(entries), name, prefer)]

    def parameters(self) -> tuple[ManifestEntry, ...]:
        return tuple(e for e in self.manifest if e.kind == "param")

    def module_entries(self) -> dict[str, Function]:
        """The entries declared at module level in the root file, by name:
        functions of their inputs alone (`linnet plan --functions`)."""
        prefix = f"{self.module}::"
        return {
            f.name.removeprefix(prefix): f
            for f in self.functions.values()
            if f.kind == "entry" and f.block is None and f.name.startswith(prefix)
        }

    def module_entry(self, name: str | None) -> Function:
        """The module-level entry called `name`, or the only one."""
        entries = self.module_entries()
        listed = ", ".join(f"`{entry}`" for entry in entries) or "none"
        if name is None:
            if len(entries) != 1:
                raise PlanError(
                    f"the module has {len(entries)} module-level entries ({listed}); name one"
                )
            return next(iter(entries.values()))
        if name not in entries:
            raise PlanError(f"no module-level entry `{name}` (there are: {listed})")
        return entries[name]

    def describe(self, block: str | None = None) -> str:
        """The block's signature, members, and functions as the language spells them."""
        name = self.root.name if block is None else block
        data = self.blocks[name]
        lines = [f"{data.module}::{name}{format_generics(data.generics)}"]
        for member in data.members:
            lines.append(f"  {member.kind} {member.name}: {format_type(member.type)}")
        for function in self.methods(name):
            lines.append("  " + format_signature(function))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def load_program(
    source: str | Path,
    *,
    root: str | None = None,
    std_root: str | Path | None = None,
    optimize: bool = True,
    numerics: str = "exact",
) -> Program:
    """Compiles a source file (`linnet plan`) and returns its typed program."""
    from .plan import compile_plan

    return compile_plan(source, root=root, std_root=std_root, optimize=optimize, numerics=numerics)


# --------------------------------------------------------------------- reader


def _object(value: JsonValue) -> dict[str, JsonValue]:
    """A plan node, which is a JSON object."""
    if not isinstance(value, dict):
        raise PlanError(f"malformed plan: expected an object, found {type(value).__name__}")
    return value


def _list(value: JsonValue | Sequence[JsonValue]) -> Sequence[JsonValue]:
    """A plan field holding a list."""
    if not isinstance(value, list | tuple):
        raise PlanError(f"malformed plan: expected a list, found {type(value).__name__}")
    return value


def _int(value: JsonValue) -> int:
    """A plan field holding an integer, or what `int` reads as one."""
    if value is None or isinstance(value, list | dict):
        raise PlanError(f"malformed plan: expected an integer, found {type(value).__name__}")
    return int(value)


class _Reader:
    """Reads the plan's JSON. A node is a JSON object (`_object`); each
    field is checked or converted (`_int`, `_list`, `str`) where it is read."""

    def program(self, plan: dict[str, JsonValue], text: str = "") -> Program:
        version = _int(plan.get("version", 0))
        if version != 1:
            raise LinnetError(f"unsupported plan version {version!r}")
        blocks = {name: self.block(name, data) for name, data in _object(plan["blocks"]).items()}
        functions = [self.function(f) for f in _list(plan["functions"])]
        # A plan of functions (`--functions`) has no root block.
        root = _object(plan["root"] or {"name": ""})
        return Program(
            version=version,
            module=str(plan["module"]),
            root=Root(
                name=str(root["name"]),
                generics=self.generics(root.get("generics", [])),
                constraints=self.constraints(root.get("constraints", [])),
            ),
            manifest=tuple(self.manifest_entry(e) for e in _list(plan["manifest"])),
            blocks=MappingProxyType(blocks),
            functions=MappingProxyType({f.name: f for f in functions}),
            constants=tuple(self.constant(c) for c in _list(plan.get("constants", []))),
            text=text,
        )

    # ---- dimensions and types

    def dim(self, data: JsonValue) -> Dim:
        if isinstance(data, bool):
            raise LinnetError("a dimension cannot be a boolean")
        if isinstance(data, int):
            return data
        node = _object(data)
        if "sym" in node:
            return DimSymbol(_int(node["sym"]), str(node.get("name", "")))
        if "packsize" in node:
            return PackSize(_int(node["packsize"]), str(node.get("name", "")))
        op = str(node["op"])
        if op not in ("add", "mul", "floordiv", "mod", "min", "max"):
            raise LinnetError(f"unknown dimension operator `{op}`")
        return DimExpr(op, tuple(self.dim(arg) for arg in _list(node["args"])))

    def unit(self, data: JsonValue) -> Unit:
        if isinstance(data, dict) and "pack" in data:
            return Pack(_int(data["pack"]), str(data.get("name", "")))
        return self.dim(data)

    def shape(self, data: JsonValue | Sequence[JsonValue]) -> Shape:
        return tuple(self.unit(unit) for unit in _list(data))

    def dtype(self, data: JsonValue) -> DType:
        if isinstance(data, str):
            return data
        node = _object(data)
        return DTypeVar(_int(node["var"]), str(node.get("name", "")))

    def generic_arg(self, data: JsonValue) -> GenericArg:
        node = _object(data)
        if "dim" in node:
            return DimArg(self.dim(node["dim"]))
        if "shape" in node:
            return ShapeArg(self.shape(node["shape"]))
        return DTypeArg(self.dtype(node["dtype"]))

    def type(self, data: JsonValue) -> Type:
        node = _object(data)
        kind = str(node["kind"])
        if kind == "scalar":
            return ScalarType(self.dtype(node["dtype"]))
        if kind == "tensor":
            return TensorType(self.shape(node["shape"]), self.dtype(node["dtype"]))
        if kind == "tuple":
            return TupleType(tuple(self.type(e) for e in _list(node["elements"])))
        if kind == "optional":
            return OptionalType(self.type(node["inner"]))
        if kind == "array":
            return ArrayType(self.type(node["element"]), self.dim(node["length"]))
        if kind in ("block", "struct", "enum"):
            return NamedType(
                kind,
                str(node["name"]),
                str(node.get("module", "")),
                tuple(self.generic_arg(a) for a in _list(node.get("args", []))),
            )
        if kind == "shape":
            return ShapeType(self.shape(node["shape"]))
        if kind == "unit":
            return UnitType()
        raise LinnetError(f"unknown type kind `{kind}`")

    # ---- declarations

    def generics(self, data: JsonValue) -> tuple[Generic, ...]:
        out: list[Generic] = []
        for item in _list(data):
            node = _object(item)
            kind = str(node["kind"])
            if kind not in ("dim", "shape", "dtype"):
                raise LinnetError(f"unknown generic kind `{kind}`")
            default = node.get("default")
            dtype_class = node.get("class")
            out.append(
                Generic(
                    name=str(node["name"]),
                    kind=kind,
                    id=_int(node["var"] if kind == "dtype" else node["sym"]),
                    default=None if default is None else self.generic_arg(default),
                    dtype_class=None
                    if dtype_class is None
                    else cast(Literal["float", "integer", "numeric", "any"], str(dtype_class)),
                )
            )
        return tuple(out)

    def constraints(self, data: JsonValue) -> tuple[Constraint, ...]:
        out: list[Constraint] = []
        for item in _list(data):
            node = _object(item)
            relation = str(node["relation"])
            if relation not in ("==", "!=", "<", "<=", ">", ">="):
                raise LinnetError(f"unknown relation `{relation}`")
            out.append(
                Constraint(
                    relation,
                    self.dim(node["lhs"]),
                    self.dim(node["rhs"]),
                )
            )
        return tuple(out)

    def block(self, name: str, data: JsonValue) -> Block:
        block = _object(data)
        members: list[Member] = []
        for item in _list(block.get("members", [])):
            node = _object(item)
            kind = str(node["kind"])
            if kind not in ("param", "buffer", "state", "sub"):
                raise LinnetError(f"unknown member kind `{kind}`")
            members.append(Member(str(node["name"]), kind, self.type(node["type"])))
        return Block(
            name=name,
            module=str(block.get("module", "")),
            pub=bool(block.get("pub", False)),
            generics=self.generics(block.get("generics", [])),
            constraints=self.constraints(block.get("constraints", [])),
            members=tuple(members),
        )

    def manifest_entry(self, data: JsonValue) -> ManifestEntry:
        node = _object(data)
        kind = str(node["kind"])
        if kind not in ("param", "buffer", "state"):
            raise LinnetError(f"unknown manifest kind `{kind}`")
        return ManifestEntry(
            path=str(node["path"]),
            kind=kind,
            dtype=self.dtype(node["dtype"]),
            shape=self.shape(node["shape"]),
            repeat=tuple(self.dim(d) for d in _list(node.get("repeat", []))),
            optional=bool(node.get("optional", False)),
        )

    def substitution(self, data: JsonValue) -> Substitution:
        node = _object(data or {})
        dims = _object(node.get("dims", {}))
        packs = _object(node.get("packs", {}))
        dtypes = _object(node.get("dtypes", {}))
        return Substitution(
            dims=MappingProxyType({int(k): self.dim(v) for k, v in dims.items()}),
            packs=MappingProxyType({int(k): self.shape(v) for k, v in packs.items()}),
            dtypes=MappingProxyType({int(k): self.dtype(v) for k, v in dtypes.items()}),
        )

    def value(self, data: JsonValue) -> Value:
        node = _object(data)
        return Value(_int(node["id"]), str(node.get("name", "")), self.type(node["type"]))

    def region(self, data: JsonValue) -> Region:
        node = _object(data)
        return Region(
            args=tuple(self.value(v) for v in _list(node.get("args", []))),
            ops=tuple(self.op(o) for o in _list(node.get("ops", []))),
        )

    def op(self, data: JsonValue) -> Op:
        node = _object(data)
        attrs = _object(node.get("attrs") or {})
        return Op(
            kind=str(node["kind"]),
            operands=tuple(_int(i) for i in _list(node.get("operands", []))),
            results=tuple(self.value(v) for v in _list(node.get("results", []))),
            attrs=MappingProxyType(dict(attrs)),
            regions=tuple(self.region(r) for r in _list(node.get("regions", []))),
        )

    def function(self, data: JsonValue) -> Function:
        node = _object(data)
        kind = str(node["kind"])
        if kind not in ("fn", "op", "entry", "kernel"):
            raise LinnetError(f"unknown function kind `{kind}`")
        block = node.get("block")
        return Function(
            name=str(node["name"]),
            kind=kind,
            block=None if block is None else str(block),
            pub=bool(node.get("pub", False)),
            generics=self.generics(node.get("generics", [])),
            constraints=self.constraints(node.get("constraints", [])),
            results=tuple(self.type(t) for t in _list(node.get("results", []))),
            states=tuple(str(s) for s in _list(node.get("states", []))),
            body=self.region(node["body"]),
            gradient=self.region(node["gradient"]) if "gradient" in node else None,
            outputs=_int(node.get("outputs", 0)),
            grid=self.region(node["grid"]) if "grid" in node else None,
            kernel=str(_object(node["kernel"])["name"]) if "kernel" in node else None,
        )

    def constant(self, data: JsonValue) -> Constant:
        node = _object(data)
        type_data = node.get("type")
        return Constant(
            name=str(node["name"]),
            pub=bool(node.get("pub", False)),
            type=None if type_data is None else self.type(type_data),
            contextual=bool(node.get("contextual", False)),
            body=self.region(node["body"]),
        )


def takes_gradient(function: Function, type: Type) -> bool:
    """Whether an op's `grad` returns a gradient for a parameter of `type`: a
    tensor of a floating dtype or of a dtype generic bounded by `Float`."""
    if not isinstance(type, TensorType):
        return False
    dtype = type.dtype
    if isinstance(dtype, DTypeVar):
        return any(
            generic.kind == "dtype" and generic.id == dtype.id and generic.dtype_class == "float"
            for generic in function.generics
        )
    return DTYPES[dtype].is_float


def parse_substitution(data: JsonValue) -> Substitution:
    """Reads a call's `substitution` attribute."""
    return _Reader().substitution(data)


def parse_dim(data: JsonValue) -> Dim:
    """Reads a dimension as an attribute spells it (a slice bound, say)."""
    return _Reader().dim(data)


def parse_shape(data: JsonValue | Sequence[JsonValue]) -> Shape:
    """Reads a shape as an attribute spells it (an index domain, say)."""
    return _Reader().shape(data)


def call_substitution(op: Op) -> Substitution:
    """The substitution a `call` applies to its callee's types."""
    return parse_substitution(op.attrs.get("substitution"))


# ------------------------------------------------------------- substitution


def substitute_dim(dim: Dim, s: Substitution) -> Dim:
    if isinstance(dim, int):
        return dim
    if isinstance(dim, DimSymbol):
        return s.dims.get(dim.id, dim)
    if isinstance(dim, PackSize):
        if dim.id in s.packs:
            shape = s.packs[dim.id]
            if all(not isinstance(u, Pack) for u in shape):
                dims = tuple(cast(Dim, u) for u in shape)
                return dims[0] if len(dims) == 1 else DimExpr("mul", dims) if dims else 1
        return dim
    return DimExpr(dim.op, tuple(substitute_dim(a, s) for a in dim.args))


def substitute_shape(shape: Shape, s: Substitution) -> Shape:
    out: list[Unit] = []
    for unit in shape:
        if isinstance(unit, Pack):
            out.extend(s.packs.get(unit.id, (unit,)))
        else:
            out.append(substitute_dim(unit, s))
    return tuple(out)


def substitute_dtype(dtype: DType, s: Substitution) -> DType:
    if isinstance(dtype, DTypeVar):
        return s.dtypes.get(dtype.id, dtype)
    return dtype


def substitute_arg(arg: GenericArg, s: Substitution) -> GenericArg:
    if isinstance(arg, DimArg):
        return DimArg(substitute_dim(arg.dim, s))
    if isinstance(arg, ShapeArg):
        return ShapeArg(substitute_shape(arg.shape, s))
    return DTypeArg(substitute_dtype(arg.dtype, s))


def substitute(type: Type, s: Substitution) -> Type:
    """The type with the callee's generics replaced by what a call binds them to."""
    if isinstance(type, ScalarType):
        return ScalarType(substitute_dtype(type.dtype, s))
    if isinstance(type, TensorType):
        return TensorType(substitute_shape(type.shape, s), substitute_dtype(type.dtype, s))
    if isinstance(type, TupleType):
        return TupleType(tuple(substitute(e, s) for e in type.elements))
    if isinstance(type, OptionalType):
        return OptionalType(substitute(type.inner, s))
    if isinstance(type, ArrayType):
        return ArrayType(substitute(type.element, s), substitute_dim(type.length, s))
    if isinstance(type, NamedType):
        return NamedType(
            type.kind, type.name, type.module, tuple(substitute_arg(a, s) for a in type.args)
        )
    if isinstance(type, ShapeType):
        return ShapeType(substitute_shape(type.shape, s))
    return type


# ---------------------------------------------------------------- evaluation


@dataclass(slots=True)
class Bindings:
    """Concrete values for generics, by symbol id: what `--bind` gives the
    compiler. A runtime binds more as a call reads its inputs."""

    dims: dict[int, int] = field(default_factory=dict[int, int])
    packs: dict[int, tuple[int, ...]] = field(default_factory=dict[int, tuple[int, ...]])
    dtypes: dict[int, str] = field(default_factory=dict[int, str])

    def dim(self, dim: Dim) -> int:
        return evaluate_dim(dim, self)

    def shape(self, shape: Shape) -> tuple[int, ...]:
        return evaluate_shape(shape, self)

    def dtype(self, dtype: DType) -> str:
        return evaluate_dtype(dtype, self)

    def holds(self, constraint: Constraint) -> bool:
        """Whether a `where` constraint holds under these bindings."""
        return holds(constraint.relation, self.dim(constraint.lhs), self.dim(constraint.rhs))

    def copy(self) -> Bindings:
        return Bindings(dict(self.dims), dict(self.packs), dict(self.dtypes))


def holds(relation: str, lhs: int, rhs: int) -> bool:
    """Whether `lhs relation rhs` (`==`, `!=`, `<`, `<=`, `>`, `>=`) holds."""
    return {
        "==": lhs == rhs,
        "!=": lhs != rhs,
        "<": lhs < rhs,
        "<=": lhs <= rhs,
        ">": lhs > rhs,
        ">=": lhs >= rhs,
    }[relation]


def default_of(generic: Generic) -> int | str | None:
    """A generic's declared default as a value: a dimension's size or a
    dtype's name; None without one, or for one in terms of other generics."""
    default = generic.default
    if isinstance(default, DimArg) and isinstance(default.dim, int):
        return default.dim
    if isinstance(default, DTypeArg) and isinstance(default.dtype, str):
        return default.dtype
    return None


def bind_generics(generics: Sequence[Generic], values: Mapping[str, int | str]) -> Bindings:
    """Binds generics by name, using declared defaults for the rest.

    Raises `LinnetError` for a generic that is neither given nor defaulted,
    and for a value of the wrong kind.
    """
    dims: dict[int, int] = {}
    packs: dict[int, tuple[int, ...]] = {}
    dtypes: dict[int, str] = {}
    for generic in generics:
        value = values.get(generic.name)
        if value is None:
            value = default_of(generic)
            if value is None:
                raise LinnetError(f"generic `{generic.name}` needs a value")
        if generic.kind == "dtype":
            if not isinstance(value, str):
                raise LinnetError(f"`{generic.name}` is a dtype; give its name")
            dtypes[generic.id] = value
        elif generic.kind == "dim":
            if not isinstance(value, int):
                raise LinnetError(f"`{generic.name}` is a dimension; give an integer")
            dims[generic.id] = value
        else:
            raise LinnetError(f"`{generic.name}` is a shape pack and cannot be bound by name")
    unknown = sorted(set(values) - {g.name for g in generics})
    if unknown:
        raise LinnetError(f"unknown generics: {', '.join(unknown)}")
    return Bindings(dims, packs, dtypes)


def bound(env: Bindings, generic: Generic) -> bool:
    """Whether `env` binds `generic`."""
    if generic.kind == "dim":
        return generic.id in env.dims
    if generic.kind == "shape":
        return generic.id in env.packs
    return generic.id in env.dtypes


def bind_named(env: Bindings, generics: Sequence[Generic], given: Mapping[str, int | str]) -> None:
    """Binds an entry's generics given by name: a dimension as an integer, a
    dtype by its name."""
    names = {generic.name for generic in generics}
    for name in given:
        if name not in names:
            raise PlanError(f"the entry has no generic parameter `{name}`")
    for generic in generics:
        if generic.name not in given:
            continue
        value = given[generic.name]
        if generic.kind == "dim":
            if not isinstance(value, int):
                raise PlanError(f"`{generic.name}` is a dimension; give an integer")
            env.dims[generic.id] = value
        elif generic.kind == "dtype":
            env.dtypes[generic.id] = str(value)
        else:
            raise PlanError(f"`{generic.name}` is a shape pack and cannot be given by name")


def align_shape(
    units: Sequence[Unit], shape: Sequence[int], name: str
) -> tuple[list[tuple[Dim, int]], tuple[Pack, list[int]] | None]:
    """Input `name`'s declared shape (`units`, with at most one shape pack,
    which covers whatever axes the others leave) lined up with its actual
    `shape`: each declared dimension with its size, and the pack with the
    sizes it covers."""
    packs = [i for i, unit in enumerate(units) if isinstance(unit, Pack)]
    if len(packs) > 1:
        raise PlanError(f"input `{name}` has more than one shape pack")
    fixed = len(units) - len(packs)
    if (packs and len(shape) < fixed) or (not packs and len(shape) != fixed):
        expected = f"at least {fixed}" if packs else str(fixed)
        raise PlanError(f"input `{name}` has rank {len(shape)}, expected {expected}")
    if not packs:
        dims = [unit for unit in units if not isinstance(unit, Pack)]
        return list(zip(dims, shape, strict=True)), None
    at, width = packs[0], len(shape) - fixed
    rest = [unit for unit in [*units[:at], *units[at + 1 :]] if not isinstance(unit, Pack)]
    sizes = [*shape[:at], *shape[at + width :]]
    pack = units[at]
    assert isinstance(pack, Pack)
    return list(zip(rest, sizes, strict=True)), (pack, list(shape[at : at + width]))


def bind_input(env: Bindings, param: Value, shape: Sequence[int], dtype: str | None) -> None:
    """Binds the generics an input of `shape` and `dtype` (its Linnet name;
    None leaves the dtype unchecked) determines, and checks it against the
    ones already bound."""
    declared = param.type
    name = param.name
    if not isinstance(declared, ScalarType | TensorType):
        raise PlanError(f"input `{name}` is neither a tensor nor a scalar")
    if dtype is not None:
        if dtype not in DTYPES:
            raise PlanError(f"input `{name}` has dtype {dtype}, which Linnet lacks")
        spec = declared.dtype
        # A dtype generic of the entry's own (a function's `T`) is bound by
        # the first input that carries it; the rest must agree.
        wanted = env.dtypes.setdefault(spec.id, dtype) if isinstance(spec, DTypeVar) else spec
        if dtype != wanted:
            raise PlanError(f"input `{name}` has dtype {dtype}, expected {wanted}")
    if isinstance(declared, ScalarType):
        if len(shape) != 0:
            raise PlanError(f"input `{name}` must be a scalar")
        return
    dims, pack = align_shape(declared.shape, shape, name)
    if pack is not None:
        unit, sizes = pack
        if env.packs.setdefault(unit.id, tuple(sizes)) != tuple(sizes):
            raise PlanError(f"input `{name}` disagrees on shape pack `{unit.name}`")
    for dim, size in dims:
        if isinstance(dim, DimSymbol):
            if env.dims.setdefault(dim.id, size) != size:
                raise PlanError(
                    f"input `{name}` has size {size} where `{dim.name}` is {env.dims[dim.id]}"
                )
        elif env.dim(dim) != size:
            raise PlanError(
                f"input `{name}` has size {size} on an axis that must be {env.dim(dim)}"
            )


def require_bound(env: Bindings, function: Function) -> None:
    """Checks that the inputs and names bound every generic of `function`
    (one left out takes its constant default) and that its `where` clause
    holds."""
    name = function.short_name
    for generic in function.generics:
        if not bound(env, generic):
            default = default_of(generic)
            if default is None:
                raise PlanError(
                    f"cannot determine `{generic.name}` of `{name}` from its inputs; "
                    f"give it by name, `{name}(..., {generic.name}=...)`"
                )
            if isinstance(default, int):
                env.dims[generic.id] = default
            else:
                env.dtypes[generic.id] = default
        if generic.kind == "dtype":
            kind = generic.dtype_class or "any"
            if env.dtypes[generic.id] not in CLASSES[kind]:
                raise PlanError(
                    f"`{generic.name}` of `{name}` is {kind}, not {env.dtypes[generic.id]}"
                )
    for constraint in function.constraints:
        if not env.holds(constraint):
            raise PlanError(f"the inputs break the `where` clause of `{name}`")


def bind_names(env: Bindings, generics: Sequence[Generic]) -> dict[str, str]:
    """The generics `env` binds as `--bind` values (a shape pack as `2,3`)."""
    names: dict[str, str] = {}
    for generic in generics:
        if not bound(env, generic):
            continue
        if generic.kind == "dim":
            names[generic.name] = str(env.dims[generic.id])
        elif generic.kind == "dtype":
            names[generic.name] = env.dtypes[generic.id]
        else:
            names[generic.name] = ",".join(map(str, env.packs[generic.id]))
    return names


class Arithmetic(Protocol[T]):
    """What a dimension's operators mean over some kind of value: integers
    (`INTEGERS`), or expressions of free symbols (`linnet.resources`)."""

    def const(self, value: int) -> T: ...
    def total(self, args: Sequence[T]) -> T: ...
    def product(self, args: Sequence[T]) -> T: ...
    def floordiv(self, a: T, b: T) -> T: ...
    def mod(self, a: T, b: T) -> T: ...
    def minimum(self, args: Sequence[T]) -> T: ...
    def maximum(self, args: Sequence[T]) -> T: ...


class _Integers:
    def const(self, value: int) -> int:
        return value

    def total(self, args: Sequence[int]) -> int:
        return sum(args)

    def product(self, args: Sequence[int]) -> int:
        return math.prod(args)

    def floordiv(self, a: int, b: int) -> int:
        if b == 0:
            raise PlanError("division by zero in a dimension")
        return a // b

    def mod(self, a: int, b: int) -> int:
        if b == 0:
            raise PlanError("division by zero in a dimension")
        return a % b

    def minimum(self, args: Sequence[int]) -> int:
        return min(args)

    def maximum(self, args: Sequence[int]) -> int:
        return max(args)


INTEGERS: Arithmetic[int] = _Integers()


def fold_dim(
    dim: Dim,
    symbol: Callable[[DimSymbol], T],
    pack_size: Callable[[PackSize], T],
    arithmetic: Arithmetic[T],
) -> T:
    """`dim` evaluated with each symbol's and pack's value from `symbol` and
    `pack_size`, its operators as `arithmetic` computes them."""
    if isinstance(dim, int):
        return arithmetic.const(dim)
    if isinstance(dim, DimSymbol):
        return symbol(dim)
    if isinstance(dim, PackSize):
        return pack_size(dim)
    args = [fold_dim(a, symbol, pack_size, arithmetic) for a in dim.args]
    if dim.op == "add":
        return arithmetic.total(args)
    if dim.op == "mul":
        return arithmetic.product(args)
    if dim.op == "floordiv":
        return arithmetic.floordiv(args[0], args[1])
    if dim.op == "mod":
        return arithmetic.mod(args[0], args[1])
    return arithmetic.minimum(args) if dim.op == "min" else arithmetic.maximum(args)


def evaluate_dim(dim: Dim, bindings: Bindings) -> int:
    def symbol(found: DimSymbol) -> int:
        if found.id not in bindings.dims:
            raise PlanError(f"dimension `{found.name}` is not bound")
        return bindings.dims[found.id]

    def pack_size(found: PackSize) -> int:
        if found.id not in bindings.packs:
            raise PlanError(f"shape pack `{found.name}` is not bound")
        return math.prod(bindings.packs[found.id])

    return fold_dim(dim, symbol, pack_size, INTEGERS)


def evaluate_shape(shape: Shape, bindings: Bindings) -> tuple[int, ...]:
    sizes: list[int] = []
    for unit in shape:
        if isinstance(unit, Pack):
            if unit.id not in bindings.packs:
                raise PlanError(f"shape pack `{unit.name}` is not bound")
            sizes.extend(bindings.packs[unit.id])
        else:
            sizes.append(evaluate_dim(unit, bindings))
    return tuple(sizes)


def evaluate_dtype(dtype: DType, bindings: Bindings) -> str:
    if isinstance(dtype, str):
        return dtype
    if dtype.id not in bindings.dtypes:
        raise PlanError(f"dtype `{dtype.name}` is not bound")
    return bindings.dtypes[dtype.id]


def repeat_paths(path: str, counts: Sequence[int]) -> list[str]:
    """`layers[*].w` repeated `(2,)` is `layers.0.w`, `layers.1.w`."""
    paths = [path]
    for count in counts:
        paths = [p.replace("[*]", f".{i}", 1) for p in paths for i in range(count)]
    return paths


def chosen_paths(
    paths: Iterable[str],
    trainable: bool | str | Sequence[str],
    tied: Mapping[str, str] | None = None,
) -> list[str]:
    """The parameter paths a `trainable` choice picks: every one for True,
    none for False, those matching a glob pattern (or any of several)
    otherwise. Paths bound to one tensor (`tied[path]` names the path that
    holds it) are picked together, when any of them matches."""
    listed = list(paths)
    if trainable is True:
        return listed
    if trainable is False:
        return []
    patterns = [trainable] if isinstance(trainable, str) else list(trainable)
    links = tied or {}
    picked = {
        links.get(path, path)
        for path in listed
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
    }
    return [path for path in listed if links.get(path, path) in picked]


def shared_paths(values: Iterable[tuple[str, object]]) -> dict[str, str]:
    """Each path whose object (a parameter, an array) an earlier path holds
    too, to that earlier path: the ties `chosen_paths` takes."""
    first: dict[int, str] = {}
    tied: dict[str, str] = {}
    for path, value in values:
        holder = first.setdefault(id(value), path)
        if holder != path:
            tied[path] = holder
    return tied


def choose_entry(
    block: str, names: Sequence[str], name: str | None, prefer: str | None = None
) -> str:
    """Entry `name` of `block`, which has entries `names`; without a name,
    the only one, or else `prefer` when it has that one."""
    if name is not None:
        if name not in names:
            raise LinnetError(f"block `{block}` has no entry `{name}`")
        return name
    if len(names) == 1:
        return names[0]
    if prefer is not None and prefer in names:
        return prefer
    raise LinnetError(f"block `{block}` has {len(names)} entries: {', '.join(names)}; name one")


def expand_paths(entry: ManifestEntry, bindings: Bindings) -> list[str]:
    """A manifest entry's paths, one per element of the arrays it is in."""
    return repeat_paths(entry.path, [evaluate_dim(r, bindings) for r in entry.repeat])


def parameter_count(
    program: Program,
    values: Mapping[str, int | str],
    mapping: Mapping[str, str] | None = None,
) -> int:
    """The number of parameter elements once the root generics are bound.
    With `mapping` (path -> checkpoint tensor), a tensor several paths are
    bound to (a tied embedding) counts once, as published figures do."""
    bindings = bind_generics(program.root.generics, values)
    counted: set[str] = set()
    total = 0
    for entry in program.manifest:
        if entry.kind != "param":
            continue
        elements = 1
        for size in evaluate_shape(entry.shape, bindings):
            elements *= size
        for path in expand_paths(entry, bindings):
            source = path if mapping is None else mapping.get(path, path)
            if source not in counted:
                counted.add(source)
                total += elements
    return total


# ------------------------------------------------------------------ printing

_PRECEDENCE = {"add": 1, "mul": 2, "floordiv": 2, "mod": 2}
_SYMBOL = {"add": " + ", "mul": " * ", "floordiv": " / ", "mod": " % "}


def format_dim(dim: Dim) -> str:
    """A dimension as the language spells it: `KvHeads * (H / Heads)`."""
    if isinstance(dim, int):
        return str(dim)
    if isinstance(dim, DimSymbol | PackSize):
        return dim.name
    if dim.op in ("min", "max"):
        return f"{dim.op}({', '.join(format_dim(a) for a in dim.args)})"
    own = _PRECEDENCE[dim.op]
    parts: list[str] = []
    for index, arg in enumerate(dim.args):
        text = format_dim(arg)
        if isinstance(arg, DimExpr) and arg.op not in ("min", "max"):
            inner = _PRECEDENCE[arg.op]
            # Integer division does not associate with multiplication, so a
            # division keeps its parentheses on either side of `*`, `/`, `%`.
            divides = arg.op in ("floordiv", "mod")
            if inner < own or (inner == own and (divides or (index > 0 and dim.op != "mul"))):
                text = f"({text})"
        parts.append(text)
    return _SYMBOL[dim.op].join(parts)


def format_shape(shape: Shape) -> str:
    return ", ".join(f"*{u.name}" if isinstance(u, Pack) else format_dim(u) for u in shape)


def format_dtype(dtype: DType) -> str:
    return dtype if isinstance(dtype, str) else dtype.name


def format_generic_arg(arg: GenericArg) -> str:
    if isinstance(arg, DimArg):
        return format_dim(arg.dim)
    if isinstance(arg, ShapeArg):
        return f"[{format_shape(arg.shape)}]"
    return format_dtype(arg.dtype)


def format_type(type: Type) -> str:
    """A type as the language spells it: `Tensor[B, S, H; T]`, `[Layer<H>; N]`, `T?`."""
    if isinstance(type, ScalarType):
        return format_dtype(type.dtype)
    if isinstance(type, TensorType):
        return f"Tensor[{format_shape(type.shape)}; {format_dtype(type.dtype)}]"
    if isinstance(type, TupleType):
        return "(" + ", ".join(format_type(e) for e in type.elements) + ")"
    if isinstance(type, OptionalType):
        return format_type(type.inner) + "?"
    if isinstance(type, ArrayType):
        return f"[{format_type(type.element)}; {format_dim(type.length)}]"
    if isinstance(type, NamedType):
        args = ", ".join(format_generic_arg(a) for a in type.args)
        return type.name + (f"<{args}>" if args else "")
    if isinstance(type, ShapeType):
        return f"Shape[{format_shape(type.shape)}]"
    return "()"


def format_generics(generics: Sequence[Generic]) -> str:
    """`<In: Dim, *S: Shape, T: Float = bf16>`, or an empty string."""
    if not generics:
        return ""
    parts: list[str] = []
    for generic in generics:
        if generic.kind == "dim":
            text = f"{generic.name}: Dim"
        elif generic.kind == "shape":
            text = f"*{generic.name}: Shape"
        else:
            classes = {"float": "Float", "integer": "Int", "numeric": "Numeric", "any": "DType"}
            text = f"{generic.name}: {classes.get(generic.dtype_class or 'any', 'DType')}"
        if generic.default is not None:
            text += f" = {format_generic_arg(generic.default)}"
        parts.append(text)
    return "<" + ", ".join(parts) + ">"


def format_signature(function: Function) -> str:
    """`pub entry forward<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T]`."""
    params = ", ".join(f"{p.name}: {format_type(p.type)}" for p in function.params)
    results = [format_type(t) for t in function.results]
    result = ""
    if results:
        result = " -> " + (results[0] if len(results) == 1 else f"({', '.join(results)})")
    visibility = "pub " if function.pub else ""
    head = f"{visibility}{function.kind} {function.short_name}{format_generics(function.generics)}"
    return f"{head}({params}){result}"
