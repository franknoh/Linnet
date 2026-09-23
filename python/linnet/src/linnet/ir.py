"""A typed view of a plan: the compiled program as immutable dataclasses.

`Program.from_json` reads what `linnet plan` prints (see `docs/plan-format.md`)
into frozen dataclasses with exhaustive unions for dimensions and types, so
that a backend, a diagram, or a model card can walk the program without
indexing JSON. Dimensions stay symbolic; `format_type` prints them the way
the language spells them (`Tensor[B, S, H; bf16]`).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from .compiler import LinnetError
from .plan import compile_plan

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
    attrs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
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
    kind: Literal["fn", "op", "entry"]
    block: str | None
    pub: bool
    generics: tuple[Generic, ...]
    constraints: tuple[Constraint, ...]
    results: tuple[Type, ...]
    states: tuple[str, ...]
    body: Region

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

    @staticmethod
    def from_json(text: str) -> Program:
        document: object = json.loads(text)
        if not isinstance(document, dict):
            raise LinnetError("a plan is a JSON object")
        return _Reader().program(cast(dict[str, Any], document))

    @property
    def root_block(self) -> Block:
        return self.blocks[self.root.name]

    def entries(self, block: str | None = None) -> tuple[Function, ...]:
        """The entries of a block (the root by default), in declaration order."""
        name = self.root.name if block is None else block
        return tuple(f for f in self.functions.values() if f.kind == "entry" and f.block == name)

    def methods(self, block: str) -> tuple[Function, ...]:
        return tuple(f for f in self.functions.values() if f.block == block)

    def method(self, block: str, name: str) -> Function:
        for function in self.functions.values():
            if function.block == block and function.name.endswith(f"::{block}.{name}"):
                return function
        raise LinnetError(f"block `{block}` has no method `{name}`")

    def entry(self, name: str | None = None) -> Function:
        """The root entry called `name`, or the only one."""
        entries = self.entries()
        if name is None:
            if len(entries) != 1:
                names = ", ".join(e.short_name for e in entries)
                raise LinnetError(f"block `{self.root.name}` has {len(entries)} entries: {names}")
            return entries[0]
        for entry in entries:
            if entry.short_name == name:
                return entry
        raise LinnetError(f"block `{self.root.name}` has no entry `{name}`")

    def parameters(self) -> tuple[ManifestEntry, ...]:
        return tuple(e for e in self.manifest if e.kind == "param")

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
    plan = compile_plan(source, root=root, std_root=std_root, optimize=optimize, numerics=numerics)
    return Program.from_json(plan.text)


# --------------------------------------------------------------------- reader


class _Reader:
    def program(self, document: dict[str, Any]) -> Program:
        version = int(document.get("version", 0))
        if version != 1:
            raise LinnetError(f"unsupported plan version {version!r}")
        blocks = {
            str(name): self.block(str(name), cast(dict[str, Any], data))
            for name, data in cast(dict[str, Any], document["blocks"]).items()
        }
        functions = [self.function(cast(dict[str, Any], f)) for f in document["functions"]]
        root = cast(dict[str, Any], document["root"])
        return Program(
            version=version,
            module=str(document["module"]),
            root=Root(
                name=str(root["name"]),
                generics=self.generics(root.get("generics", [])),
                constraints=self.constraints(root.get("constraints", [])),
            ),
            manifest=tuple(
                self.manifest_entry(cast(dict[str, Any], e)) for e in document["manifest"]
            ),
            blocks=MappingProxyType(blocks),
            functions=MappingProxyType({f.name: f for f in functions}),
            constants=tuple(
                self.constant(cast(dict[str, Any], c)) for c in document.get("constants", [])
            ),
        )

    # ---- dimensions and types

    def dim(self, data: Any) -> Dim:
        if isinstance(data, bool):
            raise LinnetError("a dimension cannot be a boolean")
        if isinstance(data, int):
            return data
        node = cast(dict[str, Any], data)
        if "sym" in node:
            return DimSymbol(int(node["sym"]), str(node.get("name", "")))
        if "packsize" in node:
            return PackSize(int(node["packsize"]), str(node.get("name", "")))
        op = str(node["op"])
        if op not in ("add", "mul", "floordiv", "mod", "min", "max"):
            raise LinnetError(f"unknown dimension operator `{op}`")
        return DimExpr(op, tuple(self.dim(arg) for arg in node["args"]))

    def unit(self, data: Any) -> Unit:
        if isinstance(data, dict) and "pack" in data:
            node = cast(dict[str, Any], data)
            return Pack(int(node["pack"]), str(node.get("name", "")))
        return self.dim(data)

    def shape(self, data: Sequence[Any]) -> Shape:
        return tuple(self.unit(unit) for unit in data)

    def dtype(self, data: Any) -> DType:
        if isinstance(data, str):
            return data
        node = cast(dict[str, Any], data)
        return DTypeVar(int(node["var"]), str(node.get("name", "")))

    def generic_arg(self, data: dict[str, Any]) -> GenericArg:
        if "dim" in data:
            return DimArg(self.dim(data["dim"]))
        if "shape" in data:
            return ShapeArg(self.shape(data["shape"]))
        return DTypeArg(self.dtype(data["dtype"]))

    def type(self, data: Any) -> Type:
        node = cast(dict[str, Any], data)
        kind = str(node["kind"])
        if kind == "scalar":
            return ScalarType(self.dtype(node["dtype"]))
        if kind == "tensor":
            return TensorType(self.shape(node["shape"]), self.dtype(node["dtype"]))
        if kind == "tuple":
            return TupleType(tuple(self.type(e) for e in node["elements"]))
        if kind == "optional":
            return OptionalType(self.type(node["inner"]))
        if kind == "array":
            return ArrayType(self.type(node["element"]), self.dim(node["length"]))
        if kind in ("block", "struct", "enum"):
            return NamedType(
                kind,
                str(node["name"]),
                str(node.get("module", "")),
                tuple(self.generic_arg(cast(dict[str, Any], a)) for a in node.get("args", [])),
            )
        if kind == "shape":
            return ShapeType(self.shape(node["shape"]))
        if kind == "unit":
            return UnitType()
        raise LinnetError(f"unknown type kind `{kind}`")

    # ---- declarations

    def generics(self, data: Sequence[Any]) -> tuple[Generic, ...]:
        out: list[Generic] = []
        for item in data:
            node = cast(dict[str, Any], item)
            kind = str(node["kind"])
            if kind not in ("dim", "shape", "dtype"):
                raise LinnetError(f"unknown generic kind `{kind}`")
            default = node.get("default")
            dtype_class = node.get("class")
            out.append(
                Generic(
                    name=str(node["name"]),
                    kind=kind,
                    id=int(node["var"] if kind == "dtype" else node["sym"]),
                    default=None
                    if default is None
                    else self.generic_arg(cast(dict[str, Any], default)),
                    dtype_class=None
                    if dtype_class is None
                    else cast(Literal["float", "integer", "numeric", "any"], str(dtype_class)),
                )
            )
        return tuple(out)

    def constraints(self, data: Sequence[Any]) -> tuple[Constraint, ...]:
        out: list[Constraint] = []
        for item in data:
            node = cast(dict[str, Any], item)
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

    def block(self, name: str, data: dict[str, Any]) -> Block:
        members: list[Member] = []
        for item in data.get("members", []):
            node = cast(dict[str, Any], item)
            kind = str(node["kind"])
            if kind not in ("param", "buffer", "state", "sub"):
                raise LinnetError(f"unknown member kind `{kind}`")
            members.append(Member(str(node["name"]), kind, self.type(node["type"])))
        return Block(
            name=name,
            module=str(data.get("module", "")),
            pub=bool(data.get("pub", False)),
            generics=self.generics(data.get("generics", [])),
            constraints=self.constraints(data.get("constraints", [])),
            members=tuple(members),
        )

    def manifest_entry(self, data: dict[str, Any]) -> ManifestEntry:
        kind = str(data["kind"])
        if kind not in ("param", "buffer", "state"):
            raise LinnetError(f"unknown manifest kind `{kind}`")
        return ManifestEntry(
            path=str(data["path"]),
            kind=kind,
            dtype=self.dtype(data["dtype"]),
            shape=self.shape(data["shape"]),
            repeat=tuple(self.dim(d) for d in data.get("repeat", [])),
            optional=bool(data.get("optional", False)),
        )

    def substitution(self, data: Any) -> Substitution:
        node = cast(dict[str, Any], data or {})
        dims = cast(dict[str, Any], node.get("dims", {}))
        packs = cast(dict[str, Any], node.get("packs", {}))
        dtypes = cast(dict[str, Any], node.get("dtypes", {}))
        return Substitution(
            dims=MappingProxyType({int(k): self.dim(v) for k, v in dims.items()}),
            packs=MappingProxyType({int(k): self.shape(v) for k, v in packs.items()}),
            dtypes=MappingProxyType({int(k): self.dtype(v) for k, v in dtypes.items()}),
        )

    def value(self, data: Any) -> Value:
        node = cast(dict[str, Any], data)
        return Value(int(node["id"]), str(node.get("name", "")), self.type(node["type"]))

    def region(self, data: Any) -> Region:
        node = cast(dict[str, Any], data)
        return Region(
            args=tuple(self.value(v) for v in node.get("args", [])),
            ops=tuple(self.op(cast(dict[str, Any], o)) for o in node.get("ops", [])),
        )

    def op(self, data: dict[str, Any]) -> Op:
        attrs = cast(dict[str, Any], data.get("attrs") or {})
        return Op(
            kind=str(data["kind"]),
            operands=tuple(int(i) for i in data.get("operands", [])),
            results=tuple(self.value(v) for v in data.get("results", [])),
            attrs=MappingProxyType(dict(attrs)),
            regions=tuple(self.region(r) for r in data.get("regions", [])),
        )

    def function(self, data: dict[str, Any]) -> Function:
        kind = str(data["kind"])
        if kind not in ("fn", "op", "entry"):
            raise LinnetError(f"unknown function kind `{kind}`")
        block = data.get("block")
        return Function(
            name=str(data["name"]),
            kind=kind,
            block=None if block is None else str(block),
            pub=bool(data.get("pub", False)),
            generics=self.generics(data.get("generics", [])),
            constraints=self.constraints(data.get("constraints", [])),
            results=tuple(self.type(t) for t in data.get("results", [])),
            states=tuple(str(s) for s in data.get("states", [])),
            body=self.region(data["body"]),
        )

    def constant(self, data: dict[str, Any]) -> Constant:
        type_data = data.get("type")
        return Constant(
            name=str(data["name"]),
            pub=bool(data.get("pub", False)),
            type=None if type_data is None else self.type(type_data),
            contextual=bool(data.get("contextual", False)),
            body=self.region(data["body"]),
        )


def parse_substitution(data: Any) -> Substitution:
    """Reads a call's `substitution` attribute."""
    return _Reader().substitution(data)


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


@dataclass(frozen=True, slots=True)
class Bindings:
    """Concrete values for generics, by symbol id: what `--bind` gives the compiler."""

    dims: Mapping[int, int]
    packs: Mapping[int, tuple[int, ...]]
    dtypes: Mapping[int, str]


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
            default = generic.default
            if default is None:
                raise LinnetError(f"generic `{generic.name}` needs a value")
            if isinstance(default, DimArg) and isinstance(default.dim, int):
                dims[generic.id] = default.dim
            elif isinstance(default, DTypeArg) and isinstance(default.dtype, str):
                dtypes[generic.id] = default.dtype
            else:
                raise LinnetError(f"generic `{generic.name}` needs a value")
            continue
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
    return Bindings(MappingProxyType(dims), MappingProxyType(packs), MappingProxyType(dtypes))


def evaluate_dim(dim: Dim, bindings: Bindings) -> int:
    if isinstance(dim, int):
        return dim
    if isinstance(dim, DimSymbol):
        if dim.id not in bindings.dims:
            raise LinnetError(f"dimension `{dim.name}` is not bound")
        return bindings.dims[dim.id]
    if isinstance(dim, PackSize):
        if dim.id not in bindings.packs:
            raise LinnetError(f"shape pack `{dim.name}` is not bound")
        count = 1
        for size in bindings.packs[dim.id]:
            count *= size
        return count
    args = [evaluate_dim(a, bindings) for a in dim.args]
    if dim.op == "add":
        return sum(args)
    if dim.op == "mul":
        product = 1
        for a in args:
            product *= a
        return product
    if dim.op in ("floordiv", "mod"):
        if args[1] == 0:
            raise LinnetError("division by zero in a dimension")
        return args[0] // args[1] if dim.op == "floordiv" else args[0] % args[1]
    return min(args) if dim.op == "min" else max(args)


def evaluate_shape(shape: Shape, bindings: Bindings) -> tuple[int, ...]:
    sizes: list[int] = []
    for unit in shape:
        if isinstance(unit, Pack):
            if unit.id not in bindings.packs:
                raise LinnetError(f"shape pack `{unit.name}` is not bound")
            sizes.extend(bindings.packs[unit.id])
        else:
            sizes.append(evaluate_dim(unit, bindings))
    return tuple(sizes)


def evaluate_dtype(dtype: DType, bindings: Bindings) -> str:
    if isinstance(dtype, str):
        return dtype
    if dtype.id not in bindings.dtypes:
        raise LinnetError(f"dtype `{dtype.name}` is not bound")
    return bindings.dtypes[dtype.id]


def parameter_count(program: Program, values: Mapping[str, int | str]) -> int:
    """The number of parameter elements once the root generics are bound."""
    bindings = bind_generics(program.root.generics, values)
    total = 0
    for entry in program.manifest:
        if entry.kind != "param":
            continue
        count = 1
        for size in evaluate_shape(entry.shape, bindings):
            count *= size
        for repeat in entry.repeat:
            count *= evaluate_dim(repeat, bindings)
        total += count
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
