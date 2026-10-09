"""What the importers share (`linnet.torch.export_linnet`,
`linnet.jax.export_linnet`, `linnet.onnx.import_onnx`): Linnet identifiers
for foreign names, the blocks a tree of parameters becomes, and writing the
result as Linnet source the compiler has checked.

Each importer reads its own tree (a module, a parameter dict, initializer
names) into `Member`s; `Hierarchy` turns them into blocks, one per distinct
shape of subtree, and says how a parameter's type is written. A
`PlanBuilder` of its own writes the operations, in the importer's own
values, types, dtypes and dimensions.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, Protocol, TypedDict, TypeVar

from .compiler import LinnetError, run_compiler, std_arguments

KEYWORDS = frozenset(
    {
        "as", "const", "type", "struct", "enum", "fn", "op", "block", "entry", "param", "buffer",
        "state", "sub", "let", "var", "return", "if", "else", "match", "static", "for", "in",
        "while", "where", "true", "false", "none", "some", "extern", "module", "use", "pub",
        "yield",
    }
)  # fmt: skip
PRELUDE = frozenset(
    {
        "Tensor", "Dim", "Shape", "DType", "Numeric", "Integer", "Float", "cast", "reshape",
        "permute", "broadcast_to", "concat", "pad", "iota", "fill", "gather", "scatter", "exp",
        "log", "sqrt", "rsqrt", "sin", "cos", "tanh", "abs", "select", "min", "max", "sum", "prod",
        "any", "all", "bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16",
        "f32", "f64",
    }
)  # fmt: skip
RESERVED = KEYWORDS | PRELUDE


def identifier(name: str) -> str:
    """A Linnet identifier for a foreign attribute, class or tensor name."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not clean or clean[0].isdigit():
        clean = "_" + clean
    if clean in RESERVED:
        clean += "_"
    return clean


@dataclass
class Block:
    """A Linnet block synthesized from a subtree."""

    name: str
    members: list[dict[str, object]] = field(default_factory=lambda: [])


@dataclass
class Member:
    """How one node of the foreign tree maps onto a block member."""

    kind: str  # "param" | "buffer" | "sub"
    name: str
    block: Block | None = None  # sub: the child's block
    length: int | None = None  # sub: array length when the children are uniform
    element: Block | None = None  # sub: the array element block
    children: dict[str, Member] = field(default_factory=lambda: {})  # sub: by foreign name
    leaf: object = None  # param: the importer's own description of its tensor


class ValueJson(TypedDict):
    """A value as the plan writes it."""

    id: int
    name: str
    type: dict[str, object]


class OpJson(TypedDict):
    """An operation as the plan writes it."""

    kind: str
    operands: list[int]
    results: list[ValueJson]
    attrs: dict[str, object]
    regions: list[dict[str, object]]


class Hierarchy:
    """The blocks of an imported model, one per distinct subtree, named
    after the foreign class or key and kept unique."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.blocks: dict[str, Block] = {}
        self._by_signature: dict[tuple[str, Hashable], Block] = {}
        # Linnet path -> the foreign name, where they differ (`bindings.json`).
        self.renamed: dict[str, str] = {}

    def leaf_type(self, member: Member) -> dict[str, object] | None:
        """A parameter's plan type; None for one the importer fills in later."""
        return None

    def block_for(self, class_name: str, signature: Hashable, member: Member) -> Block:
        """The block for `member`'s subtree: the same one for every subtree of
        the same class name and signature."""
        key = (class_name, signature)
        if key in self._by_signature:
            return self._by_signature[key]
        base = identifier(class_name)
        name = base
        for suffix in range(2, 1000):
            if name not in self.blocks:
                break
            name = f"{base}_{suffix}"
        block = Block(name)
        self.blocks[name] = block
        self._by_signature[key] = block
        for child in member.children.values():
            kind_type = self.member_type(child) if child.kind == "sub" else self.leaf_type(child)
            block.members.append({"name": child.name, "kind": child.kind, "type": kind_type})
        return block

    def block_type(self, block: Block) -> dict[str, object]:
        return {"kind": "block", "name": block.name, "module": self.module, "args": []}

    def member_type(self, member: Member) -> dict[str, object]:
        """The plan type of a `sub` member: a block, or an array of one."""
        if member.length is not None:
            assert member.element is not None
            return {
                "kind": "array",
                "element": self.block_type(member.element),
                "length": member.length,
            }
        assert member.block is not None
        return self.block_type(member.block)

    def plan(
        self,
        root: str,
        args: list[ValueJson],
        ops: list[OpJson],
        result: dict[str, object],
        generics: Sequence[Mapping[str, object]] = (),
        constraints: Sequence[Mapping[str, object]] = (),
    ) -> dict[str, object]:
        """The imported model's plan: every block, and the root's `forward`
        entry over `args` (`self` first), running `ops` and returning a
        value of type `result`."""
        blocks = {
            name: {
                "module": self.module,
                "pub": True,
                "generics": [],
                "constraints": [],
                "members": block.members,
            }
            for name, block in self.blocks.items()
        }
        return {
            "version": 1,
            "module": self.module,
            "root": {"name": root, "generics": [], "constraints": []},
            "manifest": [],
            "blocks": blocks,
            "functions": [
                {
                    "name": f"{self.module}::{root}.forward",
                    "kind": "entry",
                    "block": root,
                    "pub": True,
                    "generics": list(generics),
                    "constraints": list(constraints),
                    "results": [result],
                    "body": {"args": args, "ops": ops},
                }
            ],
            "constants": [],
        }

    def weights_files(self, directory: str | Path) -> tuple[Path, Path | None]:
        """Where the weights go under `directory` (made if need be), and the
        `bindings.json` written there when a path was renamed."""
        from .weights import write_bindings

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        bindings = None
        if self.renamed:
            bindings = write_bindings(root / "bindings.json", dict(sorted(self.renamed.items())))
        return root / "model.safetensors", bindings


def refuse_unsupported(unsupported: Sequence[str], what: str, error: type[LinnetError]) -> None:
    """Raises `error` naming every operation of `what` (the model, the
    graph) that has no Linnet mapping, if there is one."""
    if unsupported:
        raise error(
            f"{what} uses operations without a Linnet mapping:\n  "
            + "\n  ".join(sorted(set(unsupported)))
        )


class _Identified(Protocol):
    id: int


V = TypeVar("V", bound=_Identified)  # a value
K = TypeVar("K")  # a value's type
T = TypeVar("T")  # a dtype
D = TypeVar("D")  # a dimension


class PlanBuilder(ABC, Generic[V, K, T, D]):
    """Accumulates a plan's operations and regions. An importer says how its
    values, their types, its dtypes and its dimensions are made and
    written."""

    error: type[LinnetError] = LinnetError

    def __init__(self) -> None:
        self.next_id = 0
        self.regions: list[list[OpJson]] = [[]]

    # ---- what each importer says

    @abstractmethod
    def kind(self, dtype: T, shape: tuple[D, ...] | None) -> K:
        """The type of a value of `dtype`: a tensor of `shape`, or a scalar
        without one."""

    @abstractmethod
    def make(self, id: int, kind: K, type_json: dict[str, object] | None) -> V:
        """A value of the plan; `type_json` is its plan type when `kind`
        cannot say it (a block, a tuple)."""

    @abstractmethod
    def kind_json(self, kind: K) -> dict[str, object]:
        """A type as the plan writes it."""

    @abstractmethod
    def kind_of(self, value: V) -> K: ...

    @abstractmethod
    def dtype_of(self, value: V) -> T: ...

    @abstractmethod
    def literal(self, dtype: T) -> Literal["bool", "int", "float"]:
        """Which constant a value of `dtype` is written as."""

    @abstractmethod
    def dim_json(self, dim: D) -> int | Mapping[str, object]:
        """A size, or a symbolic dimension as the plan writes it."""

    @property
    @abstractmethod
    def index_dtype(self) -> T:
        """`i64`: what an index, a dimension and an opaque value are."""

    # ---- values and operations

    def type_of(self, value: V) -> dict[str, object]:
        return self.kind_json(self.kind_of(value))

    def fresh(self, kind: K, type_json: dict[str, object] | None = None) -> V:
        self.next_id += 1
        return self.make(self.next_id - 1, kind, type_json)

    def value_json(
        self, value: V, name: str = "", type_json: dict[str, object] | None = None
    ) -> ValueJson:
        return {"id": value.id, "name": name, "type": type_json or self.type_of(value)}

    def op(
        self,
        kind: str,
        operands: Sequence[V],
        result: K | None,
        attrs: dict[str, object] | None = None,
        regions: Sequence[dict[str, object]] = (),
        name: str = "",
        type_json: dict[str, object] | None = None,
    ) -> V:
        results = [self.fresh(result, type_json)] if result is not None else []
        self.regions[-1].append(
            {
                "kind": kind,
                "operands": [v.id for v in operands],
                "results": [self.value_json(v, name, type_json) for v in results],
                "attrs": attrs or {},
                "regions": list(regions),
            }
        )
        return results[0] if results else self.make(-1, self.kind(self.index_dtype, None), None)

    def const(self, value: float | int | bool, dtype: T) -> V:
        scalar = self.kind(dtype, None)
        literal = self.literal(dtype)
        if literal == "bool":
            return self.op("const.bool", [], scalar, {"value": 1 if value else 0})
        if literal == "float":
            return self.op("const.float", [], scalar, {"value": float(value)})
        return self.op("const.int", [], scalar, {"value": int(value)})

    def const_dim(self, dim: D) -> V:
        index = self.kind(self.index_dtype, None)
        return self.op("const.dim", [], index, {"value": self.dim_json(dim)})

    def region(
        self, arguments: Sequence[tuple[str, K]], body: Callable[[list[V]], V]
    ) -> dict[str, object]:
        """Runs `body` in a new region of `arguments`; the value it returns
        is yielded."""
        values = [self.fresh(kind) for _, kind in arguments]
        self.regions.append([])
        yielded = body(values)
        self.op("yield", [yielded], None)
        ops = self.regions.pop()
        return {
            "args": [self.value_json(v, n) for v, (n, _) in zip(values, arguments, strict=True)],
            "ops": ops,
        }

    def _indices(self, indices: Sequence[tuple[str, D]]) -> list[dict[str, object]]:
        return [{"name": n, "domain": [self.dim_json(d)]} for n, d in indices]

    def comprehension(
        self,
        indices: Sequence[tuple[str, D]],
        dtype: T,
        body: Callable[[list[V]], V],
        name: str = "",
    ) -> V:
        """`let out[i, j, ...] = body(i, j, ...)` over the given index domains."""
        index = self.kind(self.index_dtype, None)
        region = self.region([(n, index) for n, _ in indices], body)
        return self.op(
            "comprehension",
            [],
            self.kind(dtype, tuple(d for _, d in indices)),
            {"indices": self._indices(indices)},
            [region],
            name,
        )

    def reduce(
        self,
        kind: str,
        indices: Sequence[tuple[str, D]],
        dtype: T,
        body: Callable[[list[V]], V],
    ) -> V:
        index = self.kind(self.index_dtype, None)
        region = self.region([(n, index) for n, _ in indices], body)
        return self.op(
            "reduce",
            [],
            self.kind(dtype, None),
            {"indices": self._indices(indices), "reduce": kind},
            [region],
        )

    def element(self, tensor: V, indices: Sequence[V]) -> V:
        return self.op("tensor.element", [tensor, *indices], self.kind(self.dtype_of(tensor), None))

    def reduce_axes(
        self,
        source: V,
        shape: Sequence[D],
        reduced: Sequence[int],
        kind: str,
        dtype: T,
        divisor: D | None = None,
        name: str = "",
    ) -> V:
        """`source` (of `shape`) reduced by `kind` over the `reduced` axes
        (ascending) in `dtype`, then divided by `divisor` for a mean: a
        scalar when no axis is kept, else a comprehension over the kept
        ones named `name`."""
        rank = len(shape)
        scalar = self.kind(dtype, None)

        def body(outer: list[V]) -> V:
            def inner(inside: list[V]) -> V:
                outer_iter, inner_iter = iter(outer), iter(inside)
                indices = [
                    next(inner_iter) if axis in reduced else next(outer_iter)
                    for axis in range(rank)
                ]
                element = self.element(source, indices)
                if self.dtype_of(element) != dtype:
                    element = self.op("cast", [element], scalar)
                return element

            total = self.reduce(kind, [(f"r{a}", shape[a]) for a in reduced], dtype, inner)
            if divisor is not None:
                count = self.op("cast", [self.const_dim(divisor)], scalar)
                total = self.op("div", [total, count], scalar)
            return total

        kept = [axis for axis in range(rank) if axis not in reduced]
        if not kept:
            return body([])
        return self.comprehension([(f"o{a}", shape[a]) for a in kept], dtype, body, name)

    def member_value(
        self,
        hierarchy: Hierarchy,
        root: Member,
        self_value: V,
        path: str,
        kind: K,
        name: str | None = None,
        separators: str = ".",
    ) -> V:
        """The parameter at foreign `path` (its parts split at any of
        `separators`), of type `kind`, loaded from `self_value` through its
        members: an array's element by position, a sub-block by name. The
        value is named `name`, or after the member. A path whose Linnet
        spelling differs is noted in `hierarchy.renamed`."""
        opaque = self.kind(self.index_dtype, None)
        member = root
        current = self_value
        linnet: list[str] = []
        parts = re.split(f"[{re.escape(separators)}]", path)
        for i, part in enumerate(parts):
            child = member.children[part]
            if member.length is not None:
                assert member.element is not None
                index = self.const(int(part), self.index_dtype)
                current = self.op(
                    "array.get",
                    [current, index],
                    opaque,
                    type_json=hierarchy.block_type(member.element),
                )
                linnet.append(part)
                member = child
                continue
            linnet.append(child.name)
            if i == len(parts) - 1:
                if child.kind == "sub":
                    raise self.error(f"`{path}` names a block, not a tensor")
                if ".".join(linnet) != path:
                    hierarchy.renamed[".".join(linnet)] = path
                label = child.name if name is None else name
                return self.op("block.param", [current], kind, {"name": child.name}, name=label)
            current = self.op(
                "block.sub",
                [current],
                opaque,
                {"name": child.name},
                type_json=hierarchy.member_type(child),
            )
            member = child
        raise self.error(f"cannot resolve `{path}`")


def write_source(
    plan: Mapping[str, object],
    output: Path,
    std_root: str | Path | None,
    error: type[LinnetError],
    what: str = "exported",
) -> None:
    """Prints `plan` as Linnet source at `output` (`linnet emit`) and checks
    it (`linnet check`); `error` says what went wrong."""
    try:
        text = run_compiler("emit", "-", stdin=json.dumps(plan))
    except LinnetError as failure:
        raise error(f"the compiler rejected the {what} plan:\n{failure}") from None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    try:
        run_compiler("check", *std_arguments(std_root), str(output))
    except LinnetError as failure:
        raise error(f"the {what} source does not check:\n{failure}") from None


__all__ = [
    "KEYWORDS",
    "PRELUDE",
    "RESERVED",
    "Block",
    "Hierarchy",
    "Member",
    "OpJson",
    "PlanBuilder",
    "ValueJson",
    "identifier",
    "refuse_unsupported",
    "write_source",
]
