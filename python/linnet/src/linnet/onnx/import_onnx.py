"""ONNX -> Linnet: read a graph and write its architecture as a `.linnet`
module.

The graph is translated node by node into the Core IR plan format
(docs/plan-format.md): initializers become parameters whose dotted names
form the block hierarchy (`encoder.layers.0.q.weight`), graph inputs become
the entry's inputs with named symbolic dimensions as generic parameters,
and each node becomes the Linnet primitive or standard-library operation
with the same meaning. Shape computations (`Shape`, `Gather`, `Concat`,
`Unsqueeze` on shapes) are folded at import time into dimension
expressions, so `Reshape` and friends get compile-time shapes. `linnet emit`
prints the plan as source and the result is checked before it is written.

Weights never enter the `.linnet` file: on request the initializers are
saved as SafeTensors under their ONNX names. Nodes without a verified
mapping stop the import with a diagnostic naming every one of them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, TypeVar, cast, overload

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

from ..compiler import LinnetError
from ..dtypes import BY_NUMPY, BY_ONNX, DTYPES
from ..importing import (
    Block,
    Hierarchy,
    Member,
    PlanBuilder,
    identifier,
    refuse_unsupported,
    write_source,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    # A value known at import time: an initializer, or a folded constant.
    _Array: TypeAlias = npt.NDArray[np.generic]

_T = TypeVar("_T")


class OnnxImportError(LinnetError):
    """The graph cannot be expressed in Linnet as captured."""


def _dtype_name(elem_type: int) -> str:
    if elem_type not in BY_ONNX:
        raise OnnxImportError(f"ONNX element type {elem_type} has no Linnet equivalent")
    return BY_ONNX[elem_type].name


def _numpy_dtype_name(array: _Array) -> str:
    found = BY_NUMPY.get(np.dtype(array.dtype).name)
    if found is None:
        raise OnnxImportError(f"dtype {array.dtype} has no Linnet equivalent")
    return found.name


# A dimension: an int, or a symbol name from `dim_param`.
Dim = int | str
# A dimension as the plan writes it: a size, or a numbered symbol.
_DimJson = int | dict[str, int | str]


@dataclass
class _Type:
    shape: tuple[Dim, ...]
    dtype: str

    @property
    def is_scalar(self) -> bool:
        return len(self.shape) == 0


@dataclass
class _Value:
    """A value of the plan. `dims` holds a compile-time shape vector for
    values produced by shape arithmetic; `const` a compile-time array."""

    id: int
    type: _Type
    dims: list[Dim] | None = None
    const: _Array | None = None


class _Symbols:
    """Named symbolic dimensions of the graph, numbered for the plan."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    def declare(self, name: str) -> int:
        if name not in self.ids:
            self.ids[name] = len(self.ids)
        return self.ids[name]

    def json(self, dim: Dim) -> _DimJson:
        if isinstance(dim, int):
            return dim
        if dim not in self.ids:
            raise OnnxImportError(f"dimension `{dim}` does not come from a graph input")
        return {"sym": self.ids[dim], "name": identifier(dim)}


class _Builder(PlanBuilder[_Value, _Type, str, Dim]):
    """The plan's operations, in Linnet dtypes and the graph's dimensions."""

    error = OnnxImportError

    def __init__(self, symbols: _Symbols) -> None:
        super().__init__()
        self.symbols = symbols

    def kind(self, dtype: str, shape: tuple[Dim, ...] | None) -> _Type:
        return _Type(shape or (), dtype)

    def make(self, id: int, kind: _Type, type_json: dict[str, object] | None) -> _Value:
        return _Value(id, kind)

    def kind_json(self, kind: _Type) -> dict[str, object]:
        if kind.is_scalar:
            return {"kind": "scalar", "dtype": kind.dtype}
        return {
            "kind": "tensor",
            "shape": [self.symbols.json(d) for d in kind.shape],
            "dtype": kind.dtype,
        }

    def kind_of(self, value: _Value) -> _Type:
        return value.type

    def dtype_of(self, value: _Value) -> str:
        return value.type.dtype

    def literal(self, dtype: str) -> Literal["bool", "int", "float"]:
        if dtype == "bool":
            return "bool"
        return "float" if DTYPES[dtype].is_float else "int"

    def dim_json(self, dim: Dim) -> _DimJson:
        return self.symbols.json(dim)

    @property
    def index_dtype(self) -> str:
        return "i64"

    def call(
        self,
        callee: str,
        generics: Sequence[Mapping[str, object]],
        operands: list[_Value],
        kind: _Type,
    ) -> _Value:
        return self.op(
            "semantic.call",
            operands,
            kind,
            {
                "callee": callee,
                "substitution": {"dims": {}, "packs": {}, "dtypes": {}},
                "generics": generics,
            },
        )


def _broadcast(shapes: Sequence[tuple[Dim, ...]]) -> list[Dim]:
    """Right-aligned broadcasting over symbolic dimensions."""
    rank = max(len(s) for s in shapes)
    out: list[Dim] = []
    for i in range(rank):
        candidates = [s[len(s) - rank + i] for s in shapes if len(s) - rank + i >= 0]
        chosen: Dim = 1
        for d in candidates:
            if d != 1:
                if chosen != 1 and chosen != d:
                    raise OnnxImportError(f"cannot broadcast {chosen} with {d}")
                chosen = d
        out.append(chosen)
    return out


# The initializers by the parts of their dotted names.
_Tree: TypeAlias = "dict[str, _Tree | _Type]"


class _Hierarchy(Hierarchy):
    """Blocks from the dotted names of the initializers."""

    def leaf_type(self, member: Member) -> dict[str, object] | None:
        leaf = cast(_Type, member.leaf)
        return {"kind": "tensor", "shape": list(leaf.shape), "dtype": leaf.dtype}

    def describe(self, leaves: dict[str, _Type], root_name: str) -> Member:
        tree: _Tree = {}
        for name, kind in leaves.items():
            node = tree
            parts = re.split(r"[./]", name)
            for part in parts[:-1]:
                node = cast(_Tree, node.setdefault(part, {}))
            node[parts[-1]] = kind
        signature, member = self._describe(tree, root_name)
        if member is None:
            raise OnnxImportError("the graph has no initializers to become parameters")
        member.block = self.block_for(root_name, signature, member)
        return member

    def _describe(self, node: _Tree | _Type, name_hint: str) -> tuple[Hashable, Member | None]:
        if isinstance(node, _Type):
            return ("param", node.dtype, node.shape), Member("param", "", leaf=node)
        tree = node
        keys = list(tree)
        if (
            keys
            and all(k.isdigit() for k in keys)
            and sorted(int(k) for k in keys) == list(range(len(keys)))
        ):
            described = [self._describe(tree[str(i)], name_hint) for i in range(len(keys))]
            if len({s for s, _ in described}) == 1 and all(m is not None for _, m in described):
                signature = described[0][0]
                element_name = name_hint.rstrip("s").capitalize() or "Item"
                block = self.block_for(element_name, signature, cast(Member, described[0][1]))
                children = {str(i): cast(Member, m) for i, (_, m) in enumerate(described)}
                for child in children.values():
                    child.block = block
                return ("array", len(keys), signature), Member(
                    "sub", "", length=len(keys), element=block, children=children
                )
        entries: list[tuple[str, Hashable]] = []
        children: dict[str, Member] = {}
        for key in keys:
            signature, child = self._describe(tree[key], key)
            if child is None:
                continue
            child.name = identifier(key)
            if child.kind == "sub" and child.length is None:
                # Numbered children of a mixed container get a block name of
                # their own; a member and its block must not share a name.
                class_name = f"Item{key}" if key.isdigit() else key.capitalize()
                child.block = self.block_for(class_name, signature, child)
            children[key] = child
            entries.append((key, signature))
        if not entries:
            return (), None
        return tuple(entries), Member("sub", "", children=children)


@dataclass
class ImportResult:
    source: Path
    module: str
    root: str
    weights: Path | None
    bindings: Path | None
    plan: dict[str, object]
    notes: list[str]


def import_onnx(
    model: str | Path | onnx.ModelProto,
    *,
    output: str | Path,
    module_name: str | None = None,
    root_name: str = "Model",
    weights: str | Path | None = None,
    std_root: str | Path | None = None,
) -> ImportResult:
    """Writes the ONNX `model` (a path or a `ModelProto`) as Linnet source at
    `output`. With `weights`, the initializers are saved there as SafeTensors
    under their ONNX names (plus `bindings.json` when a name had to change)."""
    output_path = Path(output)
    module = module_name or identifier(output_path.stem)
    proto = (
        # `onnx.load` takes an unparameterized `os.PathLike`.
        onnx.load(str(model))  # pyright: ignore[reportUnknownMemberType]
        if isinstance(model, str | Path)
        else model
    )
    proto = shape_inference.infer_shapes(proto, strict_mode=False)
    translator = _Translator(proto, module, root_name)
    plan = translator.run()

    write_source(plan, output_path, std_root, OnnxImportError, "imported")

    weights_path: Path | None = None
    bindings_path: Path | None = None
    if weights is not None:
        from safetensors.numpy import save_file  # pyright: ignore[reportUnknownVariableType]

        weights_path, bindings_path = translator.hierarchy.weights_files(weights)
        save_file(
            {name: np.ascontiguousarray(array) for name, array in translator.parameters.items()},
            str(weights_path),
        )
    return ImportResult(
        output_path,
        module,
        root_name,
        weights_path,
        bindings_path,
        plan,
        sorted(set(translator.notes)),
    )


class _Translator:
    def __init__(self, proto: onnx.ModelProto, module: str, root_name: str) -> None:
        self.proto = proto
        self.graph = proto.graph
        self.module = module
        self.root_name = root_name
        self.symbols = _Symbols()
        self.builder = _Builder(self.symbols)
        self.hierarchy = _Hierarchy(module)
        self.self_value = self.builder.fresh(_Type((), "f32"))
        self.values: dict[str, _Value] = {}
        self.types: dict[str, _Type] = {}
        self.parameters: dict[str, _Array] = {}  # ONNX name -> array
        self.constants: dict[str, _Array] = {}  # values known at import time
        self.unsupported: list[str] = []
        self.notes: list[str] = []
        self.producers: dict[str, onnx.NodeProto] = {}  # ONNX value name -> the node computing it

    # ---- driver

    def run(self) -> dict[str, object]:
        for info in list(self.graph.input) + list(self.graph.output) + list(self.graph.value_info):
            kind = self._value_type(info)
            if kind is not None:
                self.types[info.name] = kind
        for initializer in self.graph.initializer:
            array = numpy_helper.to_array(initializer)
            self.constants[initializer.name] = array
            self.types[initializer.name] = _Type(
                tuple(int(d) for d in array.shape), _numpy_dtype_name(array)
            )
        # Initializers that are tensors become parameters; scalars and shape
        # vectors are folded where they are used. Graph inputs that the
        # model's metadata marks as `linnet.path.<input>` are parameters
        # too: `linnet onnx` writes weight-free models that way.
        leaves: dict[str, _Type] = {}
        for initializer in self.graph.initializer:
            array = self.constants[initializer.name]
            if array.ndim >= 1 and not (
                array.dtype == np.int64 and array.ndim == 1 and array.size <= 8
            ):
                leaves[initializer.name] = self.types[initializer.name]
        declared: dict[str, str] = {}  # graph input -> parameter path
        for prop in self.proto.metadata_props:
            if prop.key.startswith("linnet.path."):
                declared[prop.key.removeprefix("linnet.path.")] = prop.value
        for info in self.graph.input:
            if info.name in declared:
                kind = self.types.get(info.name)
                if kind is None:
                    raise OnnxImportError(f"parameter input `{info.name}` has no static shape")
                leaves[declared[info.name]] = kind
        root = self.hierarchy.describe(leaves, self.root_name) if leaves else None
        if root is None:
            raise OnnxImportError("the graph has no tensor initializers to become parameters")
        self.self_value.type = _Type((), "f32")
        root_block = cast(Block, root.block)
        initializer_names = {i.name for i in self.graph.initializer}
        for name in leaves:
            if name in self.constants:
                self.parameters[name] = self.constants[name]
                self.values[name] = self._member_value(root, name, leaves[name])
        for input_name, path in declared.items():
            self.values[input_name] = self._member_value(root, path, leaves[path])

        inputs: list[tuple[str, _Value]] = []
        for info in self.graph.input:
            if info.name in initializer_names or info.name in declared:
                continue
            kind = self.types.get(info.name)
            if kind is None:
                raise OnnxImportError(f"input `{info.name}` has no static shape")
            for dim in kind.shape:
                if isinstance(dim, str):
                    self.symbols.declare(dim)
            value = self.builder.fresh(kind)
            inputs.append((identifier(info.name), value))
            self.values[info.name] = value

        for node in self.graph.node:
            for output in node.output:
                self.producers[output] = node
        for node in self.graph.node:
            self._translate(node)
        refuse_unsupported(self.unsupported, "the graph", OnnxImportError)
        outputs = [self._materialized(o.name) for o in self.graph.output]
        if len(outputs) != 1:
            raise OnnxImportError("the graph must have exactly one output")
        result = outputs[0]
        self.builder.op("return", [result], None)

        generics = [
            {"name": identifier(name), "kind": "dim", "sym": i}
            for name, i in self.symbols.ids.items()
        ]
        root_type = self.hierarchy.block_type(root_block)
        args = [self.builder.value_json(self.self_value, "self", root_type)]
        args += [self.builder.value_json(v, n) for n, v in inputs]
        return self.hierarchy.plan(
            root_block.name,
            args,
            self.builder.regions[0],
            self.builder.type_of(result),
            generics,
        )

    def _value_type(self, info: onnx.ValueInfoProto) -> _Type | None:
        tensor = info.type.tensor_type
        if not tensor.HasField("shape"):
            return None
        shape: list[Dim] = []
        for dim in tensor.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.HasField("dim_param") and dim.dim_param:
                shape.append(dim.dim_param)
            else:
                return None
        return _Type(tuple(shape), _dtype_name(tensor.elem_type))

    def _member_value(self, root: Member, name: str, kind: _Type) -> _Value:
        return self.builder.member_value(
            self.hierarchy, root, self.self_value, name, kind, separators="./"
        )

    # ---- values

    def _materialized(self, name: str) -> _Value:
        """The plan value for an ONNX value, materializing constants."""
        if name in self.values:
            return self.values[name]
        if name in self.constants:
            array = self.constants[name]
            dtype = _numpy_dtype_name(array)
            if array.ndim == 0:
                value = self.builder.const(array.item(), dtype)
            elif np.all(array == array.flat[0]):
                fill = self.builder.const(array.flat[0].item(), dtype)
                value = self.builder.op(
                    "fill",
                    [fill],
                    _Type(tuple(int(d) for d in array.shape), dtype),
                    {"shape": list(array.shape)},
                )
            else:
                raise OnnxImportError(f"constant `{name}` is a non-splat tensor used as a value")
            value.const = array
            self.values[name] = value
            return value
        raise OnnxImportError(f"`{name}` has no value")

    def dims_of(self, name: str) -> list[Dim]:
        """A compile-time shape vector: from shape arithmetic or a constant."""
        if name in self.values and self.values[name].dims is not None:
            return list(cast(list[Dim], self.values[name].dims))
        if name in self.constants:
            array = self.constants[name]
            return [int(x) for x in np.asarray(array).reshape(-1).tolist()]
        raise OnnxImportError(f"`{name}` must be known at import time")

    def type_of(self, name: str) -> _Type:
        if name in self.types:
            return self.types[name]
        if name in self.values:
            return self.values[name].type
        raise OnnxImportError(f"the shape of `{name}` is unknown; shape inference did not reach it")

    # The attributes read without a default, by their ONNX types; any other
    # attribute has its default's type.
    @overload
    @staticmethod
    def attr(node: onnx.NodeProto, name: Literal["axes", "perm"]) -> list[int] | None: ...
    @overload
    @staticmethod
    def attr(node: onnx.NodeProto, name: Literal["value"]) -> onnx.TensorProto | None: ...
    @overload
    @staticmethod
    def attr(node: onnx.NodeProto, name: Literal["end"]) -> int | None: ...
    @overload
    @staticmethod
    def attr(node: onnx.NodeProto, name: Literal["to"]) -> int: ...
    @overload
    @staticmethod
    def attr(node: onnx.NodeProto, name: str, default: _T) -> _T: ...
    @staticmethod
    def attr(node: onnx.NodeProto, name: str, default: object = None) -> object:
        for attribute in node.attribute:
            if attribute.name == name:
                value = onnx.helper.get_attribute_value(attribute)
                return value.decode() if isinstance(value, bytes) else value  # strings are bytes
        return default

    def _translate(self, node: onnx.NodeProto) -> None:
        for name in node.input:
            if name and name not in self.values and name not in self.constants:
                return  # downstream of an unsupported node; reported already
        handler = _HANDLERS.get(node.op_type)
        if handler is None:
            self.unsupported.append(node.op_type)
            return
        try:
            handler(self, node)
        except OnnxImportError as error:
            self.unsupported.append(f"{node.op_type}: {error}")

    def define(self, node: onnx.NodeProto, value: _Value, index: int = 0) -> None:
        self.values[node.output[index]] = value
        self.types[node.output[index]] = value.type

    def result(self, node: onnx.NodeProto, index: int = 0) -> _Type:
        """The output type: ONNX's inference when it is complete, otherwise
        propagated from the operands (inference gives up after a `Reshape`
        whose shape comes from shape arithmetic)."""
        name = node.output[index]
        inferred = self.types.get(name)
        if inferred is not None and not any(
            isinstance(d, str) and d.startswith("unk__") for d in inferred.shape
        ):
            return inferred
        computed = self._propagate(node)
        if computed is None:
            raise OnnxImportError(f"the shape of `{name}` is unknown")
        self.types[name] = computed
        return computed

    def _shape_in(self, name: str) -> tuple[Dim, ...]:
        return self.type_of(name).shape

    def _propagate(self, node: onnx.NodeProto) -> _Type | None:
        kind = node.op_type
        inputs = [n for n in node.input if n]
        first = self.type_of(inputs[0]) if inputs else None
        if kind in ("Transpose",) and first is not None:
            perm = [
                int(a) for a in (self.attr(node, "perm") or list(reversed(range(len(first.shape)))))
            ]
            return _Type(tuple(first.shape[a] for a in perm), first.dtype)
        if kind in ("Reshape",) and first is not None:
            return _Type(
                tuple(_resolve_reshape(first.shape, self.dims_of(node.input[1]))), first.dtype
            )
        if kind in ("Expand",) and first is not None:
            return _Type(tuple(self.dims_of(node.input[1])), first.dtype)
        if kind in _BINARY or kind in _COMPARE or kind == "Where":
            shapes = [self._shape_in(n) for n in inputs]
            if kind == "Where":
                shapes = shapes[1:] + shapes[:1]
            dtype = (
                "bool"
                if kind in _COMPARE
                else self.type_of(inputs[-1 if kind == "Where" else 0]).dtype
            )
            if kind == "Where":
                dtype = self.type_of(inputs[1]).dtype
            return _Type(tuple(_broadcast(shapes)), dtype)
        if kind in _UNARY or (
            kind
            in (
                "Identity",
                "Reciprocal",
                "Pow",
                "Sigmoid",
                "Relu",
                "Gelu",
                "Softmax",
                "LayerNormalization",
            )
            and first is not None
        ):
            return first
        if kind == "Cast" and first is not None:
            return _Type(first.shape, _dtype_name(int(self.attr(node, "to"))))
        if kind == "MatMul" and first is not None:
            a, b = first.shape, self._shape_in(inputs[1])
            if len(b) == 2:
                return _Type((*a[:-1], b[1]), first.dtype)
            return _Type((*a[:-1], b[-1]), first.dtype)
        if kind == "Gather" and first is not None:
            ids = self._shape_in(inputs[1])
            axis = int(self.attr(node, "axis", 0)) % len(first.shape)
            return _Type((*first.shape[:axis], *ids, *first.shape[axis + 1 :]), first.dtype)
        if kind == "Concat" and first is not None:
            axis = int(self.attr(node, "axis", 0)) % len(first.shape)
            sizes = [self._shape_in(n)[axis] for n in inputs]
            if not all(isinstance(d, int) for d in sizes):
                return None
            shape = list(first.shape)
            shape[axis] = sum(cast(int, d) for d in sizes)
            return _Type(tuple(shape), first.dtype)
        if kind == "Unsqueeze" and first is not None:
            axes = [int(a) for a in (self.attr(node, "axes") or self.dims_of(node.input[1]))]
            shape = list(first.shape)
            for a in sorted(a % (len(shape) + 1) for a in axes):
                shape.insert(a, 1)
            return _Type(tuple(shape), first.dtype)
        if kind == "Slice" and first is not None:
            return None  # computed by the handler itself
        return None

    def operand(self, node: onnx.NodeProto, index: int) -> _Value:
        return self._materialized(node.input[index])


Handler = Callable[[_Translator, onnx.NodeProto], None]
_HANDLERS: dict[str, Handler] = {}


def _handles(*names: str) -> Callable[[Handler], Handler]:
    def register(handler: Handler) -> Handler:
        for name in names:
            _HANDLERS[name] = handler
        return handler

    return register


# ---- constants and shapes


@_handles("Constant")
def _constant(t: _Translator, node: onnx.NodeProto) -> None:
    tensor = t.attr(node, "value")
    if tensor is None:
        raise OnnxImportError("only tensor-valued Constant nodes are supported")
    array = numpy_helper.to_array(tensor)
    t.constants[node.output[0]] = array
    t.types[node.output[0]] = _Type(tuple(int(d) for d in array.shape), _numpy_dtype_name(array))


@_handles("Shape")
def _shape(t: _Translator, node: onnx.NodeProto) -> None:
    kind = t.type_of(node.input[0])
    start = int(t.attr(node, "start", 0))
    end = t.attr(node, "end")
    dims = list(kind.shape)[start : None if end is None else int(end)]
    value = _Value(-1, _Type((len(dims),), "i64"), dims=dims)
    t.define(node, value)


def _shape_op(t: _Translator, node: onnx.NodeProto, dims: list[Dim]) -> None:
    t.define(node, _Value(-1, _Type((len(dims),), "i64"), dims=dims))


@_handles("Gather")
def _gather(t: _Translator, node: onnx.NodeProto) -> None:
    data, indices = node.input[0], node.input[1]
    axis = int(t.attr(node, "axis", 0))
    # Shape arithmetic: picking a dimension out of a shape vector.
    if data in t.values and t.values[data].dims is not None:
        dims = t.dims_of(data)
        picked = [dims[int(cast(int, i)) % len(dims)] for i in t.dims_of(indices)]
        index_shape = t.type_of(indices).shape if indices in t.types else (len(picked),)
        _shape_op(t, node, picked if len(index_shape) else picked[:1])
        if len(index_shape) == 0:
            t.values[node.output[0]].type = _Type((), "i64")
        return
    table = t.operand(node, 0)
    kind = t.result(node)
    if indices in t.constants and t.constants[indices].ndim == 0:
        # A constant index selects one slice.
        index = int(t.constants[indices].item())
        axes = [
            {"start": index if i == axis else 0,
             "stop": index + 1 if i == axis else t.symbols.json(d),
             "step": 1, "squeeze": i == axis}
            for i, d in enumerate(table.type.shape)
        ]  # fmt: skip
        if index < 0:
            raise OnnxImportError("negative constant Gather indices are not supported")
        t.define(node, t.builder.op("slice", [table], kind, {"axes": axes}, name=node.output[0]))
        return
    if axis != 0:
        raise OnnxImportError("Gather with tensor indices is only supported along axis 0")
    ids = t.operand(node, 1)
    id_shape = list(ids.type.shape)
    trailing = list(table.type.shape)[1:]

    def body(indices_: list[_Value]) -> _Value:
        row = t.builder.element(ids, indices_[: len(id_shape)])
        if row.type.dtype != "i64":
            row = t.builder.op("cast", [row], _Type((), "i64"))
        return t.builder.element(table, [row, *indices_[len(id_shape) :]])

    t.define(
        node,
        t.builder.comprehension(
            [(f"i{i}", d) for i, d in enumerate(id_shape)]
            + [(f"h{i}", d) for i, d in enumerate(trailing)],
            kind.dtype,
            body,
            name=node.output[0],
        ),
    )


@_handles("Unsqueeze")
def _unsqueeze(t: _Translator, node: onnx.NodeProto) -> None:
    source = node.input[0]
    if source in t.values and t.values[source].dims is not None:
        _shape_op(t, node, t.dims_of(source))
        return
    if source in t.constants and node.output[0] not in t.types:
        axes = t.attr(node, "axes") or t.dims_of(node.input[1])
        array = np.expand_dims(t.constants[source], tuple(int(a) for a in axes))
        t.constants[node.output[0]] = array
        t.types[node.output[0]] = _Type(
            tuple(int(d) for d in array.shape), _numpy_dtype_name(array)
        )
        return
    _reshape_to(t, node, t.operand(node, 0), t.result(node))


@_handles("Squeeze", "Reshape", "Flatten")
def _reshape(t: _Translator, node: onnx.NodeProto) -> None:
    source = node.input[0]
    if source in t.constants and node.output[0] not in t.values and node.op_type == "Reshape":
        shape = t.dims_of(node.input[1])
        if all(isinstance(d, int) for d in shape):
            array = t.constants[source].reshape([int(d) for d in shape])
            t.constants[node.output[0]] = array
            t.types[node.output[0]] = _Type(
                tuple(int(d) for d in array.shape), _numpy_dtype_name(array)
            )
            return
    value = t.operand(node, 0)
    kind = t.result(node) if node.output[0] in t.types else None
    if kind is None and node.op_type == "Reshape":
        kind = _Type(
            tuple(_resolve_reshape(value.type.shape, t.dims_of(node.input[1]))), value.type.dtype
        )
    if kind is None:
        raise OnnxImportError("the result shape is unknown")
    _reshape_to(t, node, value, kind)


def _resolve_reshape(source: Sequence[Dim], target: Sequence[Dim]) -> list[Dim]:
    """ONNX reshape semantics: 0 copies the input dimension, -1 is inferred."""
    out: list[Dim] = []
    for i, d in enumerate(target):
        out.append(source[i] if d == 0 else d)
    if -1 in out:
        known = [d for d in out if d != -1]
        if not all(isinstance(d, int) for d in source) or not all(
            isinstance(d, int) for d in known
        ):
            raise OnnxImportError("inferring a reshape dimension needs static shapes")
        total = int(np.prod([int(d) for d in source])) if source else 1
        rest = int(np.prod([int(d) for d in known])) if known else 1
        out[out.index(-1)] = total // rest
    return out


def _reshape_to(t: _Translator, node: onnx.NodeProto, value: _Value, kind: _Type) -> None:
    if tuple(kind.shape) == tuple(value.type.shape):
        t.define(node, value)
        return
    if value.type.is_scalar:
        # A scalar has no axes to reshape; a tensor of ones of it is a fill.
        t.define(
            node,
            t.builder.op(
                "fill",
                [value],
                kind,
                {"shape": [t.symbols.json(d) for d in kind.shape]},
                name=node.output[0],
            ),
        )
        return
    t.define(
        node,
        t.builder.op(
            "reshape",
            [value],
            kind,
            {"shape": [t.symbols.json(d) for d in kind.shape]},
            name=node.output[0],
        ),
    )


@_handles("Concat")
def _concat(t: _Translator, node: onnx.NodeProto) -> None:
    if all(
        (n in t.values and t.values[n].dims is not None) or n in t.constants for n in node.input
    ) and all(
        (n in t.values and t.values[n].dims is not None) or t.constants[n].dtype == np.int64
        for n in node.input
    ):
        dims: list[Dim] = []
        for n in node.input:
            dims += t.dims_of(n)
        _shape_op(t, node, dims)
        return
    parts = [t.operand(node, i) for i in range(len(node.input))]
    axis = int(t.attr(node, "axis", 0)) % len(parts[0].type.shape)
    t.define(
        node, t.builder.op("concat", parts, t.result(node), {"axis": axis}, name=node.output[0])
    )


@_handles("ConstantOfShape")
def _constant_of_shape(t: _Translator, node: onnx.NodeProto) -> None:
    dims = t.dims_of(node.input[0])
    tensor = t.attr(node, "value")
    array = numpy_helper.to_array(tensor) if tensor is not None else np.zeros((1,), np.float32)
    dtype = _numpy_dtype_name(array)
    fill = t.builder.const(array.flat[0].item(), dtype)
    kind = _Type(tuple(dims), dtype)
    t.define(
        node,
        t.builder.op(
            "fill", [fill], kind, {"shape": [t.symbols.json(d) for d in dims]}, name=node.output[0]
        ),
    )


@_handles("Range")
def _range(t: _Translator, node: onnx.NodeProto) -> None:
    """`Range(0, n, 1)` is `iota`; other ranges shift and scale it."""
    start = t.dims_of(node.input[0])
    limit = t.dims_of(node.input[1])
    delta = t.dims_of(node.input[2])
    if (
        len(start) != 1
        or len(limit) != 1
        or len(delta) != 1
        or not all(isinstance(v[0], int) for v in (start, limit, delta))
    ):
        raise OnnxImportError("Range needs constant bounds")
    first, stop, step = cast(int, start[0]), cast(int, limit[0]), cast(int, delta[0])
    if step == 0:
        raise OnnxImportError("Range with a zero step")
    count = max(0, -(-(stop - first) // step))
    kind = t.result(node) if node.output[0] in t.types else _Type((count,), "i64")
    value = t.builder.op("iota", [], _Type((count,), "i64"), {"shape": [count]})
    if step != 1:
        value = t.builder.op("mul", [value, t.builder.const(step, "i64")], _Type((count,), "i64"))
    if first != 0:
        value = t.builder.op("add", [value, t.builder.const(first, "i64")], _Type((count,), "i64"))
    if kind.dtype != "i64":
        value = t.builder.op("cast", [value], _Type((count,), kind.dtype))
    t.types[node.output[0]] = _Type((count,), kind.dtype)
    t.define(node, value)


@_handles("GatherND")
def _gather_nd(t: _Translator, node: onnx.NodeProto) -> None:
    """`out[g...] = source[indices[g..., 0], indices[g..., 1], ...]`."""
    if int(t.attr(node, "batch_dims", 0)) != 0:
        raise OnnxImportError("GatherND with batch dimensions is not supported")
    source, indices = t.operand(node, 0), t.operand(node, 1)
    index_shape = list(indices.type.shape)
    depth = index_shape[-1]
    if not isinstance(depth, int):
        raise OnnxImportError("GatherND needs a static index depth")
    grid = index_shape[:-1]
    trailing = list(source.type.shape)[depth:]
    kind = _Type(tuple(grid + trailing), source.type.dtype)
    t.types[node.output[0]] = kind

    def body(positions: list[_Value]) -> _Value:
        rows: list[_Value] = []
        for j in range(depth):
            column = t.builder.const(j, "i64")
            row = t.builder.element(indices, [*positions[: len(grid)], column])
            if row.type.dtype != "i64":
                row = t.builder.op("cast", [row], _Type((), "i64"))
            rows.append(row)
        return t.builder.element(source, [*rows, *positions[len(grid) :]])

    t.define(
        node,
        t.builder.comprehension(
            [(f"g{i}", d) for i, d in enumerate(grid)]
            + [(f"h{i}", d) for i, d in enumerate(trailing)],
            kind.dtype,
            body,
            name=node.output[0],
        ),
    )


@_handles("Not")
def _not(t: _Translator, node: onnx.NodeProto) -> None:
    x = t.operand(node, 0)
    kind = t.result(node) if node.output[0] in t.types else x.type
    if kind.is_scalar:
        t.define(node, t.builder.op("not", [x], kind, name=node.output[0]))
        return
    t.define(
        node,
        t.builder.op(
            "select",
            [x, t.builder.const(False, "bool"), t.builder.const(True, "bool")],
            kind,
            name=node.output[0],
        ),
    )


@_handles("Expand")
def _expand(t: _Translator, node: onnx.NodeProto) -> None:
    source = t.operand(node, 0)
    kind = t.result(node)
    if tuple(kind.shape) == tuple(source.type.shape):
        t.define(node, source)
        return
    t.define(
        node,
        t.builder.op(
            "broadcast",
            [source],
            kind,
            {"shape": [t.symbols.json(d) for d in kind.shape]},
            name=node.output[0],
        ),
    )


# ---- elementwise


_BINARY = {"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Max": "max", "Min": "min"}


@_handles(*_BINARY)
def _binary(t: _Translator, node: onnx.NodeProto) -> None:
    # Shape arithmetic stays symbolic.
    if all(
        (n in t.values and t.values[n].dims is not None)
        or (n in t.constants and t.constants[n].dtype == np.int64 and t.constants[n].ndim <= 1)
        for n in node.input
    ) and any(n in t.values and t.values[n].dims is not None for n in node.input):
        a, b = t.dims_of(node.input[0]), t.dims_of(node.input[1])
        if len(a) == 1 and len(b) == 1 and isinstance(a[0], int) and isinstance(b[0], int):
            ops = {"Add": a[0] + b[0], "Sub": a[0] - b[0], "Mul": a[0] * b[0], "Div": a[0] // b[0]}
            _shape_op(t, node, [ops[node.op_type]])
            return
        raise OnnxImportError(
            "symbolic shape arithmetic other than picking dimensions is not supported"
        )
    if all(n in t.constants and n not in t.values for n in node.input):
        arrays = [t.constants[n] for n in node.input]
        folded = {
            "Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide,
            "Max": np.maximum, "Min": np.minimum,
        }[node.op_type](arrays[0], arrays[1])  # fmt: skip
        _fold_constant(t, node, np.asarray(folded, dtype=arrays[0].dtype))
        return
    recovered = _recover(t, node)
    if recovered is not None:
        t.define(node, recovered)
        return
    left, right = t.operand(node, 0), t.operand(node, 1)
    t.define(
        node,
        t.builder.op(_BINARY[node.op_type], [left, right], t.result(node), name=node.output[0]),
    )


_UNARY = {
    "Exp": "exp", "Log": "log", "Sqrt": "sqrt", "Tanh": "tanh", "Sin": "sin", "Cos": "cos",
    "Abs": "abs", "Neg": "neg",
}  # fmt: skip


_NUMPY_UNARY: dict[str, Callable[[_Array], _Array]] = {
    "Exp": np.exp, "Log": np.log, "Sqrt": np.sqrt, "Tanh": np.tanh, "Sin": np.sin,
    "Cos": np.cos, "Abs": np.abs, "Neg": np.negative,
}  # fmt: skip


def _fold_constant(t: _Translator, node: onnx.NodeProto, array: _Array) -> None:
    """A node over constants stays a constant."""
    t.constants[node.output[0]] = array
    t.types[node.output[0]] = _Type(tuple(int(d) for d in array.shape), _numpy_dtype_name(array))


@_handles(*_UNARY)
def _unary(t: _Translator, node: onnx.NodeProto) -> None:
    source = node.input[0]
    if source in t.constants and source not in t.values:
        _fold_constant(t, node, _NUMPY_UNARY[node.op_type](t.constants[source]))
        return
    t.define(
        node,
        t.builder.op(
            _UNARY[node.op_type], [t.operand(node, 0)], t.result(node), name=node.output[0]
        ),
    )


@_handles("Identity")
def _identity(t: _Translator, node: onnx.NodeProto) -> None:
    source = node.input[0]
    if source in t.constants and source not in t.values:
        _fold_constant(t, node, t.constants[source])
        return
    t.define(node, t.operand(node, 0))


@_handles("Reciprocal")
def _reciprocal(t: _Translator, node: onnx.NodeProto) -> None:
    x = t.operand(node, 0)
    t.define(
        node,
        t.builder.op(
            "div", [t.builder.const(1, x.type.dtype), x], t.result(node), name=node.output[0]
        ),
    )


@_handles("Pow")
def _pow(t: _Translator, node: onnx.NodeProto) -> None:
    x = t.operand(node, 0)
    exponent_name = node.input[1]
    if exponent_name not in t.constants or t.constants[exponent_name].size != 1:
        raise OnnxImportError("Pow needs a constant exponent")
    exponent = float(t.constants[exponent_name].reshape(-1)[0])
    kind = t.result(node)
    if exponent == 2:
        t.define(node, t.builder.op("mul", [x, x], kind, name=node.output[0]))
    elif exponent == 0.5:
        t.define(node, t.builder.op("sqrt", [x], kind, name=node.output[0]))
    elif exponent == -0.5:
        t.define(node, t.builder.op("rsqrt", [x], kind, name=node.output[0]))
    elif exponent == 3:
        squared = t.builder.op("mul", [x, x], kind)
        t.define(node, t.builder.op("mul", [squared, x], kind, name=node.output[0]))
    else:
        raise OnnxImportError(f"Pow with exponent {exponent} has no primitive form")


def _std_activation(callee: str) -> Handler:
    def handler(t: _Translator, node: onnx.NodeProto) -> None:
        x = t.operand(node, 0)
        t.define(
            node,
            t.builder.call(
                callee,
                [{"shape": [t.symbols.json(d) for d in x.type.shape]}, {"dtype": x.type.dtype}],
                [x],
                t.result(node),
            ),
        )

    return handler


_handles("Sigmoid")(_std_activation("std.nn.activations::sigmoid"))
_handles("Relu")(_std_activation("std.nn.activations::relu"))


@_handles("Gelu")
def _gelu(t: _Translator, node: onnx.NodeProto) -> None:
    if t.attr(node, "approximate", "none") != "tanh":
        raise OnnxImportError(
            "only the tanh approximation of Gelu is available; erf is not primitive"
        )
    _std_activation("std.nn.activations::gelu")(t, node)


@_handles("Softmax")
def _softmax(t: _Translator, node: onnx.NodeProto) -> None:
    x = t.operand(node, 0)
    rank = len(x.type.shape)
    axis = int(t.attr(node, "axis", -1)) % rank
    if axis != rank - 1:
        raise OnnxImportError("Softmax over an axis other than the last is not supported")
    t.define(
        node,
        t.builder.call(
            "std.nn.softmax::softmax",
            [
                {"shape": [t.symbols.json(d) for d in x.type.shape[:-1]]},
                {"dim": t.symbols.json(x.type.shape[-1])},
                {"dtype": x.type.dtype},
            ],
            [x],
            t.result(node),
        ),
    )


@_handles("LayerNormalization")
def _layer_norm(t: _Translator, node: onnx.NodeProto) -> None:
    x = t.operand(node, 0)
    rank = len(x.type.shape)
    axis = int(t.attr(node, "axis", -1)) % rank
    if axis != rank - 1:
        raise OnnxImportError("LayerNormalization over more than the last axis is not supported")
    weight = t.operand(node, 1)
    width = x.type.shape[-1]
    optional_type: dict[str, object] = {
        "kind": "optional",
        "inner": t.builder.kind_json(_Type((width,), x.type.dtype)),
    }
    if len(node.input) > 2 and node.input[2]:
        bias = t.builder.op(
            "option.some",
            [t.operand(node, 2)],
            _Type((width,), x.type.dtype),
            type_json=optional_type,
        )
    else:
        bias = t.builder.op(
            "option.none", [], _Type((width,), x.type.dtype), type_json=optional_type
        )
    epsilon = t.builder.const(float(t.attr(node, "epsilon", 1e-5)), "f32")
    t.define(
        node,
        t.builder.call(
            "std.nn.norm::layer_norm",
            [
                {"shape": [t.symbols.json(d) for d in x.type.shape[:-1]]},
                {"dim": t.symbols.json(width)},
                {"dtype": x.type.dtype},
            ],
            [x, weight, bias, epsilon],
            t.result(node),
        ),
    )


@_handles("Where")
def _where(t: _Translator, node: onnx.NodeProto) -> None:
    t.define(
        node,
        t.builder.op(
            "select", [t.operand(node, i) for i in range(3)], t.result(node), name=node.output[0]
        ),
    )


_COMPARE = {
    "Equal": "eq",
    "Less": "lt",
    "LessOrEqual": "le",
    "Greater": "gt",
    "GreaterOrEqual": "ge",
}


@_handles(*_COMPARE)
def _compare(t: _Translator, node: onnx.NodeProto) -> None:
    t.define(
        node,
        t.builder.op(
            "compare",
            [t.operand(node, 0), t.operand(node, 1)],
            t.result(node),
            {"compare": _COMPARE[node.op_type]},
            name=node.output[0],
        ),
    )


@_handles("Cast")
def _cast(t: _Translator, node: onnx.NodeProto) -> None:
    target = _dtype_name(int(t.attr(node, "to")))
    # NumPy has no bf16 to fold into; a cast to it stays an operation.
    if node.input[0] in t.constants and node.input[0] not in t.values and target != "bf16":
        _fold_constant(t, node, t.constants[node.input[0]].astype(DTYPES[target].numpy))
        return
    source = t.operand(node, 0)
    kind = t.result(node)
    if kind.dtype == source.type.dtype:
        t.define(node, source)
    else:
        t.define(node, t.builder.op("cast", [source], kind, name=node.output[0]))


@_handles("Transpose")
def _transpose(t: _Translator, node: onnx.NodeProto) -> None:
    source = t.operand(node, 0)
    rank = len(source.type.shape)
    axes = [int(a) for a in (t.attr(node, "perm") or list(reversed(range(rank))))]
    if axes == list(range(rank)):
        t.define(node, source)
        return
    t.define(
        node,
        t.builder.op("permute", [source], t.result(node), {"shape": axes}, name=node.output[0]),
    )


class _SliceAxis(TypedDict):
    """One axis of a `slice` operation."""

    start: int
    stop: _DimJson
    step: int
    squeeze: bool


@_handles("Slice")
def _slice(t: _Translator, node: onnx.NodeProto) -> None:
    source = t.operand(node, 0)
    shape = list(source.type.shape)
    starts = t.dims_of(node.input[1])
    ends = t.dims_of(node.input[2])
    axes = (
        t.dims_of(node.input[3])
        if len(node.input) > 3 and node.input[3]
        else list(range(len(starts)))
    )
    steps = t.dims_of(node.input[4]) if len(node.input) > 4 and node.input[4] else [1] * len(starts)
    per_axis: list[_SliceAxis] = [
        {"start": 0, "stop": t.symbols.json(d), "step": 1, "squeeze": False} for d in shape
    ]
    for start, end, axis, step in zip(starts, ends, axes, steps, strict=True):
        a = int(cast(int, axis)) % len(shape)
        size = shape[a]
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(step, int):
            raise OnnxImportError("Slice bounds must be constants")
        if start < 0 or end < 0:
            if not isinstance(size, int):
                raise OnnxImportError("negative Slice bounds need a static axis")
            start = start + size if start < 0 else start
            end = end + size if end < 0 else end
        if end >= 2**31:
            end_dim: Dim = size
        else:
            end_dim = end if not isinstance(size, int) else min(end, size)
        per_axis[a] = {
            "start": start,
            "stop": t.symbols.json(end_dim),
            "step": step,
            "squeeze": False,
        }
    inferred = t.types.get(node.output[0])
    if inferred is None or any(
        isinstance(d, str) and d.startswith("unk__") for d in inferred.shape
    ):
        sliced: list[Dim] = []
        for i, entry in enumerate(per_axis):
            stop = entry["stop"]
            if isinstance(stop, dict):
                if entry["start"] != 0 or entry["step"] != 1:
                    raise OnnxImportError("slicing a symbolic axis is only supported whole")
                sliced.append(shape[i])
            else:
                sliced.append(
                    (int(stop) - int(entry["start"]) + int(entry["step"]) - 1) // int(entry["step"])
                )
        t.types[node.output[0]] = _Type(tuple(sliced), source.type.dtype)
    t.define(
        node,
        t.builder.op("slice", [source], t.result(node), {"axes": per_axis}, name=node.output[0]),
    )


# ---- exact pattern recovery
#
# Exporters spell library operations out: PyTorch writes an RMS norm as
# `Mul(Mul(x, Reciprocal(Sqrt(Add(ReduceMean(Pow(x, 2)), eps)))), w)`, SiLU
# as `Mul(x, Sigmoid(x))`, and a softmax written by hand as
# `Div(Exp(Sub(x, ReduceMax(x))), ReduceSum(Exp(...)))`. When a node
# completes one of those shapes over the same operand, the import emits the
# standard-library operation instead; the nodes it replaces become dead and
# are dropped, and every recovery is recorded in the notes. The check is
# structural — operation types, shared operands, the axis, the constants —
# so nothing is recovered by name or by approximation.
#
# A pattern is a variable name (binds the ONNX value on first sight, must be
# the same value afterwards), a `_Const` (a splat constant with that value,
# or any splat constant bound under `name`), `("reduce", op_type, operand)`
# for a keepdims reduction over the last axis, `("any", *patterns)` for
# alternatives, or `("<OpType>", *operand patterns)`; commutative operations
# match in either operand order.


@dataclass(frozen=True)
class _Const:
    value: float | None = None
    name: str | None = None


_Pattern: TypeAlias = "str | _Const | tuple[str, *tuple[_Pattern, ...]]"
_COMMUTATIVE = {"Add", "Mul", "Max", "Min"}


class _Matcher:
    def __init__(self, t: _Translator) -> None:
        self.t = t
        self.bound: dict[str, str] = {}

    def splat(self, name: str) -> float | None:
        array = self.t.constants.get(self.bound[name])
        if array is None or array.size == 0 or not np.all(array == array.flat[0]):
            return None
        return float(array.flat[0])

    def match(self, name: str, pattern: _Pattern) -> bool:
        if isinstance(pattern, str):
            if pattern in self.bound:
                return self.bound[pattern] == name
            self.bound[pattern] = name
            return True
        if isinstance(pattern, _Const):
            if name not in self.t.constants or name in self.t.parameters:
                return False
            if pattern.name is not None:
                self.bound[pattern.name] = name
            splat = self.splat(pattern.name) if pattern.name is not None else None
            if pattern.value is not None:
                array = self.t.constants[name]
                if not np.all(array == array.flat[0]):
                    return False
                splat = float(array.flat[0])
                return math.isclose(splat, pattern.value, rel_tol=1e-6, abs_tol=1e-12)
            return splat is not None
        kind, *operands = pattern
        if kind == "any":
            for alternative in operands:
                saved = dict(self.bound)
                if self.match(name, alternative):
                    return True
                self.bound = saved
            return False
        node = self.t.producers.get(name)
        if node is None:
            return False
        if kind == "reduce":
            return (
                node.op_type == operands[0]
                and self._last_axis(node)
                and self.match(node.input[0], operands[1])
            )
        if node.op_type != kind:
            return False
        inputs = [n for n in node.input if n]
        if len(inputs) != len(operands):
            return False
        if kind in _COMMUTATIVE and len(operands) == 2:
            saved = dict(self.bound)
            if self.match(inputs[0], operands[0]) and self.match(inputs[1], operands[1]):
                return True
            self.bound = saved
            return self.match(inputs[0], operands[1]) and self.match(inputs[1], operands[0])
        return all(self.match(n, p) for n, p in zip(inputs, operands, strict=True))

    def _last_axis(self, node: onnx.NodeProto) -> bool:
        """A keepdims reduction over exactly the last axis of its operand."""
        try:
            rank = len(self.t.type_of(node.input[0]).shape)
            axes = self.t.attr(node, "axes")
            if axes is None:
                if len(node.input) < 2 or not node.input[1]:
                    return False
                axes = self.t.dims_of(node.input[1])
        except OnnxImportError:
            return False
        normalized = sorted(int(a) % rank for a in axes)
        return normalized == [rank - 1] and int(self.t.attr(node, "keepdims", 1)) == 1


_SQUARE: _Pattern = ("any", ("Mul", "x", "x"), ("Pow", "x", _Const(2.0)))
_VARIANCE: _Pattern = ("Add", ("reduce", "ReduceMean", _SQUARE), _Const(name="eps"))
_RSQRT: _Pattern = (
    "any",
    ("Reciprocal", ("Sqrt", _VARIANCE)),
    ("Div", _Const(1.0), ("Sqrt", _VARIANCE)),
    ("Pow", _VARIANCE, _Const(-0.5)),
)
_RMS_NORM: _Pattern = (
    "Mul",
    ("any", ("Mul", "x", _RSQRT), ("Div", "x", ("Sqrt", _VARIANCE))),
    "w",
)
_SILU: _Pattern = ("Mul", "x", ("Sigmoid", "x"))
_SOFTMAX: _Pattern = ("Div", "e", ("reduce", "ReduceSum", "e"))
_SOFTMAX_NUMERATOR: _Pattern = ("Exp", ("Sub", "x", ("reduce", "ReduceMax", "x")))


def _recover(t: _Translator, node: onnx.NodeProto) -> _Value | None:
    """The library operation whose decomposition `node` completes, if any."""
    result = node.output[0]
    kind = t.result(node)
    dims = [t.symbols.json(d) for d in kind.shape]

    def recovered(
        name: str, generics: Sequence[Mapping[str, object]], operands: list[_Value]
    ) -> _Value:
        t.notes.append(f"recovered {name} from its decomposition")
        module = {"softmax": "std.nn.softmax", "rms_norm": "std.nn.norm"}.get(
            name, "std.nn.activations"
        )
        return t.builder.call(f"{module}::{name}", generics, operands, kind)

    def source(m: _Matcher, name: str) -> _Value | None:
        value = t.values.get(m.bound[name])
        return value if value is not None and value.type.shape == kind.shape else None

    if node.op_type == "Mul":
        m = _Matcher(t)
        if m.match(result, _SILU):
            x = source(m, "x")
            if x is not None:
                return recovered("silu", [{"shape": dims}, {"dtype": kind.dtype}], [x])
        m = _Matcher(t)
        if m.match(result, _RMS_NORM) and kind.shape:
            x, eps = source(m, "x"), m.splat("eps")
            w = t.values.get(m.bound["w"])
            if (
                x is not None
                and eps is not None
                and w is not None
                and w.type.shape == kind.shape[-1:]
            ):
                generics = [{"shape": dims[:-1]}, {"dim": dims[-1]}, {"dtype": kind.dtype}]
                epsilon = t.builder.const(eps, "f32")
                return recovered("rms_norm", generics, [x, w, epsilon])
    if node.op_type == "Div":
        m = _Matcher(t)
        if m.match(result, _SOFTMAX) and m.match(m.bound["e"], _SOFTMAX_NUMERATOR) and kind.shape:
            x = source(m, "x")
            if x is not None:
                generics = [{"shape": dims[:-1]}, {"dim": dims[-1]}, {"dtype": kind.dtype}]
                return recovered("softmax", generics, [x])
    return None


# ---- contractions and reductions


@_handles("MatMul")
def _matmul(t: _Translator, node: onnx.NodeProto) -> None:
    a, b = t.operand(node, 0), t.operand(node, 1)
    kind = t.result(node)
    ra, rb = len(a.type.shape), len(b.type.shape)
    if ra == 2 and rb == 2:
        m, k = a.type.shape
        n = b.type.shape[1]
        t.define(
            node,
            t.builder.call(
                "std.linalg::matmul",
                [
                    {"dim": t.symbols.json(m)},
                    {"dim": t.symbols.json(k)},
                    {"dim": t.symbols.json(n)},
                    {"dtype": a.type.dtype},
                ],
                [a, b],
                kind,
            ),
        )
        return
    if ra == rb and ra > 2 and tuple(a.type.shape[:-2]) == tuple(b.type.shape[:-2]):
        batch = list(a.type.shape[:-2])
        m, k, n = a.type.shape[-2], a.type.shape[-1], b.type.shape[-1]
        t.define(
            node,
            t.builder.call(
                "std.linalg::batched_matmul",
                [
                    {"shape": [t.symbols.json(d) for d in batch]},
                    {"dim": t.symbols.json(m)},
                    {"dim": t.symbols.json(k)},
                    {"dim": t.symbols.json(n)},
                    {"dtype": a.type.dtype},
                ],
                [a, b],
                kind,
            ),
        )
        return
    if rb == 2 and ra > 2:
        # `[..., K] x [K, N]`: a contraction over the last axis.
        k, n = b.type.shape
        outputs = [(f"o{i}", d) for i, d in enumerate(a.type.shape[:-1])] + [("n", n)]

        def body(out: list[_Value]) -> _Value:
            def inner(kk: list[_Value]) -> _Value:
                return t.builder.op(
                    "mul",
                    [
                        t.builder.element(a, [*out[:-1], kk[0]]),
                        t.builder.element(b, [kk[0], out[-1]]),
                    ],
                    _Type((), kind.dtype),
                )

            return t.builder.reduce("sum", [("k", k)], kind.dtype, inner)

        t.define(node, t.builder.comprehension(outputs, kind.dtype, body, name=node.output[0]))
        return
    raise OnnxImportError("MatMul with broadcast batch dimensions is not supported")


@_handles("Gemm")
def _gemm(t: _Translator, node: onnx.NodeProto) -> None:
    if float(t.attr(node, "alpha", 1.0)) != 1.0 or float(t.attr(node, "beta", 1.0)) != 1.0:
        raise OnnxImportError("Gemm with alpha or beta other than 1 is not supported")
    a, b = t.operand(node, 0), t.operand(node, 1)
    if int(t.attr(node, "transA", 0)):
        a = t.builder.op(
            "permute",
            [a],
            _Type((a.type.shape[1], a.type.shape[0]), a.type.dtype),
            {"shape": [1, 0]},
        )
    if int(t.attr(node, "transB", 0)):
        b = t.builder.op(
            "permute",
            [b],
            _Type((b.type.shape[1], b.type.shape[0]), b.type.dtype),
            {"shape": [1, 0]},
        )
    m, k = a.type.shape
    n = b.type.shape[1]
    kind = t.result(node)
    product = t.builder.call(
        "std.linalg::matmul",
        [
            {"dim": t.symbols.json(m)},
            {"dim": t.symbols.json(k)},
            {"dim": t.symbols.json(n)},
            {"dtype": a.type.dtype},
        ],
        [a, b],
        _Type((m, n), a.type.dtype),
    )
    if len(node.input) > 2 and node.input[2]:
        t.define(
            node, t.builder.op("add", [product, t.operand(node, 2)], kind, name=node.output[0])
        )
    else:
        t.define(node, product)


def _reduction(kind: str, divide: bool) -> Handler:
    def handler(t: _Translator, node: onnx.NodeProto) -> None:
        x = t.operand(node, 0)
        shape = list(x.type.shape)
        rank = len(shape)
        axes_attr = t.attr(node, "axes")
        if axes_attr is None and len(node.input) > 1 and node.input[1]:
            axes_attr = t.dims_of(node.input[1])
        reduced = sorted(int(a) % rank for a in axes_attr) if axes_attr else list(range(rank))
        keepdims = int(t.attr(node, "keepdims", 1)) == 1
        result = t.result(node)
        divisor: int | None = None
        if divide:
            divisor = 1
            for a in reduced:
                size = shape[a]
                if not isinstance(size, int):
                    raise OnnxImportError("mean over a symbolic axis is not supported")
                divisor *= size
        value = t.builder.reduce_axes(
            x, shape, reduced, kind, result.dtype, divisor, "" if keepdims else node.output[0]
        )
        if len(reduced) == rank:
            if result.is_scalar:
                t.define(node, value)
            else:
                t.define(
                    node,
                    t.builder.op(
                        "fill",
                        [value],
                        result,
                        {"shape": [t.symbols.json(d) for d in result.shape]},
                        name=node.output[0],
                    ),
                )
        elif keepdims:
            _reshape_to(t, node, value, result)
        else:
            t.define(node, value)

    return handler


_handles("ReduceSum")(_reduction("sum", divide=False))
_handles("ReduceMean")(_reduction("sum", divide=True))
_handles("ReduceMax")(_reduction("max", divide=False))
_handles("ReduceMin")(_reduction("min", divide=False))
