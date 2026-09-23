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

import json
import math
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx  # type: ignore[import-untyped]
from onnx import numpy_helper, shape_inference  # type: ignore[import-untyped]

from .compiler import LinnetError, find_compiler


class OnnxImportError(LinnetError):
    """The graph cannot be expressed in Linnet as captured."""


_RESERVED = {
    "as", "const", "type", "struct", "enum", "fn", "op", "block", "entry", "param", "buffer",
    "state", "sub", "let", "var", "return", "if", "else", "match", "static", "for", "in", "while",
    "where", "true", "false", "none", "some", "extern", "module", "use", "pub", "Tensor", "Dim",
    "Shape", "DType", "Numeric", "Integer", "Float", "cast", "reshape", "permute", "broadcast_to",
    "concat", "pad", "iota", "fill", "gather", "scatter", "exp", "log", "sqrt", "rsqrt", "sin",
    "cos", "tanh", "abs", "select", "min", "max", "sum", "prod", "any", "all", "bool", "i8", "i16",
    "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16", "f32", "f64",
}  # fmt: skip


def _identifier(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not clean or clean[0].isdigit():
        clean = "_" + clean
    if clean in _RESERVED:
        clean += "_"
    return clean


_DTYPES: dict[int, str] = {
    onnx.TensorProto.BOOL: "bool",
    onnx.TensorProto.INT8: "i8",
    onnx.TensorProto.INT16: "i16",
    onnx.TensorProto.INT32: "i32",
    onnx.TensorProto.INT64: "i64",
    onnx.TensorProto.UINT8: "u8",
    onnx.TensorProto.UINT16: "u16",
    onnx.TensorProto.UINT32: "u32",
    onnx.TensorProto.UINT64: "u64",
    onnx.TensorProto.FLOAT16: "f16",
    onnx.TensorProto.BFLOAT16: "bf16",
    onnx.TensorProto.FLOAT: "f32",
    onnx.TensorProto.DOUBLE: "f64",
}

_NUMPY_DTYPES: dict[str, Any] = {
    "bool": np.bool_, "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64,
    "u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64, "f16": np.float16,
    "f32": np.float32, "f64": np.float64,
}  # fmt: skip


def _dtype_name(elem_type: int) -> str:
    if elem_type not in _DTYPES:
        raise OnnxImportError(f"ONNX element type {elem_type} has no Linnet equivalent")
    return _DTYPES[elem_type]


def _numpy_dtype_name(array: Any) -> str:
    for name, dtype in _NUMPY_DTYPES.items():
        if np.dtype(dtype) == array.dtype:
            return name
    raise OnnxImportError(f"dtype {array.dtype} has no Linnet equivalent")


# A dimension: an int, or a symbol name from `dim_param`.
Dim = int | str


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
    const: Any = None


@dataclass
class _Block:
    name: str
    members: list[dict[str, Any]] = field(default_factory=lambda: [])


@dataclass
class _Member:
    kind: str  # "param" | "sub"
    name: str
    block: _Block | None = None
    length: int | None = None
    element: _Block | None = None
    children: dict[str, _Member] = field(default_factory=lambda: {})
    leaf: _Type | None = None


class _Symbols:
    """Named symbolic dimensions of the graph, numbered for the plan."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    def declare(self, name: str) -> int:
        if name not in self.ids:
            self.ids[name] = len(self.ids)
        return self.ids[name]

    def json(self, dim: Dim) -> Any:
        if isinstance(dim, int):
            return dim
        if dim not in self.ids:
            raise OnnxImportError(f"dimension `{dim}` does not come from a graph input")
        return {"sym": self.ids[dim], "name": _identifier(dim)}


class _Builder:
    def __init__(self, symbols: _Symbols) -> None:
        self.symbols = symbols
        self.next_id = 0
        self.regions: list[list[dict[str, Any]]] = [[]]

    def type_json(self, kind: _Type) -> dict[str, Any]:
        if kind.is_scalar:
            return {"kind": "scalar", "dtype": kind.dtype}
        return {
            "kind": "tensor",
            "shape": [self.symbols.json(d) for d in kind.shape],
            "dtype": kind.dtype,
        }

    def fresh(self, kind: _Type) -> _Value:
        self.next_id += 1
        return _Value(self.next_id - 1, kind)

    def value_json(
        self, value: _Value, name: str = "", type_json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {"id": value.id, "name": name, "type": type_json or self.type_json(value.type)}

    def op(
        self,
        kind: str,
        operands: Sequence[_Value],
        result: _Type | None,
        attrs: dict[str, Any] | None = None,
        regions: Sequence[dict[str, Any]] = (),
        name: str = "",
        type_json: dict[str, Any] | None = None,
    ) -> _Value:
        results = [self.fresh(result)] if result is not None else []
        self.regions[-1].append(
            {
                "kind": kind,
                "operands": [v.id for v in operands],
                "results": [self.value_json(v, name, type_json) for v in results],
                "attrs": attrs or {},
                "regions": list(regions),
            }
        )
        return results[0] if results else _Value(-1, _Type((), "f32"))

    def const(self, value: float | int | bool, dtype: str) -> _Value:
        kind = _Type((), dtype)
        if dtype == "bool":
            return self.op("const.bool", [], kind, {"value": 1 if value else 0})
        if dtype.startswith("f") or dtype == "bf16":
            return self.op("const.float", [], kind, {"value": float(value)})
        return self.op("const.int", [], kind, {"value": int(value)})

    def const_dim(self, dim: Dim) -> _Value:
        return self.op("const.dim", [], _Type((), "i64"), {"value": self.symbols.json(dim)})

    def region(
        self, arguments: Sequence[tuple[str, _Type]], body: Callable[[list[_Value]], _Value]
    ) -> dict[str, Any]:
        values = [self.fresh(kind) for _, kind in arguments]
        self.regions.append([])
        yielded = body(values)
        self.op("yield", [yielded], None)
        ops = self.regions.pop()
        return {
            "args": [self.value_json(v, n) for v, (n, _) in zip(values, arguments, strict=True)],
            "ops": ops,
        }

    def comprehension(
        self,
        indices: Sequence[tuple[str, Dim]],
        dtype: str,
        body: Callable[[list[_Value]], _Value],
        name: str = "",
    ) -> _Value:
        region = self.region([(n, _Type((), "i64")) for n, _ in indices], body)
        return self.op(
            "comprehension",
            [],
            _Type(tuple(d for _, d in indices), dtype),
            {"indices": [{"name": n, "domain": [self.symbols.json(d)]} for n, d in indices]},
            [region],
            name,
        )

    def reduce(
        self,
        kind: str,
        indices: Sequence[tuple[str, Dim]],
        dtype: str,
        body: Callable[[list[_Value]], _Value],
    ) -> _Value:
        region = self.region([(n, _Type((), "i64")) for n, _ in indices], body)
        return self.op(
            "reduce",
            [],
            _Type((), dtype),
            {
                "indices": [{"name": n, "domain": [self.symbols.json(d)]} for n, d in indices],
                "reduce": kind,
            },
            [region],
        )

    def element(self, tensor: _Value, indices: Sequence[_Value]) -> _Value:
        return self.op("tensor.element", [tensor, *indices], _Type((), tensor.type.dtype))

    def call(
        self, callee: str, generics: list[dict[str, Any]], operands: list[_Value], kind: _Type
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


class _Hierarchy:
    """Blocks from the dotted names of the initializers."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.blocks: dict[str, _Block] = {}
        self._by_signature: dict[Any, _Block] = {}
        self.renamed: dict[str, str] = {}

    def describe(self, leaves: dict[str, _Type], root_name: str) -> _Member:
        tree: dict[str, Any] = {}
        for name, kind in leaves.items():
            node = tree
            parts = re.split(r"[./]", name)
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = kind
        signature, member = self._describe(tree, root_name)
        if member is None:
            raise OnnxImportError("the graph has no initializers to become parameters")
        member.block = self._block_for(root_name, signature, member)
        return member

    def _describe(self, node: Any, name_hint: str) -> tuple[Any, _Member | None]:
        if isinstance(node, _Type):
            return ("param", node.dtype, node.shape), _Member("param", "", leaf=node)
        tree = cast(dict[str, Any], node)
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
                block = self._block_for(element_name, signature, cast(_Member, described[0][1]))
                children = {str(i): cast(_Member, m) for i, (_, m) in enumerate(described)}
                for child in children.values():
                    child.block = block
                return ("array", len(keys), signature), _Member(
                    "sub", "", length=len(keys), element=block, children=children
                )
        entries: list[Any] = []
        children: dict[str, _Member] = {}
        for key in keys:
            signature, child = self._describe(tree[key], key)
            if child is None:
                continue
            child.name = _identifier(key)
            if child.kind == "sub" and child.length is None:
                # Numbered children of a mixed container get a block name of
                # their own; a member and its block must not share a name.
                class_name = f"Item{key}" if key.isdigit() else key.capitalize()
                child.block = self._block_for(class_name, signature, child)
            children[key] = child
            entries.append((key, signature))
        if not entries:
            return (), None
        return tuple(entries), _Member("sub", "", children=children)

    def _block_for(self, class_name: str, signature: Any, member: _Member) -> _Block:
        key = (class_name, signature)
        if key in self._by_signature:
            return self._by_signature[key]
        base = _identifier(class_name)
        name = base
        for suffix in range(2, 1000):
            if name not in self.blocks:
                break
            name = f"{base}_{suffix}"
        block = _Block(name)
        self.blocks[name] = block
        self._by_signature[key] = block
        for child in member.children.values():
            if child.kind == "param":
                block.members.append(
                    {
                        "name": child.name,
                        "kind": "param",
                        "type": {
                            "kind": "tensor",
                            "shape": list(cast(_Type, child.leaf).shape),
                            "dtype": cast(_Type, child.leaf).dtype,
                        },
                    }
                )
            else:
                block.members.append(
                    {"name": child.name, "kind": "sub", "type": self.member_type(child)}
                )
        return block

    def block_type(self, block: _Block) -> dict[str, Any]:
        return {"kind": "block", "name": block.name, "module": self.module, "args": []}

    def member_type(self, member: _Member) -> dict[str, Any]:
        if member.length is not None:
            return {
                "kind": "array",
                "element": self.block_type(cast(_Block, member.element)),
                "length": member.length,
            }
        return self.block_type(cast(_Block, member.block))


@dataclass
class ImportResult:
    source: Path
    module: str
    root: str
    weights: Path | None
    bindings: Path | None
    plan: dict[str, Any]
    notes: list[str]


def import_onnx(
    model: str | Path | Any,
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
    module = module_name or _identifier(output_path.stem)
    proto: Any = onnx.load(str(model)) if isinstance(model, str | Path) else model
    proto = shape_inference.infer_shapes(proto, strict_mode=False)
    translator = _Translator(proto, module, root_name)
    plan = translator.run()

    compiler = find_compiler()
    emitted = subprocess.run(
        [compiler, "emit", "-"], input=json.dumps(plan), capture_output=True, text=True, check=False
    )
    if emitted.returncode != 0:
        raise OnnxImportError("the compiler rejected the imported plan:\n" + emitted.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(emitted.stdout, encoding="utf-8")
    check = [compiler, "check", str(output_path)]
    if std_root is not None:
        check[2:2] = ["--std", str(std_root)]
    checked = subprocess.run(check, capture_output=True, text=True, check=False)
    if checked.returncode != 0:
        raise OnnxImportError(
            "the imported source does not check:\n" + checked.stderr + checked.stdout
        )

    weights_path: Path | None = None
    bindings_path: Path | None = None
    if weights is not None:
        from safetensors.numpy import save_file  # type: ignore[import-untyped]

        weights_dir = Path(weights)
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "model.safetensors"
        save_file(
            {name: np.ascontiguousarray(array) for name, array in translator.parameters.items()},
            str(weights_path),
        )
        if translator.hierarchy.renamed:
            bindings_path = weights_dir / "bindings.json"
            bindings_path.write_text(
                json.dumps(translator.hierarchy.renamed, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
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
    def __init__(self, proto: Any, module: str, root_name: str) -> None:
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
        self.parameters: dict[str, Any] = {}  # ONNX name -> array
        self.constants: dict[str, Any] = {}  # values known at import time
        self.unsupported: list[str] = []
        self.notes: list[str] = []
        self.producers: dict[str, Any] = {}  # ONNX value name -> the node computing it

    # ---- driver

    def run(self) -> dict[str, Any]:
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
        root_block = cast(_Block, root.block)
        initializer_names = {i.name for i in self.graph.initializer}
        for name in leaves:
            if name in self.constants:
                self.parameters[name] = self.constants[name]
                self.values[name] = self._member_value(root, name)
        for input_name, path in declared.items():
            self.values[input_name] = self._member_value(root, path)

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
            inputs.append((_identifier(info.name), value))
            self.values[info.name] = value

        for node in self.graph.node:
            for output in node.output:
                self.producers[output] = node
        for node in self.graph.node:
            self._translate(node)
        if self.unsupported:
            raise OnnxImportError(
                "the graph uses operations without a Linnet mapping:\n  "
                + "\n  ".join(sorted(set(self.unsupported)))
            )
        outputs = [self._materialized(o.name) for o in self.graph.output]
        if len(outputs) != 1:
            raise OnnxImportError("the graph must have exactly one output")
        result = outputs[0]
        self.builder.op("return", [result], None)

        generics = [
            {"name": _identifier(name), "kind": "dim", "sym": i}
            for name, i in self.symbols.ids.items()
        ]
        blocks = {
            name: {
                "module": self.module,
                "pub": True,
                "generics": [],
                "constraints": [],
                "members": block.members,
            }
            for name, block in self.hierarchy.blocks.items()
        }
        body = {
            "args": [
                self.builder.value_json(
                    self.self_value, "self", self.hierarchy.block_type(root_block)
                )
            ]
            + [self.builder.value_json(v, n) for n, v in inputs],
            "ops": self.builder.regions[0],
        }
        return {
            "version": 1,
            "module": self.module,
            "root": {"name": root_block.name, "generics": [], "constraints": []},
            "manifest": [],
            "blocks": blocks,
            "functions": [
                {
                    "name": f"{self.module}::{root_block.name}.forward",
                    "kind": "entry",
                    "block": root_block.name,
                    "pub": True,
                    "generics": generics,
                    "constraints": [],
                    "results": [self.builder.type_json(result.type)],
                    "body": body,
                }
            ],
            "constants": [],
        }

    def _value_type(self, info: Any) -> _Type | None:
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

    def _member_value(self, root: _Member, name: str) -> _Value:
        member = root
        current = self.self_value
        parts = re.split(r"[./]", name)
        linnet_parts: list[str] = []
        for i, part in enumerate(parts):
            child = member.children[part]
            is_last = i == len(parts) - 1
            if member.length is not None:
                index = self.builder.const(int(part), "i64")
                current = self.builder.op(
                    "array.get", [current, index], _Type((), "f32"),
                    type_json=self.hierarchy.block_type(cast(_Block, member.element)),
                )  # fmt: skip
                linnet_parts.append(part)
                member = child
                continue
            linnet_parts.append(child.name)
            if is_last:
                linnet_path = ".".join(linnet_parts)
                if linnet_path != name:
                    self.hierarchy.renamed[linnet_path] = name
                return self.builder.op(
                    "block.param",
                    [current],
                    cast(_Type, child.leaf),
                    {"name": child.name},
                    name=child.name,
                )
            current = self.builder.op(
                "block.sub", [current], _Type((), "f32"), {"name": child.name},
                type_json=self.hierarchy.member_type(child),
            )  # fmt: skip
            member = child
        raise OnnxImportError(f"cannot resolve `{name}`")

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

    def _int(self, name: str) -> int:
        dims = self.dims_of(name)
        if len(dims) != 1 or not isinstance(dims[0], int):
            raise OnnxImportError(f"`{name}` must be one integer")
        return dims[0]

    def type_of(self, name: str) -> _Type:
        if name in self.types:
            return self.types[name]
        if name in self.values:
            return self.values[name].type
        raise OnnxImportError(f"the shape of `{name}` is unknown; shape inference did not reach it")

    @staticmethod
    def attr(node: Any, name: str, default: Any = None) -> Any:
        for attribute in node.attribute:
            if attribute.name == name:
                value = onnx.helper.get_attribute_value(attribute)
                return value.decode() if isinstance(value, bytes) else value  # strings are bytes
        return default

    def _translate(self, node: Any) -> None:
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

    def define(self, node: Any, value: _Value, index: int = 0) -> None:
        self.values[node.output[index]] = value
        self.types[node.output[index]] = value.type

    def result(self, node: Any, index: int = 0) -> _Type:
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

    def _propagate(self, node: Any) -> _Type | None:
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

    def operand(self, node: Any, index: int) -> _Value:
        return self._materialized(node.input[index])


Handler = Callable[[_Translator, Any], None]
_HANDLERS: dict[str, Handler] = {}


def _handles(*names: str) -> Callable[[Handler], Handler]:
    def register(handler: Handler) -> Handler:
        for name in names:
            _HANDLERS[name] = handler
        return handler

    return register


# ---- constants and shapes


@_handles("Constant")
def _constant(t: _Translator, node: Any) -> None:
    tensor = t.attr(node, "value")
    if tensor is None:
        raise OnnxImportError("only tensor-valued Constant nodes are supported")
    array = numpy_helper.to_array(tensor)
    t.constants[node.output[0]] = array
    t.types[node.output[0]] = _Type(tuple(int(d) for d in array.shape), _numpy_dtype_name(array))


@_handles("Shape")
def _shape(t: _Translator, node: Any) -> None:
    kind = t.type_of(node.input[0])
    start = int(t.attr(node, "start", 0))
    end = t.attr(node, "end")
    dims = list(kind.shape)[start : None if end is None else int(end)]
    value = _Value(-1, _Type((len(dims),), "i64"), dims=dims)
    t.define(node, value)


def _shape_op(t: _Translator, node: Any, dims: list[Dim]) -> None:
    t.define(node, _Value(-1, _Type((len(dims),), "i64"), dims=dims))


@_handles("Gather")
def _gather(t: _Translator, node: Any) -> None:
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
def _unsqueeze(t: _Translator, node: Any) -> None:
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
def _reshape(t: _Translator, node: Any) -> None:
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


def _reshape_to(t: _Translator, node: Any, value: _Value, kind: _Type) -> None:
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
def _concat(t: _Translator, node: Any) -> None:
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
def _constant_of_shape(t: _Translator, node: Any) -> None:
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
def _range(t: _Translator, node: Any) -> None:
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
def _gather_nd(t: _Translator, node: Any) -> None:
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
def _not(t: _Translator, node: Any) -> None:
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
def _expand(t: _Translator, node: Any) -> None:
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
def _binary(t: _Translator, node: Any) -> None:
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


_NUMPY_UNARY: dict[str, Callable[[Any], Any]] = {
    "Exp": np.exp, "Log": np.log, "Sqrt": np.sqrt, "Tanh": np.tanh, "Sin": np.sin,
    "Cos": np.cos, "Abs": np.abs, "Neg": np.negative,
}  # fmt: skip


def _fold_constant(t: _Translator, node: Any, array: Any) -> None:
    """A node over constants stays a constant."""
    t.constants[node.output[0]] = array
    t.types[node.output[0]] = _Type(tuple(int(d) for d in array.shape), _numpy_dtype_name(array))


@_handles(*_UNARY)
def _unary(t: _Translator, node: Any) -> None:
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
def _identity(t: _Translator, node: Any) -> None:
    source = node.input[0]
    if source in t.constants and source not in t.values:
        _fold_constant(t, node, t.constants[source])
        return
    t.define(node, t.operand(node, 0))


@_handles("Reciprocal")
def _reciprocal(t: _Translator, node: Any) -> None:
    x = t.operand(node, 0)
    t.define(
        node,
        t.builder.op(
            "div", [t.builder.const(1, x.type.dtype), x], t.result(node), name=node.output[0]
        ),
    )


@_handles("Pow")
def _pow(t: _Translator, node: Any) -> None:
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
    def handler(t: _Translator, node: Any) -> None:
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
def _gelu(t: _Translator, node: Any) -> None:
    if t.attr(node, "approximate", "none") != "tanh":
        raise OnnxImportError(
            "only the tanh approximation of Gelu is available; erf is not primitive"
        )
    _std_activation("std.nn.activations::gelu")(t, node)


@_handles("Softmax")
def _softmax(t: _Translator, node: Any) -> None:
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
def _layer_norm(t: _Translator, node: Any) -> None:
    x = t.operand(node, 0)
    rank = len(x.type.shape)
    axis = int(t.attr(node, "axis", -1)) % rank
    if axis != rank - 1:
        raise OnnxImportError("LayerNormalization over more than the last axis is not supported")
    weight = t.operand(node, 1)
    width = x.type.shape[-1]
    optional_type = {
        "kind": "optional",
        "inner": t.builder.type_json(_Type((width,), x.type.dtype)),
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
def _where(t: _Translator, node: Any) -> None:
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
def _compare(t: _Translator, node: Any) -> None:
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
def _cast(t: _Translator, node: Any) -> None:
    if node.input[0] in t.constants and node.input[0] not in t.values:
        target = _NUMPY_DTYPES[_dtype_name(int(t.attr(node, "to")))]
        _fold_constant(t, node, t.constants[node.input[0]].astype(target))
        return
    source = t.operand(node, 0)
    kind = t.result(node)
    if kind.dtype == source.type.dtype:
        t.define(node, source)
    else:
        t.define(node, t.builder.op("cast", [source], kind, name=node.output[0]))


@_handles("Transpose")
def _transpose(t: _Translator, node: Any) -> None:
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


@_handles("Slice")
def _slice(t: _Translator, node: Any) -> None:
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
    per_axis: list[dict[str, Any]] = [
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


_Pattern = str | _Const | tuple[Any, ...]
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

    def _last_axis(self, node: Any) -> bool:
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


def _recover(t: _Translator, node: Any) -> _Value | None:
    """The library operation whose decomposition `node` completes, if any."""
    result = node.output[0]
    kind = t.result(node)
    dims = [t.symbols.json(d) for d in kind.shape]

    def recovered(name: str, generics: list[dict[str, Any]], operands: list[_Value]) -> _Value:
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
def _matmul(t: _Translator, node: Any) -> None:
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
def _gemm(t: _Translator, node: Any) -> None:
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
    def handler(t: _Translator, node: Any) -> None:
        x = t.operand(node, 0)
        shape = list(x.type.shape)
        rank = len(shape)
        axes_attr = t.attr(node, "axes")
        if axes_attr is None and len(node.input) > 1 and node.input[1]:
            axes_attr = t.dims_of(node.input[1])
        reduced = sorted(int(a) % rank for a in axes_attr) if axes_attr else list(range(rank))
        keepdims = int(t.attr(node, "keepdims", 1)) == 1
        result = t.result(node)
        kept = [i for i in range(rank) if i not in reduced]

        def body(out: list[_Value]) -> _Value:
            def inner(red: list[_Value]) -> _Value:
                indices: list[_Value] = []
                outer_iter, inner_iter = iter(out), iter(red)
                for axis in range(rank):
                    indices.append(next(inner_iter) if axis in reduced else next(outer_iter))
                return t.builder.element(x, indices)

            total = t.builder.reduce(
                kind, [(f"r{a}", shape[a]) for a in reduced], result.dtype, inner
            )
            if divide:
                count = 1
                for a in reduced:
                    size = shape[a]
                    if not isinstance(size, int):
                        raise OnnxImportError("mean over a symbolic axis is not supported")
                    count *= size
                divisor = t.builder.op(
                    "cast", [t.builder.const_dim(count)], _Type((), result.dtype)
                )
                total = t.builder.op("div", [total, divisor], _Type((), result.dtype))
            return total

        if not kept:
            value = body([])
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
            return
        value = t.builder.comprehension(
            [(f"o{a}", shape[a]) for a in kept],
            result.dtype,
            body,
            name="" if keepdims else node.output[0],
        )
        if keepdims:
            _reshape_to(t, node, value, result)
        else:
            t.define(node, value)

    return handler


_handles("ReduceSum")(_reduction("sum", divide=False))
_handles("ReduceMean")(_reduction("sum", divide=True))
_handles("ReduceMax")(_reduction("max", divide=False))
_handles("ReduceMin")(_reduction("min", divide=False))
