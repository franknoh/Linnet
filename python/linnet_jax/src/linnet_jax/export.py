"""JAX -> Linnet: capture a function with `jax.export` and write its
architecture as a `.linnet` module.

`jax.export` produces StableHLO with static shapes. Each operation of the
module is translated into the Core IR plan format (docs/plan-format.md): the
parameter pytree becomes a block hierarchy (`param` leaves, `sub` members for
nested dicts, sub arrays for lists of identical children), the function
becomes the root block's `entry`, and StableHLO operations become Linnet
primitives — elementwise ops, `reshape`/`permute`/`broadcast_to`, slices and
concatenation — or index notation for `dot_general`, `reduce`, and the
row-lookup form of `gather`. `linnet emit` prints the plan as source and the
result is checked before it is written.

Operations without a verified mapping stop the export with a diagnostic that
names every one of them; custom calls are never mapped.
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

import jax
import numpy as np

from .compiler import LinnetError, find_compiler


class ExportError(LinnetError):
    """The function cannot be expressed in Linnet as captured."""


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


_MLIR_DTYPES = {
    "i1": "bool",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "ui8": "u8",
    "ui16": "u16",
    "ui32": "u32",
    "ui64": "u64",
    "f16": "f16",
    "bf16": "bf16",
    "f32": "f32",
    "f64": "f64",
}

_NUMPY_DTYPES: dict[str, Any] = {
    "bool": np.bool_,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
}


@dataclass
class _Type:
    """A tensor (or scalar, rank 0) type of the module."""

    shape: tuple[int, ...]
    dtype: str  # Linnet dtype name

    @property
    def is_scalar(self) -> bool:
        return len(self.shape) == 0


def _parse_type(text: str) -> _Type:
    match = re.fullmatch(r"tensor<((?:\d+x)*)(\w+)>", text)
    if match is None:
        raise ExportError(f"type `{text}` is not a static ranked tensor")
    dims = tuple(int(d) for d in match.group(1).split("x") if d)
    element = match.group(2)
    if element not in _MLIR_DTYPES:
        raise ExportError(f"element type `{element}` has no Linnet equivalent")
    return _Type(dims, _MLIR_DTYPES[element])


@dataclass
class _Value:
    id: int
    type: _Type
    # Compile-time facts for constants: a splat value, so fills stay fills.
    splat: float | int | bool | None = None


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
    leaf: _Type | None = None  # param: its tensor type


class _Builder:
    """Accumulates the plan's operations and regions."""

    def __init__(self) -> None:
        self.next_id = 0
        self.regions: list[list[dict[str, Any]]] = [[]]

    @staticmethod
    def type_json(kind: _Type) -> dict[str, Any]:
        if kind.is_scalar:
            return {"kind": "scalar", "dtype": kind.dtype}
        return {"kind": "tensor", "shape": list(kind.shape), "dtype": kind.dtype}

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
        for operand in operands:
            if operand.id < 0:
                raise ExportError("a non-finite constant is used in arithmetic")
        results: list[_Value] = [self.fresh(result)] if result is not None else []
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
            result = self.op("const.bool", [], kind, {"value": 1 if value else 0})
        elif dtype.startswith("f") or dtype == "bf16":
            result = self.op("const.float", [], kind, {"value": float(value)})
        else:
            result = self.op("const.int", [], kind, {"value": int(value)})
        result.splat = value
        return result

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
        indices: Sequence[tuple[str, int]],
        dtype: str,
        body: Callable[[list[_Value]], _Value],
        name: str = "",
    ) -> _Value:
        index_type = _Type((), "i64")
        region = self.region([(n, index_type) for n, _ in indices], body)
        return self.op(
            "comprehension",
            [],
            _Type(tuple(d for _, d in indices), dtype),
            {"indices": [{"name": n, "domain": [d]} for n, d in indices]},
            [region],
            name,
        )

    def reduce(
        self,
        kind: str,
        indices: Sequence[tuple[str, int]],
        dtype: str,
        body: Callable[[list[_Value]], _Value],
    ) -> _Value:
        index_type = _Type((), "i64")
        region = self.region([(n, index_type) for n, _ in indices], body)
        return self.op(
            "reduce",
            [],
            _Type((), dtype),
            {"indices": [{"name": n, "domain": [d]} for n, d in indices], "reduce": kind},
            [region],
        )

    def element(self, tensor: _Value, indices: Sequence[_Value]) -> _Value:
        return self.op("tensor.element", [tensor, *indices], _Type((), tensor.type.dtype))


@dataclass
class ExportResult:
    source: Path
    module: str
    root: str
    weights: Path | None
    bindings: Path | None
    plan: dict[str, Any]
    notes: list[str]  # what the translation had to drop, if anything


def export_linnet(
    function: Callable[..., Any],
    params: Any,
    example_inputs: Sequence[Any],
    *,
    output: str | Path,
    module_name: str | None = None,
    root_name: str = "Model",
    weights: str | Path | None = None,
    std_root: str | Path | None = None,
) -> ExportResult:
    """Writes `function(params, *inputs)` as Linnet source at `output`.

    `params` is a pytree of arrays (nested dicts, and lists of identical
    subtrees become sub arrays); its leaves are the model's parameters and
    their paths (`layers.0.kernel`) are the Linnet parameter paths. The
    example inputs fix the input shapes and dtypes. With `weights`, the
    parameters are saved there as SafeTensors under those paths (with a
    `bindings.json` when a path had to be renamed).
    """
    from jax import export

    exported = export.export(jax.jit(function))(params, *example_inputs)
    return import_stablehlo(
        exported.mlir_module(),
        params,
        output=output,
        module_name=module_name,
        root_name=root_name,
        weights=weights,
        std_root=std_root,
    )


def import_stablehlo(
    text: str,
    params: Any,
    *,
    output: str | Path,
    module_name: str | None = None,
    root_name: str = "Model",
    weights: str | Path | None = None,
    std_root: str | Path | None = None,
) -> ExportResult:
    """Writes a StableHLO module as Linnet source at `output`.

    `text` is the module as `jax.export` prints it: `@main` takes the
    flattened leaves of `params` (dict keys sorted) followed by the entry's
    inputs, all with static shapes. `params` gives the parameter tree; its
    leaves may be arrays or anything with `shape` and `dtype`, such as
    `jax.ShapeDtypeStruct`, in which case `weights` cannot be written.
    """
    output_path = Path(output)
    module = module_name or _identifier(output_path.stem)
    hierarchy = _Hierarchy(module)
    root = hierarchy.describe(params, root_name)
    leaves = [path for path, _ in _flatten_with_paths(params)]
    translator = _Translator(text, hierarchy, root, leaves, module)
    plan = translator.run()

    compiler = find_compiler()
    emitted = subprocess.run(
        [compiler, "emit", "-"], input=json.dumps(plan), capture_output=True, text=True, check=False
    )
    if emitted.returncode != 0:
        raise ExportError("the compiler rejected the exported plan:\n" + emitted.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(emitted.stdout, encoding="utf-8")
    check = [compiler, "check", str(output_path)]
    if std_root is not None:
        check[2:2] = ["--std", str(std_root)]
    checked = subprocess.run(check, capture_output=True, text=True, check=False)
    if checked.returncode != 0:
        raise ExportError("the exported source does not check:\n" + checked.stderr + checked.stdout)

    weights_path: Path | None = None
    bindings_path: Path | None = None
    if weights is not None:
        from safetensors.numpy import save_file  # type: ignore[import-untyped]

        weights_dir = Path(weights)
        weights_dir.mkdir(parents=True, exist_ok=True)
        tensors = {
            path: np.ascontiguousarray(np.asarray(leaf))
            for path, leaf in _flatten_with_paths(params)
        }
        if any(array.dtype == object for array in tensors.values()):
            raise ExportError("weights need concrete arrays for every parameter")
        tensors.update(translator.constants)
        weights_path = weights_dir / "model.safetensors"
        save_file(tensors, str(weights_path))
        if hierarchy.renamed:
            bindings_path = weights_dir / "bindings.json"
            bindings_path.write_text(
                json.dumps(hierarchy.renamed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    return ExportResult(
        output_path,
        module,
        root_name,
        weights_path,
        bindings_path,
        plan,
        sorted(set(translator.notes)),
    )


def _flatten_with_paths(tree: Any) -> list[tuple[str, Any]]:
    """Leaves of a pytree with dotted paths, in the order `jax.export`
    flattens them (dict keys sorted)."""
    from jax.tree_util import DictKey, GetAttrKey, SequenceKey, tree_flatten_with_path

    out: list[tuple[str, Any]] = []
    leaves, _ = tree_flatten_with_path(tree)
    for key_path, leaf in cast(list[tuple[tuple[Any, ...], Any]], leaves):
        parts: list[str] = []
        for key in key_path:
            if isinstance(key, DictKey):
                parts.append(str(key.key))
            elif isinstance(key, SequenceKey):
                parts.append(str(key.idx))
            elif isinstance(key, GetAttrKey):
                parts.append(str(key.name))
            else:
                raise ExportError(f"unsupported pytree key {key!r}")
        out.append((".".join(parts), leaf))
    return out


def _is_indexed(node: dict[Any, Any]) -> bool:
    keys = [str(k) for k in node]
    return (
        bool(keys)
        and all(k.isdigit() for k in keys)
        and sorted(map(int, keys)) == list(range(len(keys)))
    )


def _leaf_type(node: Any) -> _Type:
    """The tensor type of a parameter leaf: an array, or anything with
    `shape` and `dtype` such as `jax.ShapeDtypeStruct`."""
    if hasattr(node, "shape") and hasattr(node, "dtype"):
        return _Type(tuple(int(d) for d in node.shape), _dtype_name(np.dtype(node.dtype)))
    return _leaf_type(np.asarray(node))


class _Hierarchy:
    def __init__(self, module: str) -> None:
        self.module = module
        self.blocks: dict[str, _Block] = {}
        self._by_signature: dict[Any, _Block] = {}
        self.renamed: dict[str, str] = {}

    def describe(self, params: Any, root_name: str) -> _Member:
        signature, member = self._describe(params, root_name)
        if member is None:
            raise ExportError("the parameter tree has no arrays")
        member.block = self._block_for(root_name, signature, member)
        return member

    def _describe(self, node: Any, name_hint: str) -> tuple[Any, _Member | None]:
        if isinstance(node, dict) and _is_indexed(node):
            # A dict keyed 0..n-1 (Flax and Haiku layer stacks) is a sub array,
            # so `layers.0.q` names the same member either way.
            keys = sorted(cast(dict[Any, Any], node), key=lambda k: int(str(k)))
            return self._describe([node[k] for k in keys], name_hint)
        if isinstance(node, dict):
            entries: list[Any] = []
            children: dict[str, _Member] = {}
            # Members in the order `jax.tree_util` flattens a dict (sorted keys).
            for key in sorted(cast(dict[Any, Any], node), key=str):
                signature, child = self._describe(node[key], str(key))
                if child is None:
                    continue
                child.name = _identifier(str(key))
                if child.kind == "sub" and child.length is None:
                    class_name = str(key) if not str(key).isdigit() else f"{name_hint}_{key}"
                    child.block = self._block_for(class_name.capitalize(), signature, child)
                children[str(key)] = child
                entries.append((str(key), signature))
            if not entries:
                return (), None
            return tuple(entries), _Member("sub", "", children=children)
        if isinstance(node, list | tuple):
            described = [self._describe(child, name_hint) for child in cast(Sequence[Any], node)]
            if not described or any(m is None for _, m in described):
                return (), None
            signatures = {s for s, _ in described}
            if len(signatures) != 1:
                raise ExportError("lists of parameters must have identical elements")
            signature = described[0][0]
            block = self._block_for(
                name_hint.rstrip("s").capitalize(), signature, cast(_Member, described[0][1])
            )
            children: dict[str, _Member] = {}
            for i, (_, child) in enumerate(described):
                cast(_Member, child).block = block
                children[str(i)] = cast(_Member, child)
            return ("array", len(described), signature), _Member(
                "sub", "", length=len(described), element=block, children=children
            )
        leaf = _leaf_type(node)
        return ("param", leaf.dtype, leaf.shape), _Member("param", "", leaf=leaf)

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
                leaf = cast(_Type, child.leaf)
                block.members.append(
                    {"name": child.name, "kind": "param", "type": _Builder.type_json(leaf)}
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
            assert member.element is not None
            return {
                "kind": "array",
                "element": self.block_type(member.element),
                "length": member.length,
            }
        assert member.block is not None
        return self.block_type(member.block)


class _Translator:
    def __init__(
        self, text: str, hierarchy: _Hierarchy, root: _Member, leaves: list[str], module: str
    ) -> None:
        self.text = text
        self.hierarchy = hierarchy
        self.root = root
        self.leaves = leaves
        self.module = module
        self.builder = _Builder()
        self.self_value = self.builder.fresh(_Type((), "f32"))
        self.functions: dict[str, Any] = {}
        self.values: dict[Any, _Value] = {}
        self.unsupported: list[str] = []
        self.notes: list[str] = []  # information lost on the way, reported to the caller
        self.constants: dict[str, Any] = {}
        self.imports: set[str] = set()

    def run(self) -> dict[str, Any]:
        from jax._src.interpreters import mlir as jax_mlir  # pyright: ignore[reportPrivateUsage]
        from jax._src.lib.mlir import ir  # pyright: ignore[reportPrivateUsage]

        mlir: Any = jax_mlir
        with mlir.make_ir_context():
            module = cast(Any, ir.Module).parse(self.text)
            functions = {
                str(f.attributes["sym_name"]).strip('"'): f
                for f in module.body.operations
                if f.operation.name == "func.func"
            }
            self.functions = functions
            function = functions.get("main", module.body.operations[0])
            block = function.regions[0].blocks[0]
            argument_types = [_parse_type(str(a.type)) for a in block.arguments]
            self_value = self.self_value
            inputs: list[tuple[str, _Value]] = []
            parameter_count = len(self.leaves)
            if len(argument_types) < parameter_count:
                raise ExportError("the exported function has fewer arguments than parameters")
            for i, argument in enumerate(block.arguments):
                kind = argument_types[i]
                if i < parameter_count:
                    self.values[_key(argument)] = self._member_value(
                        self.leaves[i], kind, self_value
                    )
                else:
                    value = self.builder.fresh(kind)
                    inputs.append((f"x{i - parameter_count}", value))
                    self.values[_key(argument)] = value
            outputs = self._translate_block(block)
        if self.unsupported:
            raise ExportError(
                "the function uses operations without a Linnet mapping:\n  "
                + "\n  ".join(sorted(set(self.unsupported)))
            )
        if len(outputs) != 1:
            raise ExportError("the function must return exactly one array")
        result = outputs[0]
        if result.type.is_scalar:
            raise ExportError("the function must return a tensor, not a scalar")
        self.builder.op("return", [result], None)
        root_block = cast(_Block, self.root.block)
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
        for name, array in self.constants.items():
            root_block.members.append(
                {
                    "name": name,
                    "kind": "buffer",
                    "type": {
                        "kind": "tensor",
                        "shape": list(array.shape),
                        "dtype": _numpy_dtype_name(array),
                    },
                }
            )
        body = {
            "args": [
                self.builder.value_json(self_value, "self", self.hierarchy.block_type(root_block))
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
                    "generics": [],
                    "constraints": [],
                    "results": [self.builder.type_json(result.type)],
                    "body": body,
                }
            ],
        }

    # ---- parameters

    def _member_value(self, path: str, kind: _Type, self_value: _Value) -> _Value:
        member = self.root
        current = self_value
        parts = path.split(".")
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
                if linnet_path != path:
                    self.hierarchy.renamed[linnet_path] = path
                return self.builder.op(
                    "block.param", [current], kind, {"name": child.name}, name=child.name
                )
            current = self.builder.op(
                "block.sub", [current], _Type((), "f32"), {"name": child.name},
                type_json=self.hierarchy.member_type(child),
            )  # fmt: skip
            member = child
        raise ExportError(f"cannot resolve `{path}`")

    # ---- translation

    def _translate_block(self, block: Any) -> list[_Value]:
        """Translates a block's operations; returns what it returns."""
        outputs: list[_Value] = []
        for op in block.operations:
            operation = op.operation
            if operation.name == "func.return":
                if not self.unsupported:
                    outputs = [self.values[_key(o)] for o in operation.operands]
                continue
            self._translate(operation)
        return outputs

    def _call(self, operation: Any) -> None:
        """Inlines a call to another function of the module."""
        callee = (self.attribute(operation, "callee") or "").lstrip("@")
        function = self.functions.get(callee)
        if function is None:
            self.unsupported.append(f"func.call @{callee}")
            return
        block = function.regions[0].blocks[0]
        for argument, operand in zip(block.arguments, operation.operands, strict=True):
            self.values[_key(argument)] = self.values[_key(operand)]
        outputs = self._translate_block(block)
        for result, value in zip(operation.results, outputs, strict=False):
            self.values[_key(result)] = value

    def _translate(self, operation: Any) -> None:
        name = str(operation.name)
        for operand in operation.operands:
            if _key(operand) not in self.values:
                return  # downstream of an unsupported operation
        if name == "func.call":
            self._call(operation)
            return
        handler = _HANDLERS.get(name)
        if handler is None:
            # Composites and custom calls carry the name that matters.
            detail = self.attribute(operation, "name") or self.attribute(
                operation, "call_target_name"
            )
            self.unsupported.append(f"{name} {detail}" if detail else name)
            return
        try:
            value = handler(self, operation)
        except ExportError as error:
            self.unsupported.append(f"{name}: {error}")
            return
        if value is not None and len(operation.results) == 1:
            self.values[_key(operation.results[0])] = value

    def operand(self, operation: Any, index: int) -> _Value:
        return self.values[_key(operation.operands[index])]

    @staticmethod
    def result_type(operation: Any) -> _Type:
        return _parse_type(str(operation.results[0].type))

    @staticmethod
    def attribute(operation: Any, name: str) -> str | None:
        for i in range(len(operation.attributes)):
            named = operation.attributes[i]
            if named.name == name:
                return str(named.attr)
        return None

    def int_array(self, operation: Any, name: str) -> list[int]:
        text = self.attribute(operation, name)
        if text is None:
            return []
        match = re.fullmatch(r"array<i64(?::\s*([-\d,\s]*))?>", text)
        if match is None:
            raise ExportError(f"attribute `{name}` is not an integer array: {text}")
        body = match.group(1) or ""
        return [int(x) for x in body.split(",") if x.strip()]

    def call(
        self, callee: str, generics: list[dict[str, Any]], operands: list[_Value], kind: _Type
    ) -> _Value:
        """A call to a standard-library operation with explicit generics."""
        return self.builder.op(
            "semantic.call",
            operands,
            kind,
            {
                "callee": callee,
                "substitution": {"dims": {}, "packs": {}, "dtypes": {}},
                "generics": generics,
            },
        )


def _key(value: Any) -> Any:
    """MLIR values hash by identity, so they key the value map directly;
    names like `%arg0` repeat across functions."""
    return value


def _numpy_dtype_name(array: Any) -> str:
    return _dtype_name(np.asarray(array).dtype)


def _dtype_name(dtype: Any) -> str:
    for name, candidate in _NUMPY_DTYPES.items():
        if np.dtype(candidate) == dtype:
            return name
    if str(dtype) == "bfloat16":
        return "bf16"
    raise ExportError(f"dtype {dtype} has no Linnet equivalent")


Handler = Callable[[_Translator, Any], _Value | None]
_HANDLERS: dict[str, Handler] = {}


def _handles(*names: str) -> Callable[[Handler], Handler]:
    def register(handler: Handler) -> Handler:
        for name in names:
            _HANDLERS[name] = handler
        return handler

    return register


@_handles("stablehlo.constant")
def _constant(t: _Translator, operation: Any) -> _Value:
    kind = t.result_type(operation)
    text = t.attribute(operation, "value") or ""
    match = re.fullmatch(r"dense<(.*)> : tensor<.*>", text, re.DOTALL)
    if match is None:
        raise ExportError(f"unreadable constant {text}")
    literal = match.group(1).strip()
    if literal.startswith("["):
        # A non-splat constant: saved with the weights as a buffer.
        array = _dense_to_numpy(literal, kind)
        name = f"const_{len(t.constants)}"
        t.constants[name] = array
        return t.builder.op("block.param", [t.self_value], kind, {"name": name}, name=name)
    value: float | int | bool
    if kind.dtype == "bool":
        value = literal == "true"
    elif literal.startswith("0x"):
        value = _hex_float(literal, kind.dtype)
    elif kind.dtype.startswith("f") or kind.dtype == "bf16":
        value = float(literal)
    else:
        value = int(literal)
    if isinstance(value, float) and not np.isfinite(value):
        # Only reduction identities are infinite; they never reach the plan.
        return _Value(-1, kind, value)
    scalar = t.builder.const(value, kind.dtype)
    if kind.is_scalar:
        return scalar
    result = t.builder.op("fill", [scalar], kind, {"shape": list(kind.shape)})
    result.splat = value
    return result


def _hex_float(literal: str, dtype: str) -> float:
    bits = int(literal, 16)
    if dtype == "f32":
        return float(np.array([bits], dtype=np.uint32).view(np.float32)[0])
    if dtype == "f64":
        return float(np.array([bits], dtype=np.uint64).view(np.float64)[0])
    if dtype == "f16":
        return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])
    if dtype == "bf16":
        return float(np.array([bits << 16], dtype=np.uint32).view(np.float32)[0])
    raise ExportError(f"hexadecimal literal for {dtype}")


def _dense_to_numpy(literal: str, kind: _Type) -> Any:
    numbers = [x for x in re.split(r"[\[\],\s]+", literal) if x]
    if kind.dtype == "bool":
        values = [x == "true" for x in numbers]
    elif kind.dtype.startswith("f") or kind.dtype == "bf16":
        values = [_hex_float(x, kind.dtype) if x.startswith("0x") else float(x) for x in numbers]
    else:
        values = [int(x) for x in numbers]
    dtype = _NUMPY_DTYPES.get(kind.dtype, np.float32)
    return np.asarray(values, dtype=dtype).reshape(kind.shape)


_ELEMENTWISE = {
    "stablehlo.add": "add",
    "stablehlo.subtract": "sub",
    "stablehlo.multiply": "mul",
    "stablehlo.divide": "div",
    "stablehlo.remainder": "rem",
    "stablehlo.maximum": "max",
    "stablehlo.minimum": "min",
    "stablehlo.negate": "neg",
    "stablehlo.exponential": "exp",
    "stablehlo.log": "log",
    "stablehlo.sqrt": "sqrt",
    "stablehlo.rsqrt": "rsqrt",
    "stablehlo.sine": "sin",
    "stablehlo.cosine": "cos",
    "stablehlo.tanh": "tanh",
    "stablehlo.abs": "abs",
}


# ---- exact pattern recovery
#
# JAX lowers `jax.nn.softmax`, `jax.nn.sigmoid`, `jax.nn.silu`, the tanh
# `jax.nn.gelu`, and an RMS norm written in `jnp` to fixed shapes of
# primitive operations. When a node completes one of those shapes over the
# same operand, the translation emits the standard-library operation
# instead; the primitives already emitted become dead and are dropped, and
# every recovery is recorded in the notes. The check is structural: the
# operation names, operands, axes, and constants must all be the ones the
# decomposition uses, so nothing is recovered by name or by approximation.
#
# A pattern is a variable name (binds the value on first sight, must be the
# same value afterwards), a `_Const` (a splat constant with that value, or
# any constant bound under `name`), a `("reduce", combine, operand)` over
# the last axis, or `("<stablehlo op>", *operand patterns)`; commutative
# operations match in either operand order.


@dataclass(frozen=True)
class _Const:
    value: float | None = None
    name: str | None = None


_Pattern = str | _Const | tuple[Any, ...]
_COMMUTATIVE = {"add", "multiply", "maximum", "minimum"}


def _owner(value: Any) -> Any:
    """The operation producing an MLIR value, or None for block arguments."""
    owner = value.owner
    return owner if hasattr(owner, "operands") else None


def _strip_broadcasts(t: _Translator, value: Any) -> Any:
    """Skips `broadcast_in_dim` chains that keep the axis order (adding
    size-one axes or broadcasting them), returning the source value."""
    while True:
        owner = _owner(value)
        if owner is None or str(owner.name) != "stablehlo.broadcast_in_dim":
            return value
        dims = t.int_array(owner, "broadcast_dimensions")
        if dims != sorted(dims):
            return value
        value = owner.operands[0]


def _reduce_over_last(t: _Translator, value: Any, combine: str) -> Any | None:
    """The operand of a `reduce` with the given body over the last axis, or None."""
    owner = _owner(value)
    if owner is None:
        return None
    if str(owner.name) == "stablehlo.maximum":
        # The `maximum(-inf, reduce)` guard of an empty reduction.
        for i in range(2):
            candidate = _owner(_strip_broadcasts(t, owner.operands[i]))
            if candidate is not None and str(candidate.name) == "stablehlo.constant":
                return _reduce_over_last(t, owner.operands[1 - i], combine)
        return None
    if str(owner.name) != "stablehlo.reduce" or len(owner.operands) != 2:
        return None
    body = [str(o.operation.name) for o in owner.regions[0].blocks[0].operations]
    if body[:1] != [f"stablehlo.{combine}"]:
        return None
    rank = len(_parse_type(str(owner.operands[0].type)).shape)
    if t.int_array(owner, "dimensions") != [rank - 1]:
        return None
    return owner.operands[0]


class _Matcher:
    def __init__(self, t: _Translator) -> None:
        self.t = t
        self.bound: dict[str, Any] = {}

    def value(self, name: str) -> _Value | None:
        return self.t.values.get(_key(self.bound[name]))

    def splat(self, name: str) -> float | int | bool | None:
        value = self.value(name)
        return None if value is None else value.splat

    def match(self, value: Any, pattern: _Pattern) -> bool:
        value = _strip_broadcasts(self.t, value)
        if isinstance(pattern, str):
            if pattern in self.bound:
                return bool(self.bound[pattern] == value)
            self.bound[pattern] = value
            return True
        if isinstance(pattern, _Const):
            owner = _owner(value)
            if owner is None or str(owner.name) != "stablehlo.constant":
                return False
            translated = self.t.values.get(_key(value))
            if translated is None or translated.splat is None:
                return False
            if pattern.value is not None and not math.isclose(
                float(translated.splat), pattern.value, rel_tol=1e-6, abs_tol=1e-12
            ):
                return False
            if pattern.name is not None:
                self.bound[pattern.name] = value
            return True
        name, *operands = pattern
        if name == "reduce":
            source = _reduce_over_last(self.t, value, operands[0])
            return source is not None and self.match(source, operands[1])
        owner = _owner(value)
        if owner is None or str(owner.name) != f"stablehlo.{name}":
            return False
        if len(owner.operands) != len(operands):
            return False
        if name in _COMMUTATIVE and len(operands) == 2:
            saved = dict(self.bound)
            if self.match(owner.operands[0], operands[0]) and self.match(
                owner.operands[1], operands[1]
            ):
                return True
            self.bound = saved
            return self.match(owner.operands[0], operands[1]) and self.match(
                owner.operands[1], operands[0]
            )
        return all(self.match(o, p) for o, p in zip(owner.operands, operands, strict=True))


_SIGMOID: _Pattern = ("divide", _Const(1.0), ("add", _Const(1.0), ("exponential", ("negate", "x"))))
_SILU: _Pattern = ("multiply", "x", _SIGMOID)
_GELU_TANH: _Pattern = (
    "multiply",
    "x",
    (
        "multiply",
        _Const(0.5),
        (
            "add",
            _Const(1.0),
            (
                "tanh",
                (
                    "multiply",
                    _Const(0.7978845608),
                    (
                        "add",
                        "x",
                        ("multiply", _Const(0.044715), ("multiply", ("multiply", "x", "x"), "x")),
                    ),
                ),
            ),
        ),
    ),
)
_SOFTMAX: _Pattern = ("divide", "e", ("reduce", "add", "e"))
_SOFTMAX_NUMERATOR: _Pattern = ("exponential", ("subtract", "x", ("reduce", "maximum", "x")))
_RMS_NORM: _Pattern = (
    "multiply",
    (
        "multiply",
        "x",
        (
            "rsqrt",
            (
                "add",
                ("divide", ("reduce", "add", ("multiply", "x", "x")), _Const(name="count")),
                _Const(name="eps"),
            ),
        ),
    ),
    "w",
)

# Shape-preserving library operations and their decompositions, by the
# operation completing them.
_ACTIVATIONS: dict[str, list[tuple[str, _Pattern]]] = {
    "stablehlo.divide": [("sigmoid", _SIGMOID)],
    "stablehlo.multiply": [("gelu", _GELU_TANH), ("silu", _SILU)],
}


def _recovered(
    t: _Translator, name: str, generics: list[dict[str, Any]], operands: list[_Value], kind: _Type
) -> _Value:
    t.notes.append(f"recovered {name} from its decomposition")
    module = {"softmax": "std.nn.softmax", "rms_norm": "std.nn.norm"}.get(
        name, "std.nn.activations"
    )
    return t.call(f"{module}::{name}", generics, operands, kind)


def _recover(t: _Translator, operation: Any) -> _Value | None:
    """The library operation whose decomposition `operation` completes, if any."""
    result = operation.results[0]
    kind = t.result_type(operation)
    shape = list(kind.shape)
    for name, pattern in _ACTIVATIONS.get(str(operation.name), []):
        m = _Matcher(t)
        x = m.value("x") if m.match(result, pattern) else None
        if x is not None and x.type.shape == kind.shape:
            return _recovered(t, name, [{"shape": shape}, {"dtype": kind.dtype}], [x], kind)
    if str(operation.name) == "stablehlo.divide":
        m = _Matcher(t)
        if m.match(result, _SOFTMAX) and m.match(m.bound["e"], _SOFTMAX_NUMERATOR):
            x = m.value("x")
            if x is not None and x.type.shape == kind.shape and shape:
                generics = [{"shape": shape[:-1]}, {"dim": shape[-1]}, {"dtype": kind.dtype}]
                return _recovered(t, "softmax", generics, [x], kind)
    if str(operation.name) == "stablehlo.multiply":
        m = _Matcher(t)
        if m.match(result, _RMS_NORM):
            x, w, eps = m.value("x"), m.value("w"), m.splat("eps")
            if (
                x is not None
                and w is not None
                and eps is not None
                and shape
                and x.type.shape == kind.shape
                and w.type.shape == kind.shape[-1:]
                and m.splat("count") == shape[-1]
            ):
                generics = [{"shape": shape[:-1]}, {"dim": shape[-1]}, {"dtype": kind.dtype}]
                epsilon = t.builder.const(float(eps), "f32")
                return _recovered(t, "rms_norm", generics, [x, w, epsilon], kind)
    return None


@_handles(*_ELEMENTWISE)
def _elementwise(t: _Translator, operation: Any) -> _Value:
    recovered = _recover(t, operation)
    if recovered is not None:
        return recovered
    operands = [t.operand(operation, i) for i in range(len(operation.operands))]
    kind = _ELEMENTWISE[str(operation.name)]
    # `maximum(-inf, x)` and `minimum(+inf, x)` guard empty reductions in
    # JAX's lowering of softmax and friends; they are identities.
    if len(operands) == 2 and kind in ("max", "min"):
        for i in range(2):
            splat = operands[i].splat
            is_infinite = operands[i].id < 0 and isinstance(splat, float) and np.isinf(splat)
            if is_infinite and (kind == "max") == (cast(float, splat) < 0):
                return operands[1 - i]
    return t.builder.op(kind, operands, t.result_type(operation))


@_handles("stablehlo.and", "stablehlo.or")
def _logical(t: _Translator, operation: Any) -> _Value:
    kind = t.result_type(operation)
    left, right = t.operand(operation, 0), t.operand(operation, 1)
    if kind.dtype != "bool":
        raise ExportError("bitwise operations on integers are not supported")
    is_and = str(operation.name).endswith("and")
    if kind.is_scalar:
        return t.builder.op("and" if is_and else "or", [left, right], kind)
    constant = t.builder.const(not is_and, "bool")
    return t.builder.op(
        "select", [left, right, constant] if is_and else [left, constant, right], kind
    )


@_handles("stablehlo.not")
def _not(t: _Translator, operation: Any) -> _Value:
    kind = t.result_type(operation)
    x = t.operand(operation, 0)
    if kind.dtype != "bool":
        raise ExportError("bitwise not on integers is not supported")
    if kind.is_scalar:
        return t.builder.op("not", [x], kind)
    return t.builder.op(
        "select", [x, t.builder.const(False, "bool"), t.builder.const(True, "bool")], kind
    )


@_handles("stablehlo.compare")
def _compare(t: _Translator, operation: Any) -> _Value:
    direction = t.attribute(operation, "comparison_direction") or ""
    match = re.search(r"comparison_direction (\w+)", direction)
    if match is None:
        raise ExportError(f"unreadable comparison {direction}")
    kinds = {"EQ": "eq", "NE": "ne", "LT": "lt", "LE": "le", "GT": "gt", "GE": "ge"}
    return t.builder.op(
        "compare",
        [t.operand(operation, 0), t.operand(operation, 1)],
        t.result_type(operation),
        {"compare": kinds[match.group(1)]},
    )


@_handles("stablehlo.select")
def _select(t: _Translator, operation: Any) -> _Value:
    condition, on_true, on_false = (t.operand(operation, i) for i in range(3))
    # `take` guards out-of-bounds rows with NaN; Linnet indexing is checked
    # instead, so the guard is dropped and the export says so.
    for guarded, other in ((on_false, on_true), (on_true, on_false)):
        if guarded.id < 0 and isinstance(guarded.splat, float) and np.isnan(guarded.splat):
            t.notes.append(
                "an out-of-bounds guard selecting NaN was dropped; Linnet checks indices"
            )
            return other
    return t.builder.op("select", [condition, on_true, on_false], t.result_type(operation))


@_handles("stablehlo.convert")
def _convert(t: _Translator, operation: Any) -> _Value:
    source = t.operand(operation, 0)
    kind = t.result_type(operation)
    if kind.dtype == source.type.dtype:
        return source
    return t.builder.op("cast", [source], kind)


@_handles("stablehlo.reshape")
def _reshape(t: _Translator, operation: Any) -> _Value:
    source = t.operand(operation, 0)
    kind = t.result_type(operation)
    if kind.shape == source.type.shape:
        return source
    if kind.is_scalar or source.type.is_scalar:
        raise ExportError("reshapes between scalars and tensors are not supported")
    return t.builder.op("reshape", [source], kind, {"shape": list(kind.shape)})


@_handles("stablehlo.transpose")
def _transpose(t: _Translator, operation: Any) -> _Value:
    axes = t.int_array(operation, "permutation")
    if axes == list(range(len(axes))):
        return t.operand(operation, 0)
    return t.builder.op(
        "permute", [t.operand(operation, 0)], t.result_type(operation), {"shape": axes}
    )


@_handles("stablehlo.broadcast_in_dim")
def _broadcast(t: _Translator, operation: Any) -> _Value:
    source = t.operand(operation, 0)
    kind = t.result_type(operation)
    dims = t.int_array(operation, "broadcast_dimensions")
    if source.id < 0:
        return _Value(-1, kind, source.splat)  # a non-finite splat stays symbolic
    if source.type.is_scalar:
        if source.splat is not None:
            result = t.builder.op("fill", [source], kind, {"shape": list(kind.shape)})
            result.splat = source.splat
            return result
        return t.builder.op("fill", [source], kind, {"shape": list(kind.shape)})
    # Place the operand's axes, then broadcast the rest; a permutation of
    # axes would need a transpose first.
    if dims != sorted(dims):
        raise ExportError("broadcast_in_dim with permuted dimensions")
    expanded = [1] * len(kind.shape)
    for axis, size in zip(dims, source.type.shape, strict=True):
        expanded[axis] = size
    value = source
    if tuple(expanded) != source.type.shape:
        value = t.builder.op(
            "reshape", [value], _Type(tuple(expanded), kind.dtype), {"shape": expanded}
        )
    if tuple(expanded) != kind.shape:
        value = t.builder.op("broadcast", [value], kind, {"shape": list(kind.shape)})
    return value


@_handles("stablehlo.iota")
def _iota(t: _Translator, operation: Any) -> _Value:
    kind = t.result_type(operation)
    axis_text = t.attribute(operation, "iota_dimension") or "0"
    axis = int(axis_text.split(":")[0])
    size = kind.shape[axis]
    value = t.builder.op("iota", [], _Type((size,), "i64"), {"shape": [size]})
    if kind.dtype != "i64":
        value = t.builder.op("cast", [value], _Type((size,), kind.dtype))
    if len(kind.shape) > 1:
        expanded = [1] * len(kind.shape)
        expanded[axis] = size
        value = t.builder.op(
            "reshape", [value], _Type(tuple(expanded), kind.dtype), {"shape": expanded}
        )
        value = t.builder.op("broadcast", [value], kind, {"shape": list(kind.shape)})
    return value


@_handles("stablehlo.slice")
def _slice(t: _Translator, operation: Any) -> _Value:
    source = t.operand(operation, 0)
    starts = t.int_array(operation, "start_indices")
    limits = t.int_array(operation, "limit_indices")
    strides = t.int_array(operation, "strides") or [1] * len(starts)
    axes = [
        {"start": s, "stop": e, "step": st, "squeeze": False}
        for s, e, st in zip(starts, limits, strides, strict=True)
    ]
    return t.builder.op("slice", [source], t.result_type(operation), {"axes": axes})


@_handles("stablehlo.concatenate")
def _concatenate(t: _Translator, operation: Any) -> _Value:
    axis = int((t.attribute(operation, "dimension") or "0").split(":")[0])
    parts = [t.operand(operation, i) for i in range(len(operation.operands))]
    return t.builder.op("concat", parts, t.result_type(operation), {"axis": axis})


@_handles("stablehlo.dot_general")
def _dot_general(t: _Translator, operation: Any) -> _Value:
    text = t.attribute(operation, "dot_dimension_numbers") or ""

    def dims(name: str) -> list[int]:
        match = re.search(name + r" = \[([\d,\s]*)\]", text)
        return [int(x) for x in match.group(1).split(",") if x.strip()] if match else []

    lhs_batch, rhs_batch = dims("lhs_batching_dimensions"), dims("rhs_batching_dimensions")
    lhs_contract, rhs_contract = (
        dims("lhs_contracting_dimensions"),
        dims("rhs_contracting_dimensions"),
    )
    a, b = t.operand(operation, 0), t.operand(operation, 1)
    kind = t.result_type(operation)
    if a.type.dtype != b.type.dtype:
        raise ExportError("dot_general operands differ in dtype")
    lhs_free = [i for i in range(len(a.type.shape)) if i not in lhs_batch and i not in lhs_contract]
    rhs_free = [i for i in range(len(b.type.shape)) if i not in rhs_batch and i not in rhs_contract]
    # The matrix-product layouts are the standard library's `matmul` and
    # `batched_matmul` exactly; anything else is spelled in index notation.
    rank = len(a.type.shape)
    is_matrix_product = (
        len(lhs_free) == 1
        and len(rhs_free) == 1
        and len(lhs_contract) == 1
        and lhs_batch == list(range(rank - 2))
        and rhs_batch == list(range(rank - 2))
        and lhs_contract == [rank - 1]
        and rhs_contract == [rank - 2]
        and len(b.type.shape) == rank
    )
    if is_matrix_product:
        m, k, n = a.type.shape[-2], a.type.shape[-1], b.type.shape[-1]
        if rank == 2:
            return t.call(
                "std.linalg::matmul",
                [{"dim": m}, {"dim": k}, {"dim": n}, {"dtype": kind.dtype}],
                [a, b],
                kind,
            )
        return t.call(
            "std.linalg::batched_matmul",
            [
                {"shape": list(a.type.shape[:-2])},
                {"dim": m},
                {"dim": k},
                {"dim": n},
                {"dtype": kind.dtype},
            ],
            [a, b],
            kind,
        )
    outputs: list[tuple[str, int]] = []
    outputs += [(f"b{i}", a.type.shape[d]) for i, d in enumerate(lhs_batch)]
    outputs += [(f"m{i}", a.type.shape[d]) for i, d in enumerate(lhs_free)]
    outputs += [(f"n{i}", b.type.shape[d]) for i, d in enumerate(rhs_free)]
    contracted = [(f"k{i}", a.type.shape[d]) for i, d in enumerate(lhs_contract)]

    def body(out: list[_Value]) -> _Value:
        batch = out[: len(lhs_batch)]
        m = out[len(lhs_batch) : len(lhs_batch) + len(lhs_free)]
        n = out[len(lhs_batch) + len(lhs_free) :]

        def inner(k: list[_Value]) -> _Value:
            a_index: list[_Value] = [_Value(-1, _Type((), "i64"))] * len(a.type.shape)
            b_index: list[_Value] = [_Value(-1, _Type((), "i64"))] * len(b.type.shape)
            for i, d in enumerate(lhs_batch):
                a_index[d] = batch[i]
            for i, d in enumerate(rhs_batch):
                b_index[d] = batch[i]
            for i, d in enumerate(lhs_free):
                a_index[d] = m[i]
            for i, d in enumerate(rhs_free):
                b_index[d] = n[i]
            for i, d in enumerate(lhs_contract):
                a_index[d] = k[i]
            for i, d in enumerate(rhs_contract):
                b_index[d] = k[i]
            product = t.builder.op(
                "mul",
                [t.builder.element(a, a_index), t.builder.element(b, b_index)],
                _Type((), kind.dtype),
            )
            return product

        if not contracted:
            return inner([])
        return t.builder.reduce("sum", contracted, kind.dtype, inner)

    if not outputs:
        raise ExportError("dot_general producing a scalar is not supported")
    return t.builder.comprehension(outputs, kind.dtype, body)


@_handles("stablehlo.reduce")
def _reduce(t: _Translator, operation: Any) -> _Value:
    if len(operation.operands) != 2:
        raise ExportError("reductions of several operands are not supported")
    source = t.operand(operation, 0)
    init = t.operand(operation, 1)
    body_ops = [o.operation.name for o in operation.regions[0].blocks[0].operations]
    if len(body_ops) != 2:
        raise ExportError("reduction bodies must be one operation")
    kinds = {
        "stablehlo.add": ("sum", 0.0),
        "stablehlo.multiply": ("prod", 1.0),
        "stablehlo.maximum": ("max", -np.inf),
        "stablehlo.minimum": ("min", np.inf),
        "stablehlo.or": ("any", False),
        "stablehlo.and": ("all", True),
    }
    if body_ops[0] not in kinds:
        raise ExportError(f"reduction with `{body_ops[0]}` is not supported")
    reduce_kind, identity = kinds[body_ops[0]]
    if init.splat is None or (
        init.splat != identity
        and not (
            isinstance(identity, float) and np.isinf(identity) and float(init.splat) == identity
        )
    ):
        raise ExportError("reduction initial value must be the operation's identity")
    dims = t.int_array(operation, "dimensions")
    kind = t.result_type(operation)
    rank = len(source.type.shape)
    kept = [i for i in range(rank) if i not in dims]

    def body(out: list[_Value]) -> _Value:
        def inner(reduced: list[_Value]) -> _Value:
            indices: list[_Value] = []
            outer_iter, inner_iter = iter(out), iter(reduced)
            for axis in range(rank):
                indices.append(next(inner_iter) if axis in dims else next(outer_iter))
            return t.builder.element(source, indices)

        return t.builder.reduce(
            reduce_kind, [(f"r{d}", source.type.shape[d]) for d in dims], kind.dtype, inner
        )

    if not kept:
        return body([])
    return t.builder.comprehension(
        [(f"o{i}", source.type.shape[i]) for i in kept], kind.dtype, body
    )


@_handles("stablehlo.gather")
def _gather(t: _Translator, operation: Any) -> _Value:
    """The row-lookup form `table[ids]`: one collapsed leading axis, the
    remaining axes offset dimensions, unit slice sizes on the index axis."""
    table, ids = t.operand(operation, 0), t.operand(operation, 1)
    text = t.attribute(operation, "dimension_numbers") or ""

    def dims(name: str) -> list[int]:
        match = re.search(name + r" = \[([\d,\s]*)\]", text)
        return [int(x) for x in match.group(1).split(",") if x.strip()] if match else []

    offset = dims("offset_dims")
    collapsed = dims("collapsed_slice_dims")
    start_map = dims("start_index_map")
    slice_sizes = t.int_array(operation, "slice_sizes")
    kind = t.result_type(operation)
    id_rank = len(ids.type.shape)
    index_vector = re.search(r"index_vector_dim = (\d+)", text)
    vector_dim = int(index_vector.group(1)) if index_vector else id_rank
    # The index vector is either implicit (past the last axis) or a trailing
    # axis of extent one.
    has_vector_axis = vector_dim == id_rank - 1 and ids.type.shape[-1] == 1
    if (
        collapsed != [0]
        or start_map != [0]
        or slice_sizes[:1] != [1]
        or not (vector_dim == id_rank or has_vector_axis)
    ):
        raise ExportError("only row lookups (`table[ids]`) are supported among gathers")
    batch_rank = len(kind.shape) - len(offset)
    if offset != list(range(batch_rank, len(kind.shape))):
        raise ExportError("gather with interleaved offset dimensions")
    id_shape = list(ids.type.shape)
    if has_vector_axis:
        ids = t.builder.op(
            "reshape", [ids], _Type(tuple(id_shape[:-1]), ids.type.dtype), {"shape": id_shape[:-1]}
        )
        id_shape = id_shape[:-1]

    def body(indices: list[_Value]) -> _Value:
        row = t.builder.element(ids, indices[:batch_rank])
        if row.type.dtype != "i64":
            row = t.builder.op("cast", [row], _Type((), "i64"))
        return t.builder.element(table, [row, *indices[batch_rank:]])

    return t.builder.comprehension(
        [(f"i{i}", d) for i, d in enumerate(id_shape)]
        + [(f"h{i}", kind.shape[batch_rank + i]) for i in range(len(offset))],
        kind.dtype,
        body,
    )
