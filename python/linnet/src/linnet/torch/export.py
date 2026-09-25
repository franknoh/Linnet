"""Torch -> Linnet: capture a `torch.nn.Module` with `torch.export` and write
its architecture as a `.linnet` module.

The captured graph is a normalized functional ATen graph with symbolic
shapes. It is translated operation by operation into a Core IR plan document
(docs/plan-format.md) — the module hierarchy becomes blocks with `param`,
`buffer`, and `sub` members, the forward graph becomes the root block's
`entry`, and every ATen operation becomes the smallest Linnet primitive or
standard-library operation that means the same thing — and `linnet emit`
turns the plan into formatted source. Nothing here parses Python source, and
no tensor data enters the `.linnet` file: weights are written separately, on
request, as SafeTensors under their PyTorch names.

The translation is lossless in the sense that it never guesses: an operation
without a verified mapping stops the export with a diagnostic that names it.
"""

from __future__ import annotations

import json
import operator
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from ..compiler import find_compiler
from ..plan import PlanError


class ExportError(PlanError):
    """The model cannot be expressed in Linnet as captured."""


# ----------------------------------------------------------------- naming

_KEYWORDS = {
    "as", "const", "type", "struct", "enum", "fn", "op", "block", "entry", "param", "buffer",
    "state", "sub", "let", "var", "return", "if", "else", "match", "static", "for", "in", "while",
    "where", "true", "false", "none", "some", "extern", "module", "use", "pub",
}  # fmt: skip
_PRELUDE = {
    "Tensor", "Dim", "Shape", "DType", "Numeric", "Integer", "Float", "cast", "reshape",
    "permute", "broadcast_to", "concat", "pad", "iota", "fill", "gather", "scatter", "exp", "log",
    "sqrt", "rsqrt", "sin", "cos", "tanh", "abs", "select", "min", "max", "sum", "prod", "any",
    "all", "bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16", "f32",
    "f64",
}  # fmt: skip
_RESERVED = _KEYWORDS | _PRELUDE


def _identifier(name: str) -> str:
    """A Linnet identifier for a PyTorch attribute or class name."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not clean or clean[0].isdigit():
        clean = "_" + clean
    if clean in _RESERVED:
        clean += "_"
    return clean


_DTYPES: dict[torch.dtype, str] = {
    torch.bool: "bool",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "u8",
    torch.uint16: "u16",
    torch.uint32: "u32",
    torch.uint64: "u64",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
}


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype not in _DTYPES:
        raise ExportError(f"dtype {dtype} has no Linnet equivalent")
    return _DTYPES[dtype]


# --------------------------------------------------------------- dimensions

# A dimension is an int or a sympy expression from torch's symbolic shapes.
Dim = Any


@dataclass
class _Symbols:
    """The symbolic dimensions of the graph, numbered for the plan."""

    ids: dict[Any, int] = field(default_factory=lambda: {})
    names: dict[Any, str] = field(default_factory=lambda: {})

    def declare(self, symbol: Any, name: str | None = None) -> int:
        if symbol not in self.ids:
            self.ids[symbol] = len(self.ids)
            self.names[symbol] = name or f"D{len(self.ids) - 1}"
        elif name is not None and self.names[symbol].startswith("D"):
            self.names[symbol] = name
        return self.ids[symbol]

    def json(self, dim: Dim) -> Any:
        if isinstance(dim, bool):
            raise ExportError("a boolean is not a dimension")
        if isinstance(dim, int):
            return dim
        if isinstance(dim, torch.SymInt):
            return self.json(dim.node.expr)
        import sympy  # type: ignore[import-untyped]  # torch depends on sympy

        if isinstance(dim, sympy.Integer):
            return int(dim)
        if isinstance(dim, sympy.Symbol):
            if dim not in self.ids:
                raise ExportError(f"dimension `{dim}` does not come from an input shape")
            return {"sym": self.ids[dim], "name": self.names[dim]}
        if isinstance(dim, sympy.Add):
            return {"op": "add", "args": [self.json(arg) for arg in dim.args]}
        if isinstance(dim, sympy.Mul):
            return {"op": "mul", "args": [self.json(arg) for arg in dim.args]}
        if isinstance(dim, sympy.Pow) and dim.exp.is_Integer and int(dim.exp) > 0:
            return {"op": "mul", "args": [self.json(dim.base)] * int(dim.exp)}
        from torch.utils._sympy.functions import FloorDiv, Mod

        if isinstance(dim, FloorDiv):
            return {"op": "floordiv", "args": [self.json(dim.args[0]), self.json(dim.args[1])]}
        if isinstance(dim, Mod):
            return {"op": "mod", "args": [self.json(dim.args[0]), self.json(dim.args[1])]}
        if isinstance(dim, sympy.Min | sympy.Max):
            op = "min" if isinstance(dim, sympy.Min) else "max"
            result = self.json(dim.args[0])
            for arg in dim.args[1:]:
                result = {"op": op, "args": [result, self.json(arg)]}
            return result
        raise ExportError(f"dimension expression `{dim}` is not supported")


def _same_dims(a: Sequence[Dim], b: Sequence[Dim]) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b, strict=True))


# ------------------------------------------------------------------- values


@dataclass
class _Value:
    """An IR value of the plan: a tensor with symbolic shape, or a scalar."""

    id: int
    dtype: torch.dtype
    shape: tuple[Dim, ...] | None  # None for a scalar
    type_json: dict[str, Any] | None = None  # blocks, arrays, optionals: the plan type itself

    @property
    def is_tensor(self) -> bool:
        return self.shape is not None and self.type_json is None


@dataclass
class _Block:
    """A Linnet block synthesized from a module class."""

    name: str
    members: list[dict[str, Any]] = field(default_factory=lambda: [])


@dataclass
class _Member:
    """How one attribute of a module maps onto a block member."""

    kind: str  # "param" | "buffer" | "sub"
    name: str
    block: _Block | None = None  # sub: the child's block
    length: int | None = None  # sub: array length when the child is a uniform container
    element: _Block | None = None  # sub: the array element block
    children: dict[str, _Member] = field(default_factory=lambda: {})  # sub: by attribute name


class _Builder:
    """Accumulates the operations of the entry body and its regions."""

    def __init__(self, symbols: _Symbols) -> None:
        self.symbols = symbols
        self.next_id = 0
        self.regions: list[list[dict[str, Any]]] = [[]]

    # ---- types

    def tensor_type(self, shape: Sequence[Dim], dtype: torch.dtype) -> dict[str, Any]:
        return {
            "kind": "tensor",
            "shape": [self.symbols.json(d) for d in shape],
            "dtype": _dtype_name(dtype),
        }

    def scalar_type(self, dtype: torch.dtype) -> dict[str, Any]:
        return {"kind": "scalar", "dtype": _dtype_name(dtype)}

    def type_of(self, value: _Value) -> dict[str, Any]:
        if value.type_json is not None:
            return value.type_json
        if value.shape is None:
            return self.scalar_type(value.dtype)
        return self.tensor_type(value.shape, value.dtype)

    # ---- values and operations

    def fresh(
        self,
        dtype: torch.dtype,
        shape: tuple[Dim, ...] | None,
        type_json: dict[str, Any] | None = None,
    ) -> _Value:
        self.next_id += 1
        return _Value(self.next_id - 1, dtype, shape, type_json)

    def value_json(self, value: _Value, name: str = "") -> dict[str, Any]:
        return {"id": value.id, "name": name, "type": self.type_of(value)}

    def op(
        self,
        kind: str,
        operands: Sequence[_Value],
        result: tuple[torch.dtype, tuple[Dim, ...] | None] | None,
        attrs: dict[str, Any] | None = None,
        regions: Sequence[dict[str, Any]] = (),
        name: str = "",
        type_json: dict[str, Any] | None = None,
    ) -> _Value:
        results: list[_Value] = []
        if result is not None:
            results.append(self.fresh(result[0], result[1], type_json))
        self.regions[-1].append(
            {
                "kind": kind,
                "operands": [v.id for v in operands],
                "results": [self.value_json(v, name) for v in results],
                "attrs": attrs or {},
                "regions": list(regions),
            }
        )
        return results[0] if results else _Value(-1, torch.float32, None)

    def const(self, value: float | int | bool, dtype: torch.dtype) -> _Value:
        if dtype == torch.bool:
            return self.op("const.bool", [], (dtype, None), {"value": 1 if value else 0})
        if dtype.is_floating_point:
            return self.op("const.float", [], (dtype, None), {"value": float(value)})
        return self.op("const.int", [], (dtype, None), {"value": int(value)})

    def const_dim(self, dim: Dim) -> _Value:
        return self.op("const.dim", [], (torch.int64, None), {"value": self.symbols.json(dim)})

    def region(
        self,
        arguments: Sequence[tuple[str, torch.dtype]],
        body: Callable[[list[_Value]], _Value],
    ) -> dict[str, Any]:
        """Runs `body` in a new region whose arguments are scalars; the value
        it returns is yielded."""
        values = [self.fresh(dtype, None) for _, dtype in arguments]
        self.regions.append([])
        yielded = body(values)
        self.op("yield", [yielded], None)
        ops = self.regions.pop()
        return {
            "args": [
                self.value_json(v, name) for v, (name, _) in zip(values, arguments, strict=True)
            ],
            "ops": ops,
        }

    def comprehension(
        self,
        indices: Sequence[tuple[str, Dim]],
        dtype: torch.dtype,
        body: Callable[[list[_Value]], _Value],
        name: str = "",
    ) -> _Value:
        """`let out[i, j, ...] = body(i, j, ...)` over the given index domains."""
        region = self.region([(n, torch.int64) for n, _ in indices], body)
        return self.op(
            "comprehension",
            [],
            (dtype, tuple(d for _, d in indices)),
            {"indices": [{"name": n, "domain": [self.symbols.json(d)]} for n, d in indices]},
            [region],
            name,
        )

    def reduce(
        self,
        kind: str,
        indices: Sequence[tuple[str, Dim]],
        dtype: torch.dtype,
        body: Callable[[list[_Value]], _Value],
    ) -> _Value:
        region = self.region([(n, torch.int64) for n, _ in indices], body)
        return self.op(
            "reduce",
            [],
            (dtype, None),
            {
                "indices": [{"name": n, "domain": [self.symbols.json(d)]} for n, d in indices],
                "reduce": kind,
            },
            [region],
        )

    def element(self, tensor: _Value, indices: Sequence[_Value]) -> _Value:
        return self.op("tensor.element", [tensor, *indices], (tensor.dtype, None))


# -------------------------------------------------------------- hierarchy


class _Hierarchy:
    """Blocks for a module tree, with parameter paths kept identical to
    PyTorch's wherever Linnet can spell them."""

    def __init__(self, symbols: _Symbols, module: str) -> None:
        self.symbols = symbols
        self.module = module
        self.blocks: dict[str, _Block] = {}
        self._by_signature: dict[Any, _Block] = {}
        self.root: _Member | None = None
        self.renamed: dict[str, str] = {}  # Linnet path -> PyTorch name, when different

    def describe(self, module: nn.Module) -> _Member | None:
        """The member tree of `module`, or None when it holds no tensors."""
        signature, member = self._describe(module)
        if member is None:
            return None
        member.block = self._block_for(type(module).__name__, signature, member)
        self.root = member
        return member

    def _describe(self, module: nn.Module) -> tuple[Any, _Member | None]:
        entries: list[Any] = []
        children: dict[str, _Member] = {}
        for name, parameter in module.named_parameters(recurse=False):
            entries.append(("param", name, _dtype_name(parameter.dtype), tuple(parameter.shape)))
            children[name] = _Member("param", _identifier(name))
        for name, buffer in module.named_buffers(recurse=False):
            entries.append(("buffer", name, _dtype_name(buffer.dtype), tuple(buffer.shape)))
            children[name] = _Member("buffer", _identifier(name))
        described: list[tuple[str, Any, _Member, str]] = []
        for name, child in module.named_children():
            child_signature, child_member = self._describe(child)
            if child_member is not None:
                described.append((name, child_signature, child_member, type(child).__name__))
        is_uniform_sequence = (
            isinstance(module, nn.ModuleList | nn.Sequential)
            and len(described) > 0
            and all(name == str(i) for i, (name, _, _, _) in enumerate(described))
            and len({(signature, cls) for _, signature, _, cls in described}) == 1
        )
        if is_uniform_sequence:
            # The container itself is the array; its parent names it.
            _, signature, first, class_name = described[0]
            block = self._block_for(class_name, signature, first)
            for name, _, child, _ in described:
                child.block = block
                children[name] = child
            entries.append(("array", len(described), class_name, signature))
            return tuple(entries), _Member(
                "sub", "", length=len(described), element=block, children=children
            )
        for name, signature, child, class_name in described:
            if child.length is None:
                child.block = self._block_for(class_name, signature, child)
            child.name = _identifier(name)
            children[name] = child
            entries.append(("sub", name, class_name, signature))
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
        for attribute, child in member.children.items():
            block.members.append(self._member_json(attribute, child))
        return block

    def _member_json(self, attribute: str, child: _Member) -> dict[str, Any]:
        if child.kind != "sub":
            return {"name": child.name, "kind": child.kind, "type": None}  # type filled later
        if child.length is not None:
            assert child.element is not None
            element_type = self.block_type(child.element)
            return {
                "name": child.name,
                "kind": "sub",
                "type": {"kind": "array", "element": element_type, "length": child.length},
            }
        assert child.block is not None
        return {"name": child.name, "kind": "sub", "type": self.block_type(child.block)}

    def block_type(self, block: _Block) -> dict[str, Any]:
        return {"kind": "block", "name": block.name, "module": self.module, "args": []}

    def member_type(self, member: _Member) -> dict[str, Any]:
        """The plan type of a `sub` member."""
        if member.length is not None:
            assert member.element is not None
            return {
                "kind": "array",
                "element": self.block_type(member.element),
                "length": member.length,
            }
        assert member.block is not None
        return self.block_type(member.block)


# ----------------------------------------------------------------- export


@dataclass
class ExportResult:
    """What `export_linnet` produced."""

    source: Path
    module: str
    root: str
    entry: str
    weights: Path | None
    bindings: Path | None
    plan: dict[str, Any]


def export_linnet(
    model: nn.Module,
    example_args: tuple[Any, ...],
    *,
    output: str | Path,
    dynamic_shapes: Any = None,
    module_name: str | None = None,
    weights: str | Path | None = None,
    std_root: str | Path | None = None,
) -> ExportResult:
    """Writes `model`'s architecture as Linnet source at `output`.

    `example_args` and `dynamic_shapes` are passed to `torch.export.export`;
    dimensions marked dynamic become generic parameters of the entry, named
    after their `torch.export.Dim`. With `weights`, the model's parameters and
    buffers are also saved there as SafeTensors under their PyTorch names,
    plus a `bindings.json` beside them when a Linnet parameter path had to
    differ from the PyTorch name. The generated source is checked with the
    compiler before it is written.
    """
    from torch.export import export

    output_path = Path(output)
    module = module_name or _identifier(output_path.stem)
    exported: Any = export(model, example_args, dynamic_shapes=dynamic_shapes)
    program = exported.run_decompositions()
    exporter = _Exporter(model, program, module, dynamic_shapes)
    plan = exporter.run()

    compiler = find_compiler()
    emitted = subprocess.run(
        [compiler, "emit", "-"],
        input=json.dumps(plan),
        capture_output=True,
        text=True,
        check=False,
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
        from safetensors.torch import save_file  # type: ignore[import-untyped]

        weights_dir = Path(weights)
        weights_dir.mkdir(parents=True, exist_ok=True)
        tensors = {
            name: tensor.detach().contiguous().cpu() for name, tensor in exporter.tensors().items()
        }
        weights_path = weights_dir / "model.safetensors"
        save_file(tensors, str(weights_path))
        if exporter.hierarchy.renamed:
            bindings_path = weights_dir / "bindings.json"
            bindings_path.write_text(
                json.dumps(exporter.hierarchy.renamed, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return ExportResult(
        output_path, module, exporter.root_name, "forward", weights_path, bindings_path, plan
    )


class _Exporter:
    def __init__(self, model: nn.Module, program: Any, module: str, dynamic_shapes: Any) -> None:
        self.model = model
        self.program = program
        self.module = module
        self.dynamic_shapes = dynamic_shapes
        self.symbols = _Symbols()
        self.builder = _Builder(self.symbols)
        self.hierarchy = _Hierarchy(self.symbols, module)
        self.self_value = self.builder.fresh(torch.float32, None)
        self.values: dict[str, _Value] = {}  # graph node name -> IR value
        self.dims: dict[str, Dim] = {}  # graph node name -> symbolic integer
        self.tuples: dict[str, list[_Value]] = {}  # graph node name -> tuple elements
        self.unsupported: list[str] = []
        self.root_name = ""
        self.imports: set[str] = set()

    # ---- driver

    def run(self) -> dict[str, Any]:
        root = self.hierarchy.describe(self.model)
        if root is None or root.block is None:
            raise ExportError("the model has no parameters or buffers to export")
        self.root_name = root.block.name
        self.self_value.type_json = self.hierarchy.block_type(root.block)
        self._fill_member_types(root)
        signature = self.program.graph_signature
        graph = self.program.graph

        inputs: list[tuple[str, _Value]] = []
        tensor_inputs = {
            spec.arg.name: spec for spec in signature.input_specs if spec.kind.name == "USER_INPUT"
        }
        self._name_symbols(graph, tensor_inputs)
        for node in graph.nodes:
            if node.op != "placeholder":
                continue
            spec = signature.inputs_to_parameters.get(node.name)
            buffer = signature.inputs_to_buffers.get(node.name)
            constant = signature.inputs_to_lifted_tensor_constants.get(node.name)
            if spec is not None or buffer is not None or constant is not None:
                self.values[node.name] = self._member_value(spec or buffer or constant, node)
            elif node.name in tensor_inputs:
                value = self._input_value(node)
                inputs.append((node.name, value))
                self.values[node.name] = value
            else:
                self.unsupported.append(f"input `{node.name}` of an unsupported kind")
        outputs: list[_Value] = []
        for node in graph.nodes:
            if node.op == "call_function":
                self._translate(node)
            elif node.op == "output" and not self.unsupported:
                outputs = [self.value_of(arg) for arg in _flatten(node.args[0])]
        if self.unsupported:
            raise ExportError(
                "the model uses operations without a Linnet mapping:\n  "
                + "\n  ".join(sorted(set(self.unsupported)))
            )
        if len(outputs) == 1:
            result = outputs[0]
        else:
            tuple_type = {"kind": "tuple", "elements": [self.builder.type_of(v) for v in outputs]}
            result = self.builder.op(
                "tuple.make", outputs, (torch.float32, None), type_json=tuple_type
            )
        self.builder.op("return", [result], None)
        body_ops = self.builder.regions[0]
        result_type = self.builder.type_of(result)

        generics = [
            {"name": self.symbols.names[s], "kind": "dim", "sym": i}
            for s, i in self.symbols.ids.items()
        ]
        body = {
            "args": [self.builder.value_json(self.self_value, "self")]
            + [self.builder.value_json(v, _identifier(n)) for n, v in inputs],
            "ops": body_ops,
        }
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
        return {
            "version": 1,
            "module": self.module,
            "root": {"name": self.root_name, "generics": [], "constraints": []},
            "manifest": [],
            "blocks": blocks,
            "functions": [
                {
                    "name": f"{self.module}::{self.root_name}.forward",
                    "kind": "entry",
                    "block": self.root_name,
                    "pub": True,
                    "generics": generics,
                    "constraints": self._constraints(),
                    "results": [result_type],
                    "body": body,
                }
            ],
        }

    # ---- inputs and members

    def _name_symbols(self, graph: Any, tensor_inputs: dict[str, Any]) -> None:
        """Declares the symbolic dimensions of the inputs, named after the
        `torch.export.Dim`s of `dynamic_shapes` when they can be matched."""
        from torch.export import Dim as ExportDim

        specs_by_name: dict[str, Any] = {}
        shapes: Any = self.dynamic_shapes
        if isinstance(shapes, Mapping):
            specs_by_name = dict(cast(Mapping[str, Any], shapes))
        elif isinstance(shapes, Sequence):
            specs_by_name = dict(
                zip(list(tensor_inputs), cast(Sequence[Any], shapes), strict=False)
            )
        for node in graph.nodes:
            if node.op != "placeholder" or node.name not in tensor_inputs:
                continue
            value = node.meta.get("val")
            if not isinstance(value, torch.Tensor):
                continue
            spec = specs_by_name.get(node.name)
            for axis, dim in enumerate(value.shape):
                if not isinstance(dim, torch.SymInt):
                    continue
                expr = dim.node.expr
                name: str | None = None
                axis_spec = None
                if isinstance(spec, Mapping):
                    axis_spec = cast(Mapping[int, Any], spec).get(axis)
                elif isinstance(spec, Sequence) and axis < len(cast(Sequence[Any], spec)):
                    axis_spec = cast(Sequence[Any], spec)[axis]
                if axis_spec is not None and isinstance(axis_spec, ExportDim):
                    name = _identifier(str(getattr(axis_spec, "__name__", "")) or "D")
                if expr.is_Symbol:
                    self.symbols.declare(expr, name)

    def _constraints(self) -> list[dict[str, Any]]:
        constraints: list[dict[str, Any]] = []
        for symbol, value_range in self.program.range_constraints.items():
            if symbol not in self.symbols.ids:
                continue
            lower, upper = value_range.lower, value_range.upper
            sym = {"sym": self.symbols.ids[symbol], "name": self.symbols.names[symbol]}
            if isinstance(lower, int) and lower > 0:
                constraints.append({"relation": ">=", "lhs": sym, "rhs": int(lower)})
            if isinstance(upper, int) and upper < 2**62:
                constraints.append({"relation": "<=", "lhs": sym, "rhs": int(upper)})
        return constraints

    def _fill_member_types(self, root: _Member) -> None:
        """Parameter and buffer members get their tensor types from the model."""
        for path, tensor in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            member, linnet_path = self._walk(root, path)
            if linnet_path != path:
                self.hierarchy.renamed[linnet_path] = path
            assert member.block is not None
            for entry in member.block.members:
                if entry["name"] == self._leaf_name(root, path):
                    entry["type"] = self.builder.tensor_type(tuple(tensor.shape), tensor.dtype)
        # Lifted tensor constants become buffers of the root block.
        for target in self.program.graph_signature.inputs_to_lifted_tensor_constants.values():
            tensor = self.program.constants[target]
            member_name = _identifier(target)
            assert root.block is not None
            root.block.members.append(
                {
                    "name": member_name,
                    "kind": "buffer",
                    "type": self.builder.tensor_type(tuple(tensor.shape), tensor.dtype),
                }
            )
            root.children[target] = _Member("buffer", member_name)
            if member_name != target:
                self.hierarchy.renamed[member_name] = target

    def _walk(self, root: _Member, path: str) -> tuple[_Member, str]:
        """The member owning the tensor at PyTorch `path` and the Linnet path."""
        member = root
        parts: list[str] = []
        for part in path.split(".")[:-1]:
            child = member.children[part]
            if member.length is not None:
                parts.append(part)  # array index, spelled the same way
            else:
                parts.append(child.name)
            member = child
        leaf = member.children[path.split(".")[-1]]
        parts.append(leaf.name)
        return member, ".".join(parts)

    @staticmethod
    def _leaf_name(root: _Member, path: str) -> str:
        member = root
        for part in path.split(".")[:-1]:
            member = member.children[part]
        return member.children[path.split(".")[-1]].name

    def _member_value(self, path: str, node: Any) -> _Value:
        """Loads a parameter, buffer, or constant through its member chain."""
        root = self.hierarchy.root
        assert root is not None
        tensor = node.meta["val"]
        member = root
        current = self.self_value
        parts = path.split(".")
        for i, part in enumerate(parts):
            child = member.children[part]
            is_last = i == len(parts) - 1
            if member.length is not None:
                # An element of a sub array, indexed by position.
                index = self.builder.const(int(part), torch.int64)
                current = self.builder.op(
                    "array.get",
                    [current, index],
                    (torch.float32, None),
                    type_json=self.hierarchy.block_type(cast(_Block, member.element)),
                )
                member = child
                continue
            if is_last:
                if child.kind == "sub":
                    raise ExportError(f"`{path}` names a block, not a tensor")
                # Buffers load exactly like parameters.
                return self.builder.op(
                    "block.param",
                    [current],
                    (tensor.dtype, tuple(tensor.shape)),
                    {"name": child.name},
                    name=node.name,
                )
            current = self.builder.op(
                "block.sub",
                [current],
                (torch.float32, None),
                {"name": child.name},
                type_json=self.hierarchy.member_type(child),
            )
            member = child
        raise ExportError(f"cannot resolve `{path}`")

    def _input_value(self, node: Any) -> _Value:
        tensor = node.meta.get("val")
        if not isinstance(tensor, torch.Tensor):
            raise ExportError(f"input `{node.name}` is not a tensor")
        if tensor.dim() == 0:
            raise ExportError(f"input `{node.name}` is a rank-0 tensor; Linnet inputs are tensors")
        return self.builder.fresh(tensor.dtype, tuple(tensor.shape))

    # ---- translation

    def value_of(self, arg: Any) -> _Value:
        """The IR value of a graph argument: a node's value, or a constant."""
        if isinstance(arg, torch.fx.Node):
            if arg.name in self.values:
                return self.values[arg.name]
            if arg.name in self.dims:
                return self.builder.const_dim(self.dims[arg.name])
            raise ExportError(f"`{arg.name}` has no value")
        raise ExportError(f"unexpected argument {arg!r}")

    def dim_of(self, arg: Any) -> Dim:
        if isinstance(arg, torch.fx.Node):
            if arg.name in self.dims:
                return self.dims[arg.name]
            raise ExportError(f"`{arg.name}` is not a compile-time integer")
        if isinstance(arg, torch.SymInt):
            return arg.node.expr
        if isinstance(arg, int):
            return arg
        raise ExportError(f"{arg!r} is not a dimension")

    def scalar(self, arg: Any, dtype: torch.dtype) -> _Value:
        """A Python number as a constant of `dtype`, or a node's value."""
        if isinstance(arg, torch.fx.Node):
            return self.value_of(arg)
        if isinstance(arg, torch.SymInt):
            return self.builder.const_dim(arg.node.expr)
        if isinstance(arg, bool | int | float):
            return self.builder.const(arg, dtype)
        raise ExportError(f"{arg!r} cannot be a scalar operand")

    def _result(self, node: Any) -> tuple[torch.dtype, tuple[Dim, ...] | None]:
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            shape = tuple(
                d.node.expr if isinstance(d, torch.SymInt) else int(d) for d in value.shape
            )
            return value.dtype, shape if value.dim() > 0 else None
        raise ExportError(f"`{node.name}` has no tensor result")

    def _translate(self, node: Any) -> None:
        target: Any = node.target
        name = str(getattr(target, "__name__", None) or target)
        # ATen overloads print as `aten.mm.default`; Python operators by name.
        overload = str(target) if str(target).startswith("aten.") else name
        for arg in _flatten(list(node.args) + list(node.kwargs.values())):
            if isinstance(arg, torch.fx.Node) and not self._is_known(arg):
                return  # downstream of an unsupported operation; already reported
        handler = _HANDLERS.get(overload) or _HANDLERS.get(overload.rsplit(".", 1)[0])
        if handler is None:
            # Python operators on symbolic integers.
            if target in _SYMBOLIC_OPERATORS:
                value = node.meta.get("val")
                if isinstance(value, torch.SymInt):
                    self.dims[node.name] = value.node.expr
                    return
                if isinstance(value, int):
                    self.dims[node.name] = value
                    return
            self.unsupported.append(f"{overload}")
            return
        try:
            value = handler(self, node)
        except ExportError as error:
            self.unsupported.append(f"{overload}: {error}")
            return
        if value is not None:
            self.values[node.name] = value

    def _is_known(self, node: Any) -> bool:
        return node.name in self.values or node.name in self.dims or node.name in self.tuples

    # ---- helpers used by handlers

    def tensor_arg(self, node: Any, index: int) -> _Value:
        return self.value_of(node.args[index])

    def result_of(self, node: Any) -> tuple[torch.dtype, tuple[Dim, ...] | None]:
        return self._result(node)

    def elementwise(
        self, node: Any, kind: str, *operands: _Value, attrs: dict[str, Any] | None = None
    ) -> _Value:
        return self.builder.op(kind, list(operands), self._result(node), attrs, name=node.name)

    def binary(self, node: Any, kind: str, attrs: dict[str, Any] | None = None) -> _Value:
        dtype, _ = self._result(node)
        a, b = node.args[0], node.args[1]
        left = self.scalar(a, dtype) if not isinstance(a, torch.fx.Node) else self.value_of(a)
        right = self.scalar(b, dtype) if not isinstance(b, torch.fx.Node) else self.value_of(b)
        left, right = self.unify(left, right, dtype)
        alpha = node.kwargs.get("alpha", 1)
        if alpha != 1:
            right = self.builder.op(
                "mul", [right, self.builder.const(alpha, dtype)], (right.dtype, right.shape)
            )
        return self.builder.op(kind, [left, right], self._result(node), attrs, name=node.name)

    def unify(self, left: _Value, right: _Value, dtype: torch.dtype) -> tuple[_Value, _Value]:
        """Casts operands to the result dtype; comparisons keep their own."""
        if left.dtype != right.dtype:
            common = (
                dtype if dtype != torch.bool else (left.dtype if left.is_tensor else right.dtype)
            )
            if left.dtype != common:
                left = self.builder.op("cast", [left], (common, left.shape))
            if right.dtype != common:
                right = self.builder.op("cast", [right], (common, right.shape))
        return left, right

    def reshape(self, value: _Value, shape: tuple[Dim, ...], name: str = "") -> _Value:
        if value.shape is not None and _same_dims(value.shape, shape):
            return value
        return self.builder.op(
            "reshape",
            [value],
            (value.dtype, shape),
            {"shape": [self.symbols.json(d) for d in shape]},
            name=name,
        )

    def call(
        self, node: Any, callee: str, generics: list[dict[str, Any]], operands: list[_Value]
    ) -> _Value:
        module_path = callee.split("::")[0]
        self.imports.add(module_path)
        return self.builder.op(
            "semantic.call",
            operands,
            self._result(node),
            {
                "callee": callee,
                "substitution": {"dims": {}, "packs": {}, "dtypes": {}},
                "generics": generics,
            },
            name=node.name,
        )

    def tensors(self) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for name, tensor in self.model.named_parameters():
            tensors[name] = tensor
        for name, tensor in self.model.named_buffers():
            tensors[name] = tensor
        for target in self.program.graph_signature.inputs_to_lifted_tensor_constants.values():
            tensors[target] = self.program.constants[target]
        return tensors


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list | tuple):
        return [item for element in cast(Sequence[Any], value) for item in _flatten(element)]
    return [value]


_SYMBOLIC_OPERATORS: set[Any] = {
    operator.mul,
    operator.add,
    operator.sub,
    getattr(operator, "floordiv"),  # noqa: B009  # its stub leaves the operands untyped
    operator.mod,
    operator.neg,
    operator.pos,
    torch.sym_max,
    torch.sym_min,
}

# ----------------------------------------------------------------- handlers

Handler = Callable[[_Exporter, Any], _Value | None]
_HANDLERS: dict[str, Handler] = {}


def _handles(*names: str) -> Callable[[Handler], Handler]:
    def register(handler: Handler) -> Handler:
        for name in names:
            _HANDLERS[name] = handler
        return handler

    return register


def _dim_json(exporter: _Exporter, dim: Dim) -> Any:
    return exporter.symbols.json(dim)


def _generic_dim(exporter: _Exporter, dim: Dim) -> dict[str, Any]:
    return {"dim": _dim_json(exporter, dim)}


def _generic_shape(exporter: _Exporter, shape: Sequence[Dim]) -> dict[str, Any]:
    return {"shape": [_dim_json(exporter, d) for d in shape]}


def _generic_dtype(dtype: torch.dtype) -> dict[str, Any]:
    return {"dtype": _dtype_name(dtype)}


@_handles("aten.sym_size.int")
def _sym_size(exporter: _Exporter, node: Any) -> _Value | None:
    value = node.meta.get("val")
    exporter.dims[node.name] = value.node.expr if isinstance(value, torch.SymInt) else int(value)
    return None


@_handles(
    "aten.view.default", "aten._unsafe_view.default", "aten.reshape.default",
    "aten.unsqueeze.default", "aten.squeeze.dim", "aten.squeeze.dims", "aten.squeeze.default",
    "aten.flatten.using_ints",
)  # fmt: skip
def _view(exporter: _Exporter, node: Any) -> _Value:
    _, shape = exporter.result_of(node)
    if shape is None:
        raise ExportError("rank-0 results are not supported")
    return exporter.reshape(exporter.tensor_arg(node, 0), shape, node.name)


@_handles(
    "aten.clone.default", "aten.alias.default", "aten.detach.default", "aten.contiguous.default",
    "aten.lift_fresh_copy.default", "aten.detach_.default",
)  # fmt: skip
def _alias(exporter: _Exporter, node: Any) -> _Value:
    return exporter.tensor_arg(node, 0)


@_handles("aten._to_copy.default", "aten.to.dtype", "aten.to.dtype_layout")
def _to_copy(exporter: _Exporter, node: Any) -> _Value:
    source = exporter.tensor_arg(node, 0)
    dtype, shape = exporter.result_of(node)
    if dtype == source.dtype:
        return source
    return exporter.builder.op("cast", [source], (dtype, shape), name=node.name)


@_handles("aten.permute.default")
def _permute(exporter: _Exporter, node: Any) -> _Value:
    source = exporter.tensor_arg(node, 0)
    axes = [int(a) for a in node.args[1]]
    if axes == list(range(len(axes))):
        return source
    return exporter.builder.op(
        "permute", [source], exporter.result_of(node), {"shape": axes}, name=node.name
    )


@_handles("aten.transpose.int")
def _transpose(exporter: _Exporter, node: Any) -> _Value:
    source = exporter.tensor_arg(node, 0)
    rank = len(source.shape or ())
    a, b = (int(node.args[1]) % rank, int(node.args[2]) % rank)
    axes = list(range(rank))
    axes[a], axes[b] = axes[b], axes[a]
    return exporter.builder.op(
        "permute", [source], exporter.result_of(node), {"shape": axes}, name=node.name
    )


@_handles("aten.t.default")
def _t(exporter: _Exporter, node: Any) -> _Value:
    source = exporter.tensor_arg(node, 0)
    if len(source.shape or ()) < 2:
        return source
    return exporter.builder.op(
        "permute", [source], exporter.result_of(node), {"shape": [1, 0]}, name=node.name
    )


@_handles("aten.expand.default", "aten.expand_copy.default", "aten.broadcast_to.default")
def _expand(exporter: _Exporter, node: Any) -> _Value:
    source = exporter.tensor_arg(node, 0)
    dtype, shape = exporter.result_of(node)
    if shape is None or (source.shape is not None and _same_dims(source.shape, shape)):
        return source
    return exporter.builder.op(
        "broadcast",
        [source],
        (dtype, shape),
        {"shape": [_dim_json(exporter, d) for d in shape]},
        name=node.name,
    )


_BINARY = {
    "aten.add": "add",
    "aten.sub": "sub",
    "aten.mul": "mul",
    "aten.div": "div",
    "aten.maximum": "max",
    "aten.minimum": "min",
}


@_handles(
    *[
        f"{name}.{variant}"
        for name in _BINARY
        for variant in ("Tensor", "Scalar", "default", "Tensor_mode")
    ]
)
def _binary(exporter: _Exporter, node: Any) -> _Value:
    base = str(node.target).rsplit(".", 1)[0]
    if node.kwargs.get("rounding_mode") not in (None, "trunc") and base == "aten.div":
        raise ExportError("floor division of tensors is not supported")
    return exporter.binary(node, _BINARY[base])


@_handles("aten.logical_and.default", "aten.bitwise_and.Tensor", "aten.logical_or.default",
          "aten.bitwise_or.Tensor")  # fmt: skip
def _logical(exporter: _Exporter, node: Any) -> _Value:
    """Boolean tensors combine through `select`; `&&` and `||` are scalar."""
    dtype, shape = exporter.result_of(node)
    if dtype != torch.bool:
        raise ExportError("bitwise operations on integers are not supported")
    left = exporter.tensor_arg(node, 0)
    right = exporter.scalar(node.args[1], torch.bool)
    is_and = "and" in str(node.target)
    if shape is None:
        return exporter.builder.op(
            "and" if is_and else "or", [left, right], (dtype, None), name=node.name
        )
    constant = exporter.builder.const(not is_and, torch.bool)
    operands = [left, right, constant] if is_and else [left, constant, right]
    return exporter.builder.op("select", operands, (dtype, shape), name=node.name)


@_handles("aten.logical_not.default", "aten.bitwise_not.default")
def _logical_not(exporter: _Exporter, node: Any) -> _Value:
    dtype, shape = exporter.result_of(node)
    if dtype != torch.bool:
        raise ExportError("bitwise not on integers is not supported")
    x = exporter.tensor_arg(node, 0)
    if shape is None:
        return exporter.builder.op("not", [x], (dtype, None), name=node.name)
    return exporter.builder.op(
        "select",
        [x, exporter.builder.const(False, torch.bool), exporter.builder.const(True, torch.bool)],
        (dtype, shape),
        name=node.name,
    )


@_handles("aten.rsub.Scalar", "aten.rsub.Tensor")
def _rsub(exporter: _Exporter, node: Any) -> _Value:
    dtype, _ = exporter.result_of(node)
    left = exporter.scalar(node.args[1], dtype)
    right = exporter.tensor_arg(node, 0)
    return exporter.builder.op("sub", [left, right], exporter.result_of(node), name=node.name)


_COMPARE = {
    "aten.eq": "eq",
    "aten.ne": "ne",
    "aten.lt": "lt",
    "aten.le": "le",
    "aten.gt": "gt",
    "aten.ge": "ge",
}


@_handles(*[f"{name}.{variant}" for name in _COMPARE for variant in ("Tensor", "Scalar")])
def _compare(exporter: _Exporter, node: Any) -> _Value:
    base = str(node.target).rsplit(".", 1)[0]
    left = exporter.tensor_arg(node, 0)
    right = exporter.scalar(node.args[1], left.dtype)
    left, right = exporter.unify(left, right, left.dtype)
    return exporter.builder.op(
        "compare",
        [left, right],
        exporter.result_of(node),
        {"compare": _COMPARE[base]},
        name=node.name,
    )


_UNARY = {
    "aten.exp.default": "exp",
    "aten.log.default": "log",
    "aten.sqrt.default": "sqrt",
    "aten.rsqrt.default": "rsqrt",
    "aten.sin.default": "sin",
    "aten.cos.default": "cos",
    "aten.tanh.default": "tanh",
    "aten.abs.default": "abs",
    "aten.neg.default": "neg",
}


@_handles(*_UNARY)
def _unary(exporter: _Exporter, node: Any) -> _Value:
    return exporter.elementwise(node, _UNARY[str(node.target)], exporter.tensor_arg(node, 0))


@_handles("aten.reciprocal.default")
def _reciprocal(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    one = exporter.builder.const(1, x.dtype)
    return exporter.builder.op("div", [one, x], exporter.result_of(node), name=node.name)


@_handles("aten.square.default")
def _square(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    return exporter.builder.op("mul", [x, x], exporter.result_of(node), name=node.name)


@_handles("aten.pow.Tensor_Scalar")
def _pow(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    exponent = node.args[1]
    result = exporter.result_of(node)
    if exponent == 2:
        return exporter.builder.op("mul", [x, x], result, name=node.name)
    if exponent == 0.5:
        return exporter.builder.op("sqrt", [x], result, name=node.name)
    if exponent == -0.5:
        return exporter.builder.op("rsqrt", [x], result, name=node.name)
    if exponent == 1:
        return x
    if exponent == -1:
        return exporter.builder.op(
            "div", [exporter.builder.const(1, x.dtype), x], result, name=node.name
        )
    if exponent == 3:
        squared = exporter.builder.op("mul", [x, x], result)
        return exporter.builder.op("mul", [squared, x], result, name=node.name)
    raise ExportError(f"pow with exponent {exponent} has no primitive form")


@_handles("aten.sigmoid.default")
def _sigmoid(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    shape = x.shape or ()
    return exporter.call(
        node,
        "std.nn.activations::sigmoid",
        [_generic_shape(exporter, shape), _generic_dtype(x.dtype)],
        [x],
    )


@_handles("aten.silu.default")
def _silu(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    return exporter.call(
        node,
        "std.nn.activations::silu",
        [_generic_shape(exporter, x.shape or ()), _generic_dtype(x.dtype)],
        [x],
    )


@_handles("aten.relu.default")
def _relu(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    return exporter.call(
        node,
        "std.nn.activations::relu",
        [_generic_shape(exporter, x.shape or ()), _generic_dtype(x.dtype)],
        [x],
    )


@_handles("aten.gelu.default")
def _gelu(exporter: _Exporter, node: Any) -> _Value:
    tanh = node.kwargs.get("approximate", "none") == "tanh"
    x = exporter.tensor_arg(node, 0)
    return exporter.call(
        node,
        "std.nn.activations::gelu" if tanh else "std.nn.activations::gelu_erf",
        [_generic_shape(exporter, x.shape or ()), _generic_dtype(x.dtype)],
        [x],
    )


@_handles("aten.where.self")
def _where(exporter: _Exporter, node: Any) -> _Value:
    dtype, _ = exporter.result_of(node)
    condition = exporter.tensor_arg(node, 0)
    a = exporter.scalar(node.args[1], dtype)
    b = exporter.scalar(node.args[2], dtype)
    return exporter.builder.op(
        "select", [condition, a, b], exporter.result_of(node), name=node.name
    )


@_handles(
    "aten.full.default", "aten.full_like.default", "aten.zeros.default", "aten.ones.default",
    "aten.zeros_like.default", "aten.ones_like.default", "aten.empty.memory_format",
    "aten.new_zeros.default", "aten.new_ones.default", "aten.scalar_tensor.default",
)  # fmt: skip
def _fill(exporter: _Exporter, node: Any) -> _Value:
    dtype, shape = exporter.result_of(node)
    target = str(node.target)
    if "full" in target:
        fill_value = node.args[1]
    elif "zeros" in target or "empty" in target:
        fill_value = 0
    elif "ones" in target:
        fill_value = 1
    else:  # scalar_tensor
        fill_value = node.args[0]
    if isinstance(fill_value, torch.fx.Node):
        raise ExportError("fill values must be constants")
    value = exporter.builder.const(fill_value, dtype)
    if shape is None:
        return value
    return exporter.builder.op(
        "fill",
        [value],
        (dtype, shape),
        {"shape": [_dim_json(exporter, d) for d in shape]},
        name=node.name,
    )


@_handles("aten.arange.default", "aten.arange.start", "aten.arange.start_step")
def _arange(exporter: _Exporter, node: Any) -> _Value:
    dtype, shape = exporter.result_of(node)
    if shape is None or len(shape) != 1:
        raise ExportError("arange must produce a vector")
    target = str(node.target)
    start = 0 if target.endswith("default") else exporter.dim_of(node.args[0])
    step = (
        exporter.dim_of(node.args[2]) if target.endswith("start_step") and len(node.args) > 2 else 1
    )
    positions = exporter.builder.op(
        "iota", [], (torch.int64, shape), {"shape": [_dim_json(exporter, shape[0])]}
    )
    if step != 1:
        positions = exporter.builder.op(
            "mul", [positions, exporter.builder.const_dim(step)], (torch.int64, shape)
        )
    if start != 0:
        positions = exporter.builder.op(
            "add", [positions, exporter.builder.const_dim(start)], (torch.int64, shape)
        )
    if dtype != torch.int64:
        positions = exporter.builder.op("cast", [positions], (dtype, shape))
    exporter.builder.regions[-1][-1]["results"][0]["name"] = node.name
    return positions


@_handles("aten.mm.default")
def _mm(exporter: _Exporter, node: Any) -> _Value:
    a = exporter.tensor_arg(node, 0)
    b = exporter.tensor_arg(node, 1)
    m, k = cast(tuple[Dim, Dim], a.shape)
    n = cast(tuple[Dim, Dim], b.shape)[1]
    return exporter.call(
        node,
        "std.linalg::matmul",
        [
            _generic_dim(exporter, m),
            _generic_dim(exporter, k),
            _generic_dim(exporter, n),
            _generic_dtype(a.dtype),
        ],
        [a, b],
    )


@_handles("aten.bmm.default")
def _bmm(exporter: _Exporter, node: Any) -> _Value:
    a = exporter.tensor_arg(node, 0)
    b = exporter.tensor_arg(node, 1)
    batch, m, k = cast(tuple[Dim, Dim, Dim], a.shape)
    n = cast(tuple[Dim, Dim, Dim], b.shape)[2]
    return exporter.call(
        node,
        "std.linalg::batched_matmul",
        [
            _generic_shape(exporter, [batch]),
            _generic_dim(exporter, m),
            _generic_dim(exporter, k),
            _generic_dim(exporter, n),
            _generic_dtype(a.dtype),
        ],
        [a, b],
    )


@_handles("aten.addmm.default")
def _addmm(exporter: _Exporter, node: Any) -> _Value:
    bias = exporter.tensor_arg(node, 0)
    a = exporter.tensor_arg(node, 1)
    b = exporter.tensor_arg(node, 2)
    m, k = cast(tuple[Dim, Dim], a.shape)
    n = cast(tuple[Dim, Dim], b.shape)[1]
    product = exporter.builder.op(
        "semantic.call",
        [a, b],
        (a.dtype, (m, n)),
        {
            "callee": "std.linalg::matmul",
            "substitution": {"dims": {}, "packs": {}, "dtypes": {}},
            "generics": [
                _generic_dim(exporter, m),
                _generic_dim(exporter, k),
                _generic_dim(exporter, n),
                _generic_dtype(a.dtype),
            ],
        },
    )
    exporter.imports.add("std.linalg")
    return exporter.builder.op("add", [product, bias], exporter.result_of(node), name=node.name)


@_handles("aten.linear.default")
def _linear(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    weight = exporter.tensor_arg(node, 1)
    out, inner = cast(tuple[Dim, Dim], weight.shape)
    leading = list(x.shape or ())[:-1]
    optional_type = {"kind": "optional", "inner": exporter.builder.tensor_type((out,), x.dtype)}
    if len(node.args) > 2 and node.args[2] is not None:
        bias = exporter.builder.op(
            "option.some",
            [exporter.tensor_arg(node, 2)],
            (x.dtype, (out,)),
            type_json=optional_type,
        )
    else:
        bias = exporter.builder.op("option.none", [], (x.dtype, (out,)), type_json=optional_type)
    return exporter.call(
        node,
        "std.nn.linear::linear",
        [
            _generic_shape(exporter, leading),
            _generic_dim(exporter, inner),
            _generic_dim(exporter, out),
            _generic_dtype(x.dtype),
        ],
        [x, weight, bias],
    )


@_handles("aten._softmax.default", "aten.softmax.int")
def _softmax(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    shape = list(x.shape or ())
    axis = int(node.args[1]) % len(shape)
    if axis != len(shape) - 1:
        raise ExportError("softmax over an axis other than the last is not supported")
    return exporter.call(
        node,
        "std.nn.softmax::softmax",
        [
            _generic_shape(exporter, shape[:-1]),
            _generic_dim(exporter, shape[-1]),
            _generic_dtype(x.dtype),
        ],
        [x],
    )


def _reduction(exporter: _Exporter, node: Any, kind: str, divide: bool) -> _Value:
    x = exporter.tensor_arg(node, 0)
    shape = list(x.shape or ())
    rank = len(shape)
    dims_arg = node.args[1] if len(node.args) > 1 else None
    raw: list[Any] = (
        list(cast(Sequence[Any], dims_arg)) if isinstance(dims_arg, list | tuple) else [dims_arg]
    )
    if dims_arg is None or not raw:
        reduced = list(range(rank))
    else:
        reduced = sorted(int(d) % rank for d in raw)
    keepdim = bool(node.args[2]) if len(node.args) > 2 else bool(node.kwargs.get("keepdim", False))
    dtype, result_shape = exporter.result_of(node)
    kept = [i for i in range(rank) if i not in reduced]
    accumulate = dtype

    def body(outputs: list[_Value]) -> _Value:
        def inner(inner_indices: list[_Value]) -> _Value:
            indices: list[_Value] = []
            outer_iter = iter(outputs)
            inner_iter = iter(inner_indices)
            for axis in range(rank):
                indices.append(next(inner_iter) if axis in reduced else next(outer_iter))
            element = exporter.builder.element(x, indices)
            if element.dtype != accumulate:
                element = exporter.builder.op("cast", [element], (accumulate, None))
            return element

        total = exporter.builder.reduce(
            kind, [(f"r{axis}", shape[axis]) for axis in reduced], accumulate, inner
        )
        if divide:
            count: Dim = 1
            for axis in reduced:
                count = count * shape[axis]
            divisor = exporter.builder.op(
                "cast", [exporter.builder.const_dim(count)], (accumulate, None)
            )
            total = exporter.builder.op("div", [total, divisor], (accumulate, None))
        return total

    if not kept:
        # Reducing everything gives a scalar; keepdim reshapes it into a
        # rank-`rank` tensor of ones, which Linnet cannot fill from a scalar
        # without a comprehension over nothing — express it as fill.
        value = body([])
        if result_shape is None:
            return value
        return exporter.builder.op(
            "fill",
            [value],
            (dtype, result_shape),
            {"shape": [_dim_json(exporter, d) for d in result_shape]},
            name=node.name,
        )
    result = exporter.builder.comprehension(
        [(f"o{axis}", shape[axis]) for axis in kept], dtype, body,
        name="" if keepdim else node.name,
    )  # fmt: skip
    if keepdim and result_shape is not None:
        return exporter.reshape(result, result_shape, node.name)
    return result


@_handles("aten.sum.dim_IntList", "aten.sum.default")
def _sum(exporter: _Exporter, node: Any) -> _Value:
    return _reduction(exporter, node, "sum", divide=False)


@_handles("aten.mean.dim", "aten.mean.default")
def _mean(exporter: _Exporter, node: Any) -> _Value:
    return _reduction(exporter, node, "sum", divide=True)


@_handles("aten.amax.default")
def _amax(exporter: _Exporter, node: Any) -> _Value:
    return _reduction(exporter, node, "max", divide=False)


@_handles("aten.amin.default")
def _amin(exporter: _Exporter, node: Any) -> _Value:
    return _reduction(exporter, node, "min", divide=False)


@_handles("aten.embedding.default")
def _embedding(exporter: _Exporter, node: Any) -> _Value:
    table = exporter.tensor_arg(node, 0)
    ids = exporter.tensor_arg(node, 1)
    dtype, shape = exporter.result_of(node)
    if shape is None:
        raise ExportError("embedding must produce a tensor")
    id_shape = list(ids.shape or ())

    def body(indices: list[_Value]) -> _Value:
        row = exporter.builder.element(ids, indices[:-1])
        if row.dtype != torch.int64:
            row = exporter.builder.op("cast", [row], (torch.int64, None))
        return exporter.builder.element(table, [row, indices[-1]])

    return exporter.builder.comprehension(
        [(f"i{axis}", d) for axis, d in enumerate(id_shape)] + [("h", shape[-1])],
        dtype,
        body,
        name=node.name,
    )


@_handles("aten.slice.Tensor")
def _slice(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    shape = list(x.shape or ())
    axis = int(node.args[1]) % len(shape)
    start = exporter.dim_of(node.args[2]) if len(node.args) > 2 and node.args[2] is not None else 0
    stop = (
        exporter.dim_of(node.args[3])
        if len(node.args) > 3 and node.args[3] is not None
        else shape[axis]
    )
    step = int(node.args[4]) if len(node.args) > 4 else 1
    if isinstance(stop, int) and stop >= 2**62:
        stop = shape[axis]
    dtype, result_shape = exporter.result_of(node)
    if result_shape is not None and _same_dims(result_shape, shape):
        return x
    axes: list[dict[str, Any]] = []
    for i, size in enumerate(shape):
        if i == axis:
            axes.append(
                {
                    "start": _dim_json(exporter, start),
                    "stop": _dim_json(exporter, stop),
                    "step": step,
                    "squeeze": False,
                }
            )
        else:
            axes.append(
                {"start": 0, "stop": _dim_json(exporter, size), "step": 1, "squeeze": False}
            )
    return exporter.builder.op("slice", [x], (dtype, result_shape), {"axes": axes}, name=node.name)


@_handles("aten.select.int")
def _select(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    shape = list(x.shape or ())
    axis = int(node.args[1]) % len(shape)
    index = exporter.dim_of(node.args[2])
    if isinstance(index, int) and index < 0:
        index = shape[axis] + index
    dtype, result_shape = exporter.result_of(node)
    if result_shape is None:
        raise ExportError("selecting from a vector gives a scalar, which slicing cannot express")
    axes: list[dict[str, Any]] = []
    for i, size in enumerate(shape):
        if i == axis:
            axes.append(
                {
                    "start": _dim_json(exporter, index),
                    "stop": _dim_json(exporter, index + 1),
                    "step": 1,
                    "squeeze": True,
                }
            )
        else:
            axes.append(
                {"start": 0, "stop": _dim_json(exporter, size), "step": 1, "squeeze": False}
            )
    return exporter.builder.op("slice", [x], (dtype, result_shape), {"axes": axes}, name=node.name)


@_handles("aten.cat.default")
def _cat(exporter: _Exporter, node: Any) -> _Value:
    parts = [exporter.value_of(arg) for arg in node.args[0]]
    rank = len(parts[0].shape or ())
    axis = int(node.args[1]) % rank if len(node.args) > 1 else 0
    return exporter.builder.op(
        "concat", parts, exporter.result_of(node), {"axis": axis}, name=node.name
    )


@_handles("aten.split.Tensor", "aten.split_with_sizes.default", "aten.unbind.int")
def _split(exporter: _Exporter, node: Any) -> _Value | None:
    x = exporter.tensor_arg(node, 0)
    shape = list(x.shape or ())
    target = str(node.target)
    values = node.meta.get("val")
    if not isinstance(values, list | tuple):
        raise ExportError("split must produce a tuple")
    if target.startswith("aten.split"):
        raw_axis = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
    else:
        raw_axis = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    axis = int(raw_axis) % len(shape)
    pieces: list[_Value] = []
    offset: Dim = 0
    for piece in cast(Sequence[Any], values):
        piece_shape = tuple(
            d.node.expr if isinstance(d, torch.SymInt) else int(d) for d in piece.shape
        )
        is_unbind = target.startswith("aten.unbind")
        size: Dim = 1 if is_unbind else piece_shape[axis]
        axes: list[dict[str, Any]] = []
        for i, dim in enumerate(shape):
            if i == axis:
                axes.append(
                    {
                        "start": _dim_json(exporter, offset),
                        "stop": _dim_json(exporter, offset + size),
                        "step": 1,
                        "squeeze": is_unbind,
                    }
                )
            else:
                axes.append(
                    {"start": 0, "stop": _dim_json(exporter, dim), "step": 1, "squeeze": False}
                )
        pieces.append(exporter.builder.op("slice", [x], (x.dtype, piece_shape), {"axes": axes}))
        offset = offset + size
    exporter.tuples[node.name] = pieces
    return None


@_handles("_operator.getitem", "getitem")
def _getitem(exporter: _Exporter, node: Any) -> _Value:
    source = node.args[0]
    if isinstance(source, torch.fx.Node) and source.name in exporter.tuples:
        element = exporter.tuples[source.name][int(node.args[1])]
        if element.id < 0:
            raise ExportError("this result of the operation is not available")
        return element
    raise ExportError("indexing into a value that is not a known tuple")


@_handles("aten.native_layer_norm.default", "aten.layer_norm.default")
def _layer_norm(exporter: _Exporter, node: Any) -> _Value | None:
    """`layer_norm` over the last axis is the standard library's op; the
    statistics PyTorch also returns are not offered."""
    x = exporter.tensor_arg(node, 0)
    shape = list(x.shape or ())
    normalized = [exporter.dim_of(d) for d in node.args[1]]
    if len(normalized) != 1 or normalized[0] != shape[-1]:
        raise ExportError("layer_norm over more than the last axis is not supported")
    weight_node, bias_node = node.args[2], node.args[3]
    eps = node.args[4] if len(node.args) > 4 else node.kwargs.get("eps", 1e-5)
    if weight_node is None:
        raise ExportError("layer_norm without an affine weight is not supported")
    weight = exporter.value_of(weight_node)
    optional_type = {
        "kind": "optional",
        "inner": exporter.builder.tensor_type((shape[-1],), x.dtype),
    }
    if bias_node is not None:
        bias = exporter.builder.op(
            "option.some",
            [exporter.value_of(bias_node)],
            (x.dtype, (shape[-1],)),
            type_json=optional_type,
        )
    else:
        bias = exporter.builder.op(
            "option.none", [], (x.dtype, (shape[-1],)), type_json=optional_type
        )
    epsilon = exporter.builder.const(float(eps), torch.float32)
    result_type = (x.dtype, tuple(shape))
    call = exporter.builder.op(
        "semantic.call",
        [x, weight, bias, epsilon],
        result_type,
        {
            "callee": "std.nn.norm::layer_norm",
            "substitution": {"dims": {}, "packs": {}, "dtypes": {}},
            "generics": [
                _generic_shape(exporter, shape[:-1]),
                _generic_dim(exporter, shape[-1]),
                _generic_dtype(x.dtype),
            ],
        },
        name=node.name,
    )
    exporter.imports.add("std.nn.norm")
    if str(node.target).startswith("aten.native_layer_norm"):
        unavailable = _Value(-1, torch.float32, None)
        exporter.tuples[node.name] = [call, unavailable, unavailable]
        return None
    return call


@_handles("aten.clamp.default", "aten.clamp_min.default", "aten.clamp_max.default")
def _clamp(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    target = str(node.target)
    result = exporter.result_of(node)
    low = node.args[1] if len(node.args) > 1 else None
    high = node.args[2] if len(node.args) > 2 else None
    if target.endswith("clamp_max.default"):
        low, high = None, node.args[1]
    value = x
    if low is not None:
        value = exporter.builder.op("max", [value, exporter.scalar(low, x.dtype)], result)
    if high is not None:
        value = exporter.builder.op("min", [value, exporter.scalar(high, x.dtype)], result)
    exporter.builder.regions[-1][-1]["results"][0]["name"] = node.name
    return value


@_handles("aten.tril.default")
def _tril(exporter: _Exporter, node: Any) -> _Value:
    x = exporter.tensor_arg(node, 0)
    dtype, shape = exporter.result_of(node)
    if shape is None or len(shape) < 2:
        raise ExportError("tril needs a matrix")
    diagonal = int(node.args[1]) if len(node.args) > 1 else 0
    rows, cols = shape[-2], shape[-1]
    row_positions = exporter.builder.op(
        "iota", [], (torch.int64, (rows,)), {"shape": [_dim_json(exporter, rows)]}
    )
    col_positions = exporter.builder.op(
        "iota", [], (torch.int64, (cols,)), {"shape": [_dim_json(exporter, cols)]}
    )
    rows_column = exporter.reshape(row_positions, (rows, 1))
    cols_row = exporter.reshape(col_positions, (1, cols))
    if diagonal != 0:
        rows_column = exporter.builder.op(
            "add",
            [rows_column, exporter.builder.const(diagonal, torch.int64)],
            (torch.int64, (rows, 1)),
        )
    keep = exporter.builder.op(
        "compare", [cols_row, rows_column], (torch.bool, (rows, cols)), {"compare": "le"}
    )
    zero = exporter.builder.const(0, dtype)
    return exporter.builder.op("select", [keep, x, zero], (dtype, shape), name=node.name)
