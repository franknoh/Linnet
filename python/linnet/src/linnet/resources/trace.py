"""From a compiled program to a `TensorGraph`.

`trace` walks one entry of a plan (`linnet.ir.Program`) the way a backend
runs it: calls are followed into their callees, `static_for` loops are
unrolled over the block array, and each library call either runs its native
implementation, as the plan's `selected` attribute records for the numerics
policy, or has its canonical body traced in its place. Every tensor value
becomes a memory object with a symbolic size; every operation that computes
one becomes a step.

What the backend does with an operation's storage is the `Lowering`'s
business, not the graph's: which operations are views of their operand, and
which native implementations update a state argument in place. The default
describes the generated PyTorch code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypedDict, cast

from .. import ir
from ..compiler import LinnetError
from . import expr as ex
from .graph import Category, MemoryObject, Step, TensorGraph
from .storage import tensor_bytes


class TraceError(LinnetError):
    """The program cannot be traced with the bindings given."""


# Arithmetic on every element: one floating-point (or integer) operation each.
ARITHMETIC = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "rem",
        "min",
        "max",
        "compare",
        "and",
        "or",
        "bitand",
        "bitor",
        "bitxor",
        "shl",
        "shr",
        "not",
        "neg",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "sin",
        "cos",
        "tanh",
        "abs",
        "select",
    }
)
CONSTANTS = frozenset({"const.int", "const.float", "const.bool", "const.dim"})


@dataclass(frozen=True, slots=True)
class Lowering:
    """What a backend does with operations, as far as storage goes.

    A library call runs the native implementation the plan selected for it,
    or its canonical body when `selected` is the canonical decomposition (or
    `inline_all` is set, which counting arithmetic uses). An implementation
    in `in_place_writes` updates its first operand in place when that operand
    is a state nothing else reads, as the generated PyTorch code does for
    cache writes. An implementation in `identities` returns its operand (a
    collective on one device). A `reshape` is a view of contiguous storage
    and a copy of anything else; `permute`, `broadcast` and `slice` are
    views."""

    name: str = "torch"
    in_place_writes: frozenset[str] = frozenset(
        {
            "torch.Tensor.index_copy",
            "torch.Tensor.index_put",
            "torch.Tensor.index_put(tokens)",
        }
    )
    inline_all: bool = False
    # Collectives over the processes of a tensor-parallel group: on one
    # device the value is its own sum and its own gather, returned as is.
    identities: frozenset[str] = frozenset(
        {"torch.distributed.all_reduce", "torch.distributed.all_gather", "torch.distributed.shared"}
    )

    # The memory order of a native result's axes where it is not their own:
    # fused attention writes [batch, positions, heads, width] and returns
    # it as [batch, heads, positions, width].
    layouts: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("torch.nn.functional.scaled_dot_product_attention", (0, 2, 1, 3)),
    )

    def layout(self, implementation: str) -> tuple[int, ...]:
        for prefix, order in self.layouts:
            if implementation.startswith(prefix):
                return order
        return ()

    def native(self, op: ir.Op) -> str | None:
        if op.kind != "semantic.call" or self.inline_all:
            return None
        selected = op.attrs.get("selected")
        if selected is None or selected == "canonical decomposition":
            return None
        return str(selected)


# ------------------------------------------------------------------- values


@dataclass(frozen=True, slots=True)
class SymEnv:
    """Bindings of generic parameters by the compiler's ids: dimensions as
    expressions, shape packs as sequences of them, dtypes by name."""

    dims: Mapping[int, ex.Expr] = field(default_factory=lambda: MappingProxyType({}))
    packs: Mapping[int, tuple[ex.Expr, ...]] = field(default_factory=lambda: MappingProxyType({}))
    dtypes: Mapping[int, str] = field(default_factory=lambda: MappingProxyType({}))

    def dim(self, dim: ir.Dim) -> ex.Expr:
        return ir.fold_dim(dim, self._symbol, self._pack_size, ex.EXPRESSIONS)

    def _symbol(self, symbol: ir.DimSymbol) -> ex.Expr:
        if symbol.id not in self.dims:
            raise TraceError(f"dimension `{symbol.name}` is not bound")
        return self.dims[symbol.id]

    def _pack_size(self, pack: ir.PackSize) -> ex.Expr:
        if pack.id not in self.packs:
            raise TraceError(f"shape pack `{pack.name}` is not bound")
        return ex.product(self.packs[pack.id])

    def shape(self, shape: ir.Shape) -> tuple[ex.Expr, ...]:
        out: list[ex.Expr] = []
        for unit in shape:
            if isinstance(unit, ir.Pack):
                if unit.id not in self.packs:
                    raise TraceError(f"shape pack `{unit.name}` is not bound")
                out.extend(self.packs[unit.id])
            else:
                out.append(self.dim(unit))
        return tuple(out)

    def dtype(self, dtype: ir.DType) -> str:
        if isinstance(dtype, str):
            return dtype
        if dtype.id not in self.dtypes:
            raise TraceError(f"dtype `{dtype.name}` is not bound")
        return self.dtypes[dtype.id]

    def bind(self, substitution: ir.Substitution, base: SymEnv | None = None) -> SymEnv:
        """The callee's bindings: `substitution` evaluated here, over `base`
        (a method's receiver bindings)."""
        dims = dict(base.dims) if base else {}
        packs = dict(base.packs) if base else {}
        dtypes = dict(base.dtypes) if base else {}
        dims.update({k: self.dim(v) for k, v in substitution.dims.items()})
        packs.update({k: self.shape(v) for k, v in substitution.packs.items()})
        dtypes.update({k: self.dtype(v) for k, v in substitution.dtypes.items()})
        return SymEnv(dims, packs, dtypes)

    def _with(
        self,
        generic: ir.Generic,
        arg: ir.GenericArg,
        dims: dict[int, ex.Expr],
        packs: dict[int, tuple[ex.Expr, ...]],
        dtypes: dict[int, str],
    ) -> SymEnv:
        if isinstance(arg, ir.DimArg):
            dims[generic.id] = self.dim(arg.dim)
        elif isinstance(arg, ir.ShapeArg):
            packs[generic.id] = self.shape(arg.shape)
        else:
            dtypes[generic.id] = self.dtype(arg.dtype)
        return SymEnv(dims, packs, dtypes)

    def instance(self, block: ir.Block, args: Sequence[ir.GenericArg]) -> SymEnv:
        """The bindings of a block instance whose generic arguments are
        `args`, written in these bindings' terms."""
        dims: dict[int, ex.Expr] = {}
        packs: dict[int, tuple[ex.Expr, ...]] = {}
        dtypes: dict[int, str] = {}
        for generic, arg in zip(block.generics, args, strict=False):
            self._with(generic, arg, dims, packs, dtypes)
        return SymEnv(dims, packs, dtypes)


@dataclass(frozen=True, slots=True)
class TensorValue:
    """A tensor, and the order its axes lie in memory, outermost first: `()`
    for its own order (contiguous), None for no order of its axes (a
    broadcast, a strided slice)."""

    id: int
    order: tuple[int, ...] | None = ()

    @property
    def contiguous(self) -> bool:
        return self.order is not None and list(self.order) == sorted(self.order)


@dataclass(frozen=True, slots=True)
class ScalarValue:
    """A scalar; `value` when it is a compile-time integer."""

    value: ex.Expr | None = None


@dataclass(frozen=True, slots=True)
class BlockRef:
    path: str
    block: str
    env: SymEnv


@dataclass(frozen=True, slots=True)
class ArrayRef:
    path: str
    elements: tuple[BlockRef, ...]


@dataclass(frozen=True, slots=True)
class TupleValue:
    items: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class OptionValue:
    inner: Value | None


@dataclass(frozen=True, slots=True)
class EnumValue:
    variant: str


Value = TensorValue | ScalarValue | BlockRef | ArrayRef | TupleValue | OptionValue | EnumValue


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def root_env(program: ir.Program, values: Mapping[str, int | str | ex.Expr]) -> SymEnv:
    """The root block's bindings: a value for each generic from `values`
    (an integer, an expression of free symbols, or a dtype name), or its
    declared default."""
    dims: dict[int, ex.Expr] = {}
    packs: dict[int, tuple[ex.Expr, ...]] = {}
    dtypes: dict[int, str] = {}
    missing: list[str] = []
    for generic in program.root.generics:
        value = values.get(generic.name)
        if value is None:
            value = ir.default_of(generic)
        if generic.kind == "dtype":
            if isinstance(value, str):
                dtypes[generic.id] = value
            else:
                missing.append(generic.name)
        elif generic.kind == "dim":
            if isinstance(value, int):
                dims[generic.id] = ex.const(value)
            elif value is not None and not isinstance(value, str):
                dims[generic.id] = value
            else:
                missing.append(generic.name)
        else:
            raise TraceError(f"shape pack `{generic.name}` on the root block is not supported")
    if missing:
        raise TraceError(f"bind the generics {', '.join(f'`{m}`' for m in missing)}")
    return SymEnv(dims, packs, dtypes)


def constant(expr: ex.Expr, what: str) -> int:
    """`expr` as an integer, which the program's structure needs it to be."""
    if not isinstance(expr, ex.Const):
        raise TraceError(
            f"{what} is `{ex.format_expr(expr)}`: give its symbols values, since they decide "
            "the program's structure, not only its sizes"
        )
    return expr.value


# ------------------------------------------------------------------- tracer


@dataclass(frozen=True, slots=True)
class TraceOptions:
    """What the trace cannot read from the program.

    `present` decides whether an optional parameter or child block is
    there (the weights decide that at load). `tied` maps a parameter path to
    the path whose storage it shares, as two paths bound to one checkpoint
    tensor do. `kv_states` are the state paths (with `[*]` for arrays) that
    hold a key/value cache. `parts(path, kind, shape)` says into how many
    devices' parts tensor parallelism splits a parameter, buffer or state
    (1: whole on each), as DTensor splits a model with no `Shards` of its
    own."""

    present: Callable[[str], bool] = lambda path: False
    tied: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    kv_states: frozenset[str] = frozenset()
    parts: Callable[[str, str, tuple[ex.Expr, ...]], int] = lambda path, kind, shape: 1


class _Tracer:
    def __init__(self, program: ir.Program, lowering: Lowering, options: TraceOptions) -> None:
        self.program = program
        self.lowering = lowering
        self.options = options
        self.objects: list[MemoryObject] = []
        self.steps: list[Step] = []
        self.params: dict[str, int] = {}
        self.states: dict[str, int] = {}
        self.retained: list[int] = []
        self.flops_cache: dict[tuple[str, str], ex.Expr] = {}
        self.scope = ""
        self.uses = _static_uses(program)

    # ---- objects

    def new_object(
        self,
        name: str,
        category: Category,
        shape: tuple[ex.Expr, ...],
        dtype: str,
        producer: int | None,
        storage: int | None = None,
        persistent: bool = False,
        path: str | None = None,
        parts: int = 1,
    ) -> int:
        id = len(self.objects)
        nbytes = tensor_bytes(dtype, ex.product(shape)) if storage is None else ex.ZERO
        if parts > 1:
            nbytes = ex.floordiv(nbytes, ex.const(parts))
        self.objects.append(
            MemoryObject(
                id=id,
                name=name,
                category=category,
                shape=shape,
                dtype=dtype,
                nbytes=nbytes,
                producer=producer,
                storage=id if storage is None else self.objects[storage].storage,
                persistent=persistent,
                path=path,
            )
        )
        return id

    def tensor_result(
        self, value: ir.Value, env: SymEnv, producer: int, view_of: int | None = None
    ) -> int:
        if not isinstance(value.type, ir.TensorType):
            raise TraceError(f"`{value.name or value.id}` is not a tensor")
        return self.new_object(
            value.name or f"v{value.id}",
            Category.ACTIVATION,
            env.shape(value.type.shape),
            env.dtype(value.type.dtype),
            producer,
            storage=view_of,
        )

    def emit(
        self,
        kind: str,
        label: str,
        inputs: Sequence[Value],
        flops: ex.Expr,
        implementation: str | None = None,
    ) -> int:
        """Records a step and returns its index; its outputs are added by
        the caller with `producer` set to it."""
        index = len(self.steps)
        self.steps.append(
            Step(
                index=index,
                kind=kind,
                label=label,
                inputs=tuple(dict.fromkeys(i for v in inputs for i in _tensors(v))),
                outputs=(),
                scope=self.scope,
                flops=flops,
                implementation=implementation,
            )
        )
        return index

    def finish(self, index: int, outputs: Sequence[int]) -> None:
        step = self.steps[index]
        self.steps[index] = Step(
            index=step.index,
            kind=step.kind,
            label=step.label,
            inputs=step.inputs,
            outputs=tuple(outputs),
            scope=step.scope,
            flops=step.flops,
            implementation=step.implementation,
        )

    # ---- the module's own memory

    def instantiate(self, env: SymEnv) -> None:
        """A persistent object for every parameter, buffer and state."""
        for entry in self.program.manifest:
            repeat = [constant(env.dim(d), f"the length of `{entry.path}`") for d in entry.repeat]
            shape = env.shape(entry.shape)
            dtype = env.dtype(entry.dtype)
            for path in ir.repeat_paths(entry.path, repeat):
                if entry.optional and not self.options.present(path):
                    continue
                parts = self.options.parts(path, entry.kind, shape)
                if entry.kind == "state":
                    category = (
                        Category.KV_CACHE
                        if entry.path in self.options.kv_states
                        else Category.STATE
                    )
                    self.states[path] = self.new_object(
                        path, category, shape, dtype, None, persistent=True, path=path, parts=parts
                    )
                    continue
                category = Category.PARAMETER if entry.kind == "param" else Category.BUFFER
                tied = self.options.tied.get(path)
                storage = self.params.get(tied) if tied is not None else None
                self.params[path] = self.new_object(
                    path,
                    category,
                    shape,
                    dtype,
                    None,
                    storage,
                    persistent=True,
                    path=path,
                    parts=parts,
                )

    # ---- functions and regions

    def call(self, function: ir.Function, env: SymEnv, arguments: Sequence[Value]) -> list[Value]:
        if len(function.body.args) != len(arguments):
            raise TraceError(
                f"`{function.name}` takes {len(function.body.args)} arguments, got {len(arguments)}"
            )
        values: dict[int, Value] = {
            arg.id: value for arg, value in zip(function.body.args, arguments, strict=True)
        }
        outer = self.scope
        receiver = arguments[0] if arguments else None
        if function.block is not None and isinstance(receiver, BlockRef):
            self.scope = receiver.path
        try:
            return self.region(function.body, env, values)
        finally:
            self.scope = outer

    def region(self, region: ir.Region, env: SymEnv, values: dict[int, Value]) -> list[Value]:
        for op in region.ops:
            if op.kind in ("return", "yield"):
                return [values[id] for id in op.operands]
            results = self.op(op, env, values)
            for result, value in zip(op.results, results, strict=True):
                values[result.id] = value
        raise TraceError("a region without a terminator")

    def op(self, op: ir.Op, env: SymEnv, values: dict[int, Value]) -> list[Value]:
        kind = op.kind
        operands = [values[id] for id in op.operands]
        result = op.results[0] if op.results else None
        rtype = result.type if result is not None else None

        if kind in CONSTANTS:
            if kind == "const.int":
                return [ScalarValue(ex.const(_int_attr(op, "value")))]
            if kind == "const.dim":
                return [ScalarValue(env.dim(ir.parse_dim(op.attrs["value"])))]
            return [ScalarValue()]
        if kind == "enum.const":
            return [EnumValue(str(op.attrs.get("variant", op.attrs.get("name", ""))))]
        if kind in ARITHMETIC or kind == "cast":
            if not isinstance(rtype, ir.TensorType):
                return [ScalarValue(_fold(kind, operands))]
            assert result is not None
            if kind == "cast" and isinstance(operands[0], TensorValue):
                source = self.objects[operands[0].id]
                if source.dtype == env.dtype(rtype.dtype):
                    return [operands[0]]
            numel = ex.product(env.shape(rtype.shape))
            # Times or over a scalar is linear in the tensor: `scale`, so
            # that autograd's rules can tell it from `x * x`.
            scalar = [not isinstance(o, TensorValue) for o in operands]
            linear = (kind == "mul" and scalar.count(True) == 1) or (
                kind == "div" and scalar == [False, True]
            )
            step = self.emit(
                "scale" if linear else kind, kind, operands, ex.ZERO if kind == "cast" else numel
            )
            out = self.tensor_result(result, env, step)
            self.finish(step, [out])
            return [TensorValue(out)]
        if kind in ("reshape", "permute", "broadcast", "slice"):
            return [self.view(op, env, operands)]
        if kind in ("concat", "fill", "iota", "tensor.element"):
            if not isinstance(rtype, ir.TensorType):
                return [ScalarValue()]
            assert result is not None
            step = self.emit(kind, kind, operands, ex.ZERO)
            out = self.tensor_result(result, env, step)
            self.finish(step, [out])
            return [TensorValue(out)]
        if kind in ("comprehension", "reduce"):
            return [self.comprehension(op, env, values)]
        if kind == "tuple.make":
            return [TupleValue(tuple(operands))]
        if kind in ("tuple.get", "struct.get"):
            whole = operands[0]
            index = _int_attr(op, "value", _int_attr(op, "index"))
            if isinstance(whole, TupleValue):
                return [whole.items[index]]
            return [ScalarValue()]
        if kind == "option.some":
            return [OptionValue(operands[0])]
        if kind == "option.none":
            return [OptionValue(None)]
        if kind == "option.match":
            some, none = op.regions
            scrutinee = operands[0]
            if isinstance(scrutinee, OptionValue) and scrutinee.inner is not None:
                inner = dict(values)
                inner[some.args[0].id] = scrutinee.inner
                return self.region(some, env, inner)
            return self.region(none, env, dict(values))
        if kind == "enum.match":
            scrutinee = operands[0]
            variants = [str(v) for v in cast(list[str], op.attrs.get("variants", []))]
            for variant, region in zip(variants, op.regions, strict=True):
                if (
                    isinstance(scrutinee, EnumValue) and variant == scrutinee.variant
                ) or variant == "_":
                    return self.region(region, env, dict(values))
            raise TraceError("an `enum.match` whose value the trace cannot know")
        if kind == "if":
            # A runtime condition: both branches run in the trace, one after
            # the other, so the peak covers whichever the data picks.
            self.region(op.regions[0], env, dict(values))
            return self.region(op.regions[1], env, dict(values))
        if kind in ("call", "semantic.call"):
            return self.call_op(op, env, operands)
        if kind == "block.param":
            ref = _block(operands[0])
            path = _join(ref.path, str(op.attrs["name"]))
            if path in self.params:
                tensor = TensorValue(self.params[path])
                return [OptionValue(tensor) if isinstance(rtype, ir.OptionalType) else tensor]
            if isinstance(rtype, ir.OptionalType):
                return [OptionValue(None)]
            raise TraceError(f"parameter `{path}` is missing")
        if kind == "block.sub":
            return [self.sub(_block(operands[0]), str(op.attrs["name"]))]
        if kind == "state.read":
            path = _join(_block(operands[0]).path, str(op.attrs["name"]))
            return [TensorValue(self.states[path])]
        if kind == "state.write":
            path = _join(_block(operands[0]).path, str(op.attrs["name"]))
            written = operands[1]
            if isinstance(written, TensorValue):
                current = self.objects[self.states[path]].storage
                if self.objects[written.id].storage != current:
                    self.states[path] = written.id
                    self.retained.append(written.id)
            return []
        if kind == "array.get":
            array = operands[0]
            index = operands[1]
            if (
                isinstance(array, ArrayRef)
                and isinstance(index, ScalarValue)
                and index.value is not None
            ):
                return [array.elements[constant(index.value, "an array index")]]
            raise TraceError("an array index the trace cannot know")
        if kind == "static_for":
            array = operands[0]
            if not isinstance(array, ArrayRef):
                raise TraceError("`static_for` over something not a block array")
            carried = list(operands[1:])
            region = op.regions[0]
            for element in array.elements:
                inner = dict(values)
                inner[region.args[0].id] = element
                for arg, value in zip(region.args[1:], carried, strict=True):
                    inner[arg.id] = value
                carried = self.region(region, env, inner)
            return carried
        if kind == "static_range":
            bounds = operands[:2]
            if not all(isinstance(b, ScalarValue) and b.value is not None for b in bounds):
                raise TraceError("a `static for` range the trace cannot know")
            start = constant(cast(ScalarValue, bounds[0]).value or ex.ZERO, "a loop bound")
            stop = constant(cast(ScalarValue, bounds[1]).value or ex.ZERO, "a loop bound")
            carried = list(operands[2:])
            region = op.regions[0]
            for position in range(start, stop):
                inner = dict(values)
                inner[region.args[0].id] = ScalarValue(ex.const(position))
                for arg, value in zip(region.args[1:], carried, strict=True):
                    inner[arg.id] = value
                carried = self.region(region, env, inner)
            return carried
        if kind == "while":
            # One iteration stands for all: the carried values keep their
            # shapes, so every iteration has the same live memory.
            condition, body = op.regions
            inner = dict(values)
            for arg, value in zip(condition.args, operands, strict=True):
                inner[arg.id] = value
            self.region(condition, env, inner)
            inner = dict(values)
            for arg, value in zip(body.args, operands, strict=True):
                inner[arg.id] = value
            return self.region(body, env, inner)
        raise TraceError(f"the Core IR operation `{kind}` is not supported")

    def view(self, op: ir.Op, env: SymEnv, operands: list[Value]) -> Value:
        result = op.results[0]
        base = operands[0]
        if not isinstance(base, TensorValue) or not isinstance(result.type, ir.TensorType):
            return ScalarValue()
        if op.kind == "reshape" and not base.contiguous:
            step = self.emit("reshape", "reshape (copy)", operands, ex.ZERO)
            out = self.tensor_result(result, env, step)
            self.finish(step, [out])
            return TensorValue(out)
        step = self.emit(op.kind, op.kind, operands, ex.ZERO)
        out = self.tensor_result(result, env, step, view_of=base.id)
        self.finish(step, [out])
        order = base.order
        if op.kind == "permute":
            axes = [int(a) for a in cast(list[int], op.attrs.get("shape", []))]
            # The memory order in the permuted tensor's own axes.
            before = base.order or tuple(range(len(axes))) if base.order is not None else None
            order = None if before is None else tuple(axes.index(a) for a in before)
            if order is not None and list(order) == sorted(order):
                order = ()
        elif op.kind == "reshape":
            order = ()
        elif op.kind == "broadcast" or (
            op.kind == "slice"
            and not (base.contiguous and _slice_contiguous(op, env, self.objects[base.id].shape))
        ):
            order = None
        return TensorValue(out, order)

    def comprehension(self, op: ir.Op, env: SymEnv, values: dict[int, Value]) -> Value:
        result = op.results[0]
        region = op.regions[0]
        reads = [values[id] for id in _outer_reads(region) if id in values]
        body = _region_flops(region, env, self.program)
        domain = _domain(op, env) if op.kind == "reduce" else ex.ONE
        if not isinstance(result.type, ir.TensorType):
            if reads:
                step = self.emit(op.kind, op.kind, reads, ex.mul(domain, ex.add(body, ex.ONE)))
                self.finish(step, [])
            return ScalarValue()
        numel = ex.product(env.shape(result.type.shape))
        flops = (
            ex.mul(numel, body)
            if op.kind == "comprehension"
            else ex.mul(numel, domain, ex.add(body, ex.ONE))
        )
        step = self.emit(op.kind, op.kind, reads, flops)
        out = self.tensor_result(result, env, step)
        self.finish(step, [out])
        return TensorValue(out)

    def sub(self, ref: BlockRef, name: str) -> Value:
        block = self.program.blocks[ref.block]
        member = block.member(name)
        path = _join(ref.path, name)
        member_type = member.type
        optional = isinstance(member_type, ir.OptionalType)
        if isinstance(member_type, ir.OptionalType):
            member_type = member_type.inner
        if isinstance(member_type, ir.ArrayType):
            element = member_type.element
            if not isinstance(element, ir.NamedType):
                raise TraceError(f"`{path}` is not an array of blocks")
            length = constant(ref.env.dim(member_type.length), f"the length of `{path}`")
            child = self.program.blocks[element.name]
            env = ref.env.instance(child, element.args)
            return ArrayRef(
                path, tuple(BlockRef(f"{path}.{i}", element.name, env) for i in range(length))
            )
        if not isinstance(member_type, ir.NamedType):
            raise TraceError(f"`{path}` is not a block")
        child = self.program.blocks[member_type.name]
        value = BlockRef(path, member_type.name, ref.env.instance(child, member_type.args))
        if optional:
            return OptionValue(value if self.options.present(path) else None)
        return value

    def call_op(self, op: ir.Op, env: SymEnv, operands: list[Value]) -> list[Value]:
        callee = self.program.functions[str(op.attrs["callee"])]
        receiver = operands[0] if operands else None
        base = receiver.env if callee.block is not None and isinstance(receiver, BlockRef) else None
        callee_env = env.bind(ir.call_substitution(op), base)
        native = self.lowering.native(op)
        if native is None:
            return self.call(callee, callee_env, operands)
        label = str(op.attrs["callee"]).rsplit("::", 1)[-1]
        flops = self.canonical_flops(callee, callee_env)
        step = self.emit("semantic.call", label, operands, flops, implementation=native)
        in_place = self.in_place_target(native, operands, op)
        if native in self.lowering.identities and operands and isinstance(operands[0], TensorValue):
            in_place = operands[0].id
        outputs: list[int] = []
        results: list[Value] = []
        for result in op.results:
            if not isinstance(result.type, ir.TensorType):
                results.append(ScalarValue())
                continue
            out = self.tensor_result(result, env, step, view_of=in_place)
            outputs.append(out)
            results.append(TensorValue(out, self.lowering.layout(native)))
            in_place = None
        self.finish(step, outputs)
        return results

    def in_place_target(self, native: str, operands: list[Value], op: ir.Op) -> int | None:
        """The state operand a cache write updates in place: one nothing
        else reads."""
        if native not in self.lowering.in_place_writes or not operands:
            return None
        first = operands[0]
        if not isinstance(first, TensorValue):
            return None
        storage = self.objects[first.id].storage
        if not self.objects[storage].persistent or self.uses.get(op.operands[0], 0) > 1:
            return None
        return first.id

    def canonical_flops(self, function: ir.Function, env: SymEnv) -> ex.Expr:
        """The arithmetic of a library call's canonical body, which a native
        implementation performs too."""
        key = (function.name, _env_key(env))
        if key in self.flops_cache:
            return self.flops_cache[key]
        dry = _Tracer(self.program, Lowering(inline_all=True), self.options)
        arguments: list[Value] = []
        for arg in function.body.args:
            arguments.append(dry.placeholder(arg.type, env))
        try:
            dry.call(function, env, arguments)
            flops = ex.total(s.flops for s in dry.steps)
        except TraceError:
            flops = ex.ZERO
        self.flops_cache[key] = flops
        return flops

    def placeholder(self, type: ir.Type, env: SymEnv) -> Value:
        if isinstance(type, ir.TensorType):
            return TensorValue(
                self.new_object(
                    "input", Category.INPUT, env.shape(type.shape), env.dtype(type.dtype), None
                )
            )
        if isinstance(type, ir.OptionalType):
            return OptionValue(None)
        if isinstance(type, ir.TupleType):
            return TupleValue(tuple(self.placeholder(t, env) for t in type.elements))
        return ScalarValue()


def _block(value: Value) -> BlockRef:
    if not isinstance(value, BlockRef):
        raise TraceError("a block member read from something not a block")
    return value


def _tensors(value: Value) -> Iterator[int]:
    if isinstance(value, TensorValue):
        yield value.id
    elif isinstance(value, TupleValue):
        for item in value.items:
            yield from _tensors(item)
    elif isinstance(value, OptionValue) and value.inner is not None:
        yield from _tensors(value.inner)


def _static_uses(program: ir.Program) -> dict[int, int]:
    """How many operations read each value, over the whole program: value
    ids are unique across its functions."""
    uses: dict[int, int] = {}
    for function in program.functions.values():
        for op in function.body.walk():
            for id in op.operands:
                uses[id] = uses.get(id, 0) + 1
    return uses


def _fold(kind: str, operands: list[Value]) -> ex.Expr | None:
    """Compile-time integer arithmetic on scalars, as loop bounds need."""
    values = [o.value if isinstance(o, ScalarValue) else None for o in operands]
    if any(v is None for v in values) or kind not in ("add", "sub", "mul", "div"):
        return None
    a, b = cast(ex.Expr, values[0]), cast(ex.Expr, values[1])
    if kind == "add":
        return ex.add(a, b)
    if kind == "mul":
        return ex.mul(a, b)
    if isinstance(a, ex.Const) and isinstance(b, ex.Const):
        return ex.const(
            a.value - b.value if kind == "sub" else a.value // b.value if b.value else 0
        )
    return None


def _outer_reads(region: ir.Region) -> list[int]:
    """The values a region reads that it does not define."""
    defined: set[int] = set()
    reads: list[int] = []

    def visit(region: ir.Region) -> None:
        defined.update(arg.id for arg in region.args)
        for op in region.ops:
            for id in op.operands:
                if id not in defined:
                    reads.append(id)
            defined.update(result.id for result in op.results)
            for inner in op.regions:
                visit(inner)

    visit(region)
    return list(dict.fromkeys(reads))


def _domain(op: ir.Op, env: SymEnv) -> ex.Expr:
    sizes: list[ex.Expr] = []
    for index in cast(list[dict[str, ir.JsonValue]], op.attrs.get("indices", [])):
        sizes.extend(env.shape(ir.parse_shape(cast(list[ir.JsonValue], index.get("domain", [])))))
    return ex.product(sizes)


def _region_flops(region: ir.Region, env: SymEnv, program: ir.Program, depth: int = 0) -> ex.Expr:
    """Operations per element of a comprehension or reduction body."""
    count: list[ex.Expr] = []
    for op in region.ops:
        if op.kind in ARITHMETIC:
            count.append(ex.ONE)
        elif op.kind == "reduce":
            body = _region_flops(op.regions[0], env, program, depth)
            count.append(ex.mul(_domain(op, env), ex.add(body, ex.ONE)))
        elif op.kind in ("call", "semantic.call") and depth < 8:
            callee = program.functions.get(str(op.attrs.get("callee")))
            if callee is not None:
                try:
                    inner = env.bind(ir.call_substitution(op))
                    count.append(_region_flops(callee.body, inner, program, depth + 1))
                except TraceError:
                    pass
        elif op.regions and op.kind not in ("comprehension",):
            for inner in op.regions:
                count.append(_region_flops(inner, env, program, depth))
    return ex.total(count)


class _SliceAxis(TypedDict, total=False):
    """One axis of a `slice` op's `axes` attribute: kept `whole`, or cut
    from `start` to `stop` by `step` (and `squeeze`d out when indexed)."""

    whole: ir.JsonValue
    start: ir.JsonValue
    stop: ir.JsonValue
    step: int
    squeeze: bool


def _slice_contiguous(op: ir.Op, env: SymEnv, shape: tuple[ex.Expr, ...]) -> bool:
    """A slice keeps storage contiguous when it narrows at most one axis by
    a unit step and every axis after that one is kept whole."""
    axes = cast(list[_SliceAxis], op.attrs.get("axes", []))
    narrowed = False
    position = 0
    for axis in axes:
        if "whole" in axis:
            if narrowed:
                pass
            position += len(env.shape(ir.parse_shape([axis["whole"]])))
            continue
        size = shape[position] if position < len(shape) else None
        position += 1
        start = env.dim(ir.parse_dim(axis.get("start", 0)))
        stop = env.dim(ir.parse_dim(axis.get("stop", 0)))
        step = int(axis.get("step", 1))
        whole = start == ex.ZERO and step == 1 and size is not None and stop == size
        if bool(axis.get("squeeze")):
            if narrowed:
                return False
            continue
        if whole:
            continue
        if narrowed or step != 1:
            return False
        narrowed = True
    return True


def _env_key(env: SymEnv) -> str:
    dims = ",".join(f"{k}={ex.format_expr(v)}" for k, v in sorted(env.dims.items()))
    packs = ",".join(
        f"{k}=[{';'.join(ex.format_expr(d) for d in v)}]" for k, v in sorted(env.packs.items())
    )
    dtypes = ",".join(f"{k}={v}" for k, v in sorted(env.dtypes.items()))
    return f"{dims}|{packs}|{dtypes}"


def trace(
    program: ir.Program,
    entry: str | None = None,
    root: Mapping[str, int | str | ex.Expr] | None = None,
    inputs: Mapping[str, int | ex.Expr] | None = None,
    *,
    lowering: Lowering | None = None,
    options: TraceOptions | None = None,
) -> TensorGraph:
    """The memory objects and steps of one root entry.

    `root` binds the root block's generics and `inputs` the entry's own
    (an expression keeps one free: `ex.sym("B")`); generics with defaults
    may be left out. Symbols that decide the program's structure (a loop
    count, an array length) must be integers."""
    tracer = _Tracer(program, lowering or Lowering(), options or TraceOptions())
    env = root_env(program, root or {})
    tracer.instantiate(env)
    function = program.entry(entry)
    bound = entry_env(function, env, inputs or {})
    receiver = BlockRef("", program.root.name, env)
    arguments: list[Value] = [receiver]
    input_ids: list[int] = []
    for param in function.params:
        if isinstance(param.type, ir.TensorType):
            id = tracer.new_object(
                param.name,
                Category.INPUT,
                bound.shape(param.type.shape),
                bound.dtype(param.type.dtype),
                None,
            )
            input_ids.append(id)
            arguments.append(TensorValue(id))
        else:
            arguments.append(tracer.placeholder(param.type, bound))
    results = tracer.call(function, bound, arguments)
    outputs = [i for value in results for i in _tensors(value)]
    return TensorGraph(
        entry=function.name,
        objects=tuple(tracer.objects),
        steps=tuple(tracer.steps),
        inputs=tuple(input_ids),
        outputs=tuple(dict.fromkeys([*outputs, *tracer.retained])),
    )


def entry_env(function: ir.Function, root: SymEnv, inputs: Mapping[str, int | ex.Expr]) -> SymEnv:
    dims = dict(root.dims)
    packs = dict(root.packs)
    dtypes = dict(root.dtypes)
    missing: list[str] = []
    for generic in function.generics:
        value: int | str | ex.Expr | None = inputs.get(generic.name)
        if value is None:
            value = ir.default_of(generic)
        if isinstance(value, int):
            value = ex.const(value)
        if generic.kind == "dim" and value is not None and not isinstance(value, str):
            dims[generic.id] = value
        elif generic.kind == "dtype" and isinstance(value, str):
            dtypes[generic.id] = value
        else:
            missing.append(generic.name)
    if missing:
        raise TraceError(
            f"bind the generics of `{function.short_name}`: " + ", ".join(f"`{m}`" for m in missing)
        )
    return SymEnv(dims, packs, dtypes)


def _int_attr(op: ir.Op, key: str, default: int = 0) -> int:
    """An integer attribute of `op`, `default` when it has none."""
    value = op.attrs.get(key, default)
    if not isinstance(value, int):
        raise LinnetError(f"`{op.kind}` has a non-integer `{key}`")
    return value
