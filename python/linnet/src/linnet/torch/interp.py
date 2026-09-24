"""Evaluates Core IR with PyTorch operations.

Every Linnet value has a Python representation:

- scalars and tensors are `torch.Tensor` (scalars as 0-d tensors);
- tuples are Python tuples, optionals are the value or `None`;
- enum values are their variant name;
- block instances are `BlockInstance`, sub-block arrays are lists of them;
- inside index notation an index variable is an `IndexValue`: one broadcastable
  position grid per axis of its domain.

Index notation is evaluated on grids: a comprehension gives each index axis a
position in a shared grid, `tensor.element` becomes advanced indexing with the
grids, and a reduction sums (or otherwise reduces) over the axes it introduces.
This follows the canonical semantics exactly; it is not the fast path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from ..plan import Env, Plan, PlanError
from .dtypes import torch_dtype
from .native import NATIVE, causal_mask

Value = Any


@dataclass
class BlockInstance:
    """One instantiated block: its bindings, tensors, and sub-blocks."""

    name: str
    env: Env
    params: dict[str, torch.Tensor | None] = field(default_factory=dict)  # type: ignore[arg-type]
    subs: dict[str, BlockInstance | list[BlockInstance]] = field(default_factory=dict)  # type: ignore[arg-type]
    # `state` members: the current values, and a hook that persists a write
    # in the owning module so the next call starts from it.
    states: dict[str, torch.Tensor] = field(default_factory=dict)  # type: ignore[arg-type]
    on_write: Callable[[str, torch.Tensor], None] | None = None


@dataclass
class IndexValue:
    """An index variable: the grid axes its domain occupies."""

    axes: list[tuple[int, int]]  # (position in the grid, size)


@dataclass
class Grid:
    """The index axes introduced by the enclosing comprehension and reductions.

    Every tensor computed inside index notation has one axis per grid axis,
    in grid order, so that broadcasting lines the axes up.
    """

    sizes: list[int] = field(default_factory=list[int])

    def add_axes(self, sizes: list[int]) -> IndexValue:
        axes = [(len(self.sizes) + i, size) for i, size in enumerate(sizes)]
        self.sizes.extend(sizes)
        return IndexValue(axes)

    def rank(self) -> int:
        return len(self.sizes)

    def position(self, axis: int, size: int, device: torch.device) -> torch.Tensor:
        shape = [1] * self.rank()
        shape[axis] = size
        return torch.arange(size, device=device).reshape(shape)

    def pad(self, value: torch.Tensor) -> torch.Tensor:
        """Gives `value` trailing singleton axes up to the grid's rank."""
        if value.dim() < self.rank():
            return value.reshape(list(value.shape) + [1] * (self.rank() - value.dim()))
        return value

    def expand(self, value: torch.Tensor) -> torch.Tensor:
        """Broadcasts `value` to the full grid."""
        value = self.pad(value)
        return value.expand(self.sizes) if list(value.shape) != self.sizes else value


Reduction = Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]


def _prod(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    for dim in sorted(dims, reverse=True):
        x = torch.prod(x, dim=dim)
    return x


def _logical(x: torch.Tensor, dims: tuple[int, ...], is_any: bool) -> torch.Tensor:
    for dim in sorted(dims, reverse=True):
        x = torch.any(x, dim=dim) if is_any else torch.all(x, dim=dim)
    return x


REDUCTIONS: dict[str, Reduction] = {
    "sum": lambda x, dims: torch.sum(x, dim=dims),
    "prod": _prod,
    "max": lambda x, dims: torch.amax(x, dim=dims),
    "min": lambda x, dims: torch.amin(x, dim=dims),
    "any": lambda x, dims: _logical(x, dims, True),
    "all": lambda x, dims: _logical(x, dims, False),
}


class Interpreter:
    def __init__(self, plan: Plan, device: torch.device) -> None:
        self.plan = plan
        self.device = device

    # ------------------------------------------------------------ functions

    def call(self, function: dict[str, Any], env: Env, arguments: list[Value]) -> Value:
        for constraint in function["constraints"]:
            if not env.relation_holds(constraint):
                raise PlanError(f"constraint of `{function['name']}` does not hold at runtime")
        body = function["body"]
        if len(body["args"]) != len(arguments):
            raise PlanError(
                f"`{function['name']}` takes {len(body['args'])} arguments, got {len(arguments)}"
            )
        values: dict[int, Value] = {}
        for arg, value in zip(body["args"], arguments, strict=True):
            values[arg["id"]] = value
        result = self.run_region(body, env, values, Grid())
        return result[0] if len(result) == 1 else tuple(result)

    def run_region(
        self, region: dict[str, Any], env: Env, values: dict[int, Value], grid: Grid
    ) -> list[Value]:
        for op in region["ops"]:
            kind = op["kind"]
            if kind in ("return", "yield"):
                return [values[operand] for operand in op["operands"]]
            results = self.run_op(op, env, values, grid)
            for result, value in zip(op["results"], results, strict=True):
                values[result["id"]] = value
        raise PlanError("region without a terminator")

    # ------------------------------------------------------------ operations

    def run_op(
        self, op: dict[str, Any], env: Env, values: dict[int, Value], grid: Grid
    ) -> list[Value]:
        kind: str = op["kind"]
        attrs: dict[str, Any] = op["attrs"]
        operands: list[Any] = [values[operand] for operand in op["operands"]]
        result_type: dict[str, Any] | None = op["results"][0]["type"] if op["results"] else None

        def tensor(i: int) -> torch.Tensor:
            return cast(torch.Tensor, operands[i])

        def result_dtype() -> torch.dtype:
            assert result_type is not None
            return torch_dtype(env, result_type["dtype"])

        if kind == "const.int":
            return [torch.tensor(int(attrs["value"]), dtype=result_dtype(), device=self.device)]
        if kind == "const.float":
            return [torch.tensor(float(attrs["value"]), dtype=result_dtype(), device=self.device)]
        if kind == "const.bool":
            return [torch.tensor(bool(attrs["value"]), device=self.device)]
        if kind == "const.dim":
            return [torch.tensor(env.dim(attrs["value"]), dtype=torch.int64, device=self.device)]
        if kind == "enum.const":
            return [attrs["name"]]

        binary: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
            "div": self._div,
            "rem": torch.remainder,
            "min": torch.minimum,
            "max": torch.maximum,
            "and": torch.logical_and,
            "or": torch.logical_or,
            "bitand": torch.bitwise_and,
            "bitor": torch.bitwise_or,
            "bitxor": torch.bitwise_xor,
            "shl": torch.bitwise_left_shift,
            "shr": torch.bitwise_right_shift,
        }
        if kind in binary:
            return [binary[kind](tensor(0), tensor(1))]
        if kind == "compare":
            comparisons: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
                "eq": torch.eq,
                "ne": torch.ne,
                "lt": torch.lt,
                "le": torch.le,
                "gt": torch.gt,
                "ge": torch.ge,
            }
            if isinstance(operands[0], str):  # enum values compare by variant
                is_equal = operands[0] == operands[1]
                return [torch.tensor(is_equal if attrs["compare"] == "eq" else not is_equal)]
            return [comparisons[attrs["compare"]](tensor(0), tensor(1))]
        unary: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "not": torch.logical_not,
            "neg": torch.neg,
            "exp": torch.exp,
            "log": torch.log,
            "sqrt": torch.sqrt,
            "rsqrt": torch.rsqrt,
            "sin": torch.sin,
            "cos": torch.cos,
            "tanh": torch.tanh,
            "abs": torch.abs,
        }
        if kind in unary:
            return [unary[kind](tensor(0))]
        if kind == "cast":
            return [operands[0].to(result_dtype())]
        if kind == "select":
            return [torch.where(operands[0], operands[1], operands[2])]

        if kind == "reshape":
            return [operands[0].reshape(env.shape(attrs["shape"]))]
        if kind == "permute":
            return [operands[0].permute(env.shape(attrs["shape"]))]
        if kind == "broadcast":
            return [operands[0].expand(env.shape(attrs["shape"]))]
        if kind == "slice":
            return [self._slice(operands[0], attrs["axes"], env)]
        if kind == "concat":
            return [torch.cat(operands, dim=int(attrs["axis"]))]
        if kind == "fill":
            return [
                torch.full(
                    env.shape(attrs["shape"]),
                    operands[0].item(),
                    dtype=result_dtype(),
                    device=self.device,
                )
            ]
        if kind == "iota":
            return [
                torch.arange(env.shape(attrs["shape"])[0], dtype=result_dtype(), device=self.device)
            ]

        if kind == "tensor.element":
            return [self._element(operands[0], operands[1:], grid)]
        if kind == "comprehension":
            return [self._comprehension(op, env, values, grid)]
        if kind == "reduce":
            return [self._reduce(op, env, values, grid)]

        if kind == "tuple.make":
            return [tuple(operands)]
        if kind == "tuple.get" or kind == "struct.get":
            return [operands[0][int(attrs["value"])]]
        if kind == "option.some":
            return [operands[0]]
        if kind == "option.none":
            return [None]
        if kind == "option.match":
            some_region, none_region = op["regions"]
            if operands[0] is not None:
                inner = dict(values)
                inner[some_region["args"][0]["id"]] = operands[0]
                return self.run_region(some_region, env, inner, grid)
            return self.run_region(none_region, env, dict(values), grid)
        if kind == "enum.match":
            for variant, region in zip(attrs["variants"], op["regions"], strict=True):
                if variant == operands[0] or variant == "_":
                    return self.run_region(region, env, dict(values), grid)
            raise PlanError(f"no arm matches enum value `{operands[0]}`")
        if kind == "if":
            chosen = op["regions"][0] if bool(tensor(0).item()) else op["regions"][1]
            return self.run_region(chosen, env, dict(values), grid)

        if kind in ("call", "semantic.call"):
            selected = attrs.get("selected", "canonical decomposition")
            if selected == "torch.tril":
                assert result_type is not None
                return [causal_mask(env.shape(result_type["shape"]), self.device)]
            if selected != "canonical decomposition":
                if selected not in NATIVE:
                    raise PlanError(
                        f"the plan selected `{selected}`, which this materializer lacks"
                    )
                return [NATIVE[selected](operands, result_dtype() if result_type else None)]
            callee = self.plan.functions[attrs["callee"]]
            callee_env = self._callee_env(callee, attrs["substitution"], env, operands)
            return [self.call(callee, callee_env, operands)]
        if kind == "block.param":
            return [cast(BlockInstance, operands[0]).params[attrs["name"]]]
        if kind == "block.sub":
            return [cast(BlockInstance, operands[0]).subs[attrs["name"]]]
        if kind == "state.read":
            return [cast(BlockInstance, operands[0]).states[attrs["name"]]]
        if kind == "state.write":
            instance = cast(BlockInstance, operands[0])
            instance.states[attrs["name"]] = tensor(1)
            if instance.on_write is not None:
                instance.on_write(attrs["name"], tensor(1))
            return []
        if kind == "array.get":
            return [cast(list[BlockInstance], operands[0])[int(tensor(1).item())]]
        if kind == "static_for":
            carried: list[Any] = list(operands[1:])
            region = op["regions"][0]
            for element in cast(list[BlockInstance], operands[0]):
                inner = dict(values)
                inner[region["args"][0]["id"]] = element
                for arg, value in zip(region["args"][1:], carried, strict=True):
                    inner[arg["id"]] = value
                carried = self.run_region(region, env, inner, grid)
            return carried
        if kind == "while":
            carried = list(operands)
            condition, body = op["regions"]
            while True:
                inner = dict(values)
                for arg, value in zip(condition["args"], carried, strict=True):
                    inner[arg["id"]] = value
                if not bool(self.run_region(condition, env, inner, grid)[0].item()):
                    break
                inner = dict(values)
                for arg, value in zip(body["args"], carried, strict=True):
                    inner[arg["id"]] = value
                carried = self.run_region(body, env, inner, grid)
            return carried
        if kind == "static_range":
            start, stop = int(tensor(0).item()), int(tensor(1).item())
            carried = list(operands[2:])
            region = op["regions"][0]
            for position in range(start, stop):
                inner = dict(values)
                inner[region["args"][0]["id"]] = torch.tensor(
                    position, dtype=torch.int64, device=self.device
                )
                for arg, value in zip(region["args"][1:], carried, strict=True):
                    inner[arg["id"]] = value
                carried = self.run_region(region, env, inner, grid)
            return carried

        raise PlanError(f"unsupported Core IR operation `{kind}`")

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.dtype.is_floating_point or b.dtype.is_floating_point:
            return torch.div(a, b)
        return torch.div(a, b, rounding_mode="floor")

    @staticmethod
    def _slice(base: torch.Tensor, axes: list[dict[str, Any]], env: Env) -> torch.Tensor:
        index: list[Any] = []
        for axis in axes:
            if "whole" in axis:
                # A whole axis stands for every axis of a shape pack.
                index.extend([slice(None)] * len(env.shape(axis["whole"])))
            elif axis["squeeze"]:
                index.append(env.dim(axis["start"]))
            else:
                index.append(
                    slice(env.dim(axis["start"]), env.dim(axis["stop"]), int(axis["step"]))
                )
        return base[tuple(index)]

    def _element(self, base: torch.Tensor, indices: list[Value], grid: Grid) -> torch.Tensor:
        positions: list[torch.Tensor] = []
        for index in indices:
            if isinstance(index, IndexValue):
                positions.extend(
                    grid.position(axis, size, self.device) for axis, size in index.axes
                )
            else:
                positions.append(grid.pad(index.to(torch.int64)))
        return grid.pad(base[tuple(positions)])

    def _index_sizes(self, indices: list[dict[str, Any]], env: Env) -> list[list[int]]:
        return [env.shape(index["domain"]) for index in indices]

    def _comprehension(
        self, op: dict[str, Any], env: Env, values: dict[int, Value], grid: Grid
    ) -> torch.Tensor:
        region = op["regions"][0]
        inner_grid = Grid(list(grid.sizes))
        inner = dict(values)
        for arg, sizes in zip(
            region["args"], self._index_sizes(op["attrs"]["indices"], env), strict=True
        ):
            inner[arg["id"]] = inner_grid.add_axes(sizes)
        body = cast(torch.Tensor, self.run_region(region, env, inner, inner_grid)[0])
        return inner_grid.expand(body).contiguous()

    def _reduce(
        self, op: dict[str, Any], env: Env, values: dict[int, Value], grid: Grid
    ) -> torch.Tensor:
        region = op["regions"][0]
        inner_grid = Grid(list(grid.sizes))
        first_new_axis = grid.rank()
        inner = dict(values)
        for arg, sizes in zip(
            region["args"], self._index_sizes(op["attrs"]["indices"], env), strict=True
        ):
            inner[arg["id"]] = inner_grid.add_axes(sizes)
        body = inner_grid.expand(
            cast(torch.Tensor, self.run_region(region, env, inner, inner_grid)[0])
        )
        dims = tuple(range(first_new_axis, inner_grid.rank()))
        return REDUCTIONS[op["attrs"]["reduce"]](body, dims)

    def _callee_env(
        self, callee: dict[str, Any], substitution: dict[str, Any], env: Env, operands: list[Value]
    ) -> Env:
        callee_env = Env()
        for symbol, expr in substitution["dims"].items():
            callee_env.dims[int(symbol)] = env.dim(expr)
        for symbol, units in substitution["packs"].items():
            callee_env.packs[int(symbol)] = env.shape(units)
        for var, spec in substitution["dtypes"].items():
            callee_env.dtypes[int(var)] = env.dtype_name(spec)
        # A method sees its block's bindings through the receiver.
        if callee["block"] is not None and operands and isinstance(operands[0], BlockInstance):
            receiver = operands[0]
            callee_env.dims = {**receiver.env.dims, **callee_env.dims}
            callee_env.packs = {**receiver.env.packs, **callee_env.packs}
            callee_env.dtypes = {**receiver.env.dtypes, **callee_env.dtypes}
        return callee_env
