"""The compiler's plan: a JSON document produced by `linnet plan`.

Dimensions in a plan are symbolic expressions. An `Env` binds the generic
parameters of one function or block instance and evaluates them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compiler import LinnetError, run_compiler, std_arguments

DimExpr = int | dict[str, Any]
ShapeUnit = DimExpr  # a dimension, or {"pack": id, "name": ...}

DTYPES: tuple[str, ...] = (
    "bool",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "f16",
    "bf16",
    "f32",
    "f64",
)


class PlanError(LinnetError):
    """A plan could not be produced, read, or instantiated."""


@dataclass
class Env:
    """Bindings of generic parameters, by the compiler's symbol ids."""

    dims: dict[int, int] = field(default_factory=dict[int, int])
    packs: dict[int, list[int]] = field(default_factory=dict[int, list[int]])
    dtypes: dict[int, str] = field(default_factory=dict[int, str])

    def dim(self, expr: DimExpr) -> int:
        if isinstance(expr, int):
            return expr
        if "sym" in expr:
            symbol = int(expr["sym"])
            if symbol not in self.dims:
                raise PlanError(f"dimension `{expr['name']}` is not bound")
            return self.dims[symbol]
        if "packsize" in expr:
            symbol = int(expr["packsize"])
            if symbol not in self.packs:
                raise PlanError(f"shape pack `{expr['name']}` is not bound")
            count = 1
            for size in self.packs[symbol]:
                count *= size
            return count
        args = [self.dim(arg) for arg in expr["args"]]
        op = expr["op"]
        if op == "add":
            return sum(args)
        if op == "mul":
            product = 1
            for arg in args:
                product *= arg
            return product
        if op == "floordiv":
            if args[1] == 0:
                raise PlanError("division by zero in a dimension")
            return args[0] // args[1]
        if op == "mod":
            if args[1] == 0:
                raise PlanError("division by zero in a dimension")
            return args[0] % args[1]
        if op == "min":
            return min(args)
        if op == "max":
            return max(args)
        raise PlanError(f"unknown dimension operator `{op}`")

    def shape(self, units: list[ShapeUnit]) -> list[int]:
        sizes: list[int] = []
        for unit in units:
            if isinstance(unit, dict) and "pack" in unit:
                symbol = int(unit["pack"])
                if symbol not in self.packs:
                    raise PlanError(f"shape pack `{unit['name']}` is not bound")
                sizes.extend(self.packs[symbol])
            else:
                sizes.append(self.dim(unit))
        return sizes

    def dtype_name(self, spec: str | dict[str, Any]) -> str:
        if isinstance(spec, str):
            return spec
        var = int(spec["var"])
        if var not in self.dtypes:
            raise PlanError(f"dtype `{spec['name']}` is not bound")
        return self.dtypes[var]

    def relation_holds(self, constraint: dict[str, Any]) -> bool:
        lhs = self.dim(constraint["lhs"])
        rhs = self.dim(constraint["rhs"])
        relation = constraint["relation"]
        return {
            "==": lhs == rhs,
            "!=": lhs != rhs,
            "<": lhs < rhs,
            "<=": lhs <= rhs,
            ">": lhs > rhs,
            ">=": lhs >= rhs,
        }[relation]


@dataclass
class Plan:
    """The plan document as dictionaries; `linnet.ir.Program` is the typed view."""

    root: dict[str, Any]
    manifest: list[dict[str, Any]]
    blocks: dict[str, dict[str, Any]]
    functions: dict[str, dict[str, Any]]
    text: str = field(default="", repr=False)

    @staticmethod
    def from_json(text: str) -> Plan:
        document = json.loads(text)
        if document.get("version") != 1:
            raise PlanError(f"unsupported plan version {document.get('version')!r}")
        return Plan(
            root=document["root"],
            manifest=document["manifest"],
            blocks=document["blocks"],
            functions={function["name"]: function for function in document["functions"]},
            text=text,
        )

    def entries_of(self, block: str) -> list[dict[str, Any]]:
        return [
            function
            for function in self.functions.values()
            if function["kind"] == "entry" and function["block"] == block
        ]

    def method(self, block: str, name: str) -> dict[str, Any]:
        for function in self.functions.values():
            if function["block"] == block and function["name"].endswith(f"::{block}.{name}"):
                return function
        raise PlanError(f"block `{block}` has no method `{name}`")


def compile_plan(
    source: str | Path,
    root: str | None = None,
    std_root: str | Path | None = None,
    optimize: bool = True,
    numerics: str = "exact",
) -> Plan:
    """Runs `linnet plan` on a source file. Checking never executes the model.

    With `optimize`, the compiler's exact canonicalization passes run first;
    they never change results. `numerics` is `"exact"` (every semantic
    operation runs its canonical decomposition), `"equivalent"` (PyTorch
    library calls replace the decompositions they agree with up to rounding),
    or `"fast"` (the same calls in the input dtype, without the f32
    accumulation the canonical bodies specify).
    """
    if numerics not in ("exact", "equivalent", "fast"):
        raise PlanError('numerics must be "exact", "equivalent", or "fast"')
    command = ["plan", "--numerics", numerics]
    if not optimize:
        command.append("--no-optimize")
    if root is not None:
        command += ["--root", root]
    command += [*std_arguments(std_root), str(source)]
    try:
        return Plan.from_json(run_compiler(*command))
    except LinnetError as error:
        raise PlanError(str(error)) from None
