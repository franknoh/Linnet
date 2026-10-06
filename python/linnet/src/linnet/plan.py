"""The compiler's plan: a JSON document produced by `linnet plan`.

Dimensions in a plan are symbolic expressions. An `Env` binds the generic
parameters of one function or block instance and evaluates them.
"""

from __future__ import annotations

import json
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .compiler import LinnetError, run_compiler, std_arguments

DimExpr = int | dict[str, Any]
ShapeUnit = DimExpr  # a dimension, or {"pack": id, "name": ...}


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
        lhs, rhs = self.dim(constraint["lhs"]), self.dim(constraint["rhs"])
        return holds(constraint["relation"], lhs, rhs)


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


def align_shape(
    units: Sequence[ShapeUnit], shape: Sequence[int], name: str
) -> tuple[list[tuple[ShapeUnit, int]], tuple[dict[str, Any], list[int]] | None]:
    """Input `name`'s declared shape (`units`, with at most one shape pack,
    which covers whatever axes the others leave) lined up with its actual
    `shape`: each declared dimension with its size, and the pack with the
    sizes it covers."""
    packs = [i for i, unit in enumerate(units) if isinstance(unit, dict) and "pack" in unit]
    if len(packs) > 1:
        raise PlanError(f"input `{name}` has more than one shape pack")
    fixed = len(units) - len(packs)
    if (packs and len(shape) < fixed) or (not packs and len(shape) != fixed):
        expected = f"at least {fixed}" if packs else str(fixed)
        raise PlanError(f"input `{name}` has rank {len(shape)}, expected {expected}")
    if not packs:
        return list(zip(units, shape, strict=True)), None
    at, width = packs[0], len(shape) - fixed
    rest = [*units[:at], *units[at + 1 :]]
    sizes = [*shape[:at], *shape[at + width :]]
    pack = cast(dict[str, Any], units[at])
    return list(zip(rest, sizes, strict=True)), (pack, list(shape[at : at + width]))


def bind_shape_names(
    units: Sequence[ShapeUnit],
    shape: Sequence[int],
    name: str,
    bindings: MutableMapping[str, Any],
) -> None:
    """Binds the generics of input `name`'s declared shape from its actual
    `shape`, by name (a shape pack as `"2,3"`, as `--bind` takes it), and
    checks the sizes already bound or fixed."""
    dims, pack = align_shape(units, shape, name)
    if pack is not None:
        unit, sizes = pack
        text = ",".join(map(str, sizes))
        if str(bindings.setdefault(str(unit["name"]), text)) != text:
            raise PlanError(f"input `{name}` disagrees on shape pack `{unit['name']}`")
    for unit, size in dims:
        if isinstance(unit, dict) and "sym" in unit:
            symbol = str(unit["name"])
            if str(bindings.setdefault(symbol, size)) != str(size):
                raise PlanError(
                    f"input `{name}` has size {size} where `{symbol}` is {bindings[symbol]}"
                )
        elif isinstance(unit, int) and unit != size:
            raise PlanError(f"input `{name}` has size {size} on an axis that must be {unit}")


@dataclass
class Plan:
    """The plan document as dictionaries; `linnet.ir.Program` is the typed view.

    A plan of functions (`linnet plan --functions`) has no root block: `root`
    is empty and so is the manifest."""

    root: dict[str, Any]
    manifest: list[dict[str, Any]]
    blocks: dict[str, dict[str, Any]]
    functions: dict[str, dict[str, Any]]
    text: str = field(default="", repr=False)
    module: str = ""  # the root file's module path

    @staticmethod
    def from_json(text: str) -> Plan:
        document = json.loads(text)
        if document.get("version") != 1:
            raise PlanError(f"unsupported plan version {document.get('version')!r}")
        return Plan(
            root=document["root"] or {},
            manifest=document["manifest"],
            blocks=document["blocks"],
            functions={function["name"]: function for function in document["functions"]},
            text=text,
            module=document["module"],
        )

    def module_entries(self) -> dict[str, dict[str, Any]]:
        """The entries declared at module level in the root file, by name:
        functions of their inputs alone."""
        prefix = f"{self.module}::"
        return {
            function["name"].removeprefix(prefix): function
            for function in self.functions.values()
            if function["kind"] == "entry"
            and function["block"] is None
            and function["name"].startswith(prefix)
        }

    def module_entry(self, name: str | None) -> dict[str, Any]:
        """The module-level entry called `name`, or the only one."""
        entries = self.module_entries()
        if name is None:
            if len(entries) != 1:
                listed = ", ".join(f"`{entry}`" for entry in entries) or "none"
                raise PlanError(
                    f"the module has {len(entries)} module-level entries ({listed}); name one"
                )
            return next(iter(entries.values()))
        if name not in entries:
            listed = ", ".join(f"`{entry}`" for entry in entries) or "none"
            raise PlanError(f"no module-level entry `{name}` (there are: {listed})")
        return entries[name]

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
    functions: bool = False,
) -> Plan:
    """Runs `linnet plan` on a source file. Checking never executes the model.

    With `functions`, the plan is of the module-level entries, with no root
    block (`linnet plan --functions`).

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
    if functions:
        command.append("--functions")
    command += [*std_arguments(std_root), str(source)]
    try:
        return Plan.from_json(run_compiler(*command))
    except LinnetError as error:
        raise PlanError(str(error)) from None
