"""The compiler's plan: `linnet plan` compiles a source file into the
program it checked, which `linnet.ir.Program` holds, typed. An input's
actual shape lines up with the shape it declares here, binding the
generics it determines.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any

from . import ir
from .compiler import LinnetError, PlanError, run_compiler, std_arguments


def align_shape(
    units: Sequence[ir.Unit], shape: Sequence[int], name: str
) -> tuple[list[tuple[ir.Dim, int]], tuple[ir.Pack, list[int]] | None]:
    """Input `name`'s declared shape (`units`, with at most one shape pack,
    which covers whatever axes the others leave) lined up with its actual
    `shape`: each declared dimension with its size, and the pack with the
    sizes it covers."""
    packs = [i for i, unit in enumerate(units) if isinstance(unit, ir.Pack)]
    if len(packs) > 1:
        raise PlanError(f"input `{name}` has more than one shape pack")
    fixed = len(units) - len(packs)
    if (packs and len(shape) < fixed) or (not packs and len(shape) != fixed):
        expected = f"at least {fixed}" if packs else str(fixed)
        raise PlanError(f"input `{name}` has rank {len(shape)}, expected {expected}")
    if not packs:
        dims = [unit for unit in units if not isinstance(unit, ir.Pack)]
        return list(zip(dims, shape, strict=True)), None
    at, width = packs[0], len(shape) - fixed
    rest = [unit for unit in [*units[:at], *units[at + 1 :]] if not isinstance(unit, ir.Pack)]
    sizes = [*shape[:at], *shape[at + width :]]
    pack = units[at]
    assert isinstance(pack, ir.Pack)
    return list(zip(rest, sizes, strict=True)), (pack, list(shape[at : at + width]))


def bind_shape_names(
    units: Sequence[ir.Unit],
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
        if str(bindings.setdefault(unit.name, text)) != text:
            raise PlanError(f"input `{name}` disagrees on shape pack `{unit.name}`")
    for dim, size in dims:
        if isinstance(dim, ir.DimSymbol):
            if str(bindings.setdefault(dim.name, size)) != str(size):
                raise PlanError(
                    f"input `{name}` has size {size} where `{dim.name}` is {bindings[dim.name]}"
                )
        elif isinstance(dim, int) and dim != size:
            raise PlanError(f"input `{name}` has size {size} on an axis that must be {dim}")


def compile_plan(
    source: str | Path,
    root: str | None = None,
    std_root: str | Path | None = None,
    optimize: bool = True,
    numerics: str = "exact",
    functions: bool = False,
) -> ir.Program:
    """Runs `linnet plan` on a source file and returns the typed program.
    Checking never executes the model.

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
        return ir.Program.from_json(run_compiler(*command))
    except LinnetError as error:
        raise PlanError(str(error)) from None


__all__ = ["PlanError", "align_shape", "bind_shape_names", "compile_plan"]
