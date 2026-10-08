"""The compiler's plan: `linnet plan` compiles a source file into the
program it checked, which `linnet.ir.Program` holds, typed."""

from __future__ import annotations

from pathlib import Path

from . import ir
from .compiler import LinnetError, PlanError, check_numerics, run_compiler, std_arguments


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
    check_numerics(numerics)
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


__all__ = ["PlanError", "compile_plan"]
