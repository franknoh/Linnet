"""Materialize Linnet models as PyTorch modules, and export PyTorch modules as Linnet."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from .export import ExportError, ExportResult, export_linnet
from .module import LinnetModule, bind_weights
from .plan import Plan, PlanError, compile_plan

__all__ = [
    "ExportError",
    "ExportResult",
    "LinnetModule",
    "Plan",
    "PlanError",
    "bind_weights",
    "compile_plan",
    "export_linnet",
    "load",
]


def load(
    source: str | Path,
    *,
    generics: Mapping[str, int | str],
    root: str | None = None,
    std_root: str | Path | None = None,
    weights: str | Path | None = None,
    bindings: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
    optimize: bool = True,
    numerics: str = "exact",
) -> LinnetModule:
    """Compiles a Linnet source file and returns its root block as a module.

    `generics` gives the root block's generic arguments by name. With
    `weights`, the parameters are loaded from SafeTensors and every name,
    shape, and dtype is checked before anything runs; otherwise they stay
    zero. Compilation runs `linnet plan`, which never executes the model.

    `numerics="equivalent"` lets PyTorch library calls stand in for the
    standard library's semantic operations; results then agree with the
    canonical definitions up to floating-point rounding.
    """
    plan = compile_plan(source, root=root, std_root=std_root, optimize=optimize, numerics=numerics)
    module = LinnetModule(plan, generics, torch.device(device))
    if weights is not None:
        bind_weights(module, weights, bindings, strict=strict)
    return module
