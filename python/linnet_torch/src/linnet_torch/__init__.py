"""Materialize Linnet models as PyTorch modules."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from .module import LinnetModule, bind_weights
from .plan import Plan, PlanError, compile_plan

__all__ = ["LinnetModule", "Plan", "PlanError", "bind_weights", "compile_plan", "load"]


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
) -> LinnetModule:
    """Compiles a Linnet source file and returns its root block as a module.

    `generics` gives the root block's generic arguments by name. With
    `weights`, the parameters are loaded from SafeTensors and every name,
    shape, and dtype is checked before anything runs; otherwise they stay
    zero. Compilation runs `linnet plan`, which never executes the model.
    """
    plan = compile_plan(source, root=root, std_root=std_root, optimize=optimize)
    module = LinnetModule(plan, generics, torch.device(device))
    if weights is not None:
        bind_weights(module, weights, bindings, strict=strict)
    return module
