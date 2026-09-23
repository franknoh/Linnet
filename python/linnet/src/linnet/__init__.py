"""Linnet models in Python: plans, checkpoints, and a backend per framework.

`linnet.torch`, `linnet.jax`, and `linnet.onnx` each import their framework
on first use; this package itself needs only NumPy.
"""

from __future__ import annotations

from .compiler import LinnetError, find_compiler, run_compiler
from .plan import Env, Plan, PlanError, compile_plan
from .weights import apply_bindings, read_arrays, read_bindings, safetensors_files

__all__ = [
    "Env",
    "LinnetError",
    "Plan",
    "PlanError",
    "apply_bindings",
    "compile_plan",
    "find_compiler",
    "read_arrays",
    "read_bindings",
    "run_compiler",
    "safetensors_files",
]
