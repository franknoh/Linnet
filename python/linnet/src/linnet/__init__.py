"""Linnet models in Python: plans, checkpoints, and a backend per framework.

`load_program` compiles a source file and returns the typed `Program`
(`linnet.ir`); `linnet.diagram` draws it. `linnet.torch`, `linnet.jax`, and
`linnet.onnx` each import their framework on first use; this package itself
needs only NumPy.
"""

from __future__ import annotations

from .compiler import LinnetError, find_compiler, run_compiler
from .ir import Program, load_program
from .plan import Env, Plan, PlanError, compile_plan
from .weights import apply_bindings, read_arrays, read_bindings, safetensors_files

__all__ = [
    "Env",
    "LinnetError",
    "Plan",
    "PlanError",
    "Program",
    "apply_bindings",
    "compile_plan",
    "find_compiler",
    "load_program",
    "read_arrays",
    "read_bindings",
    "run_compiler",
    "safetensors_files",
]
