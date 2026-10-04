"""Linnet models in Python: plans, checkpoints, and a backend per framework.

`load_program` compiles a source file and returns the typed `Program`
(`linnet.ir`); `linnet.diagram` draws it. `linnet.torch`, `linnet.jax`, and
`linnet.onnx` each import their framework on first use; this package itself
needs only NumPy.
"""

from __future__ import annotations

from importlib import metadata

from .compiler import LinnetError, find_compiler, run_compiler
from .ir import Program, load_program
from .plan import Env, Plan, PlanError, compile_plan
from .weights import (
    RawTensor,
    apply_bindings,
    iter_safetensors,
    read_arrays,
    read_bindings,
    safetensors_files,
)

try:
    __version__ = metadata.version("linnet-lang")
except metadata.PackageNotFoundError:  # imported from a checkout without installing
    __version__ = "0.0.0"

__all__ = [
    "Env",
    "LinnetError",
    "Plan",
    "PlanError",
    "Program",
    "RawTensor",
    "apply_bindings",
    "compile_plan",
    "find_compiler",
    "iter_safetensors",
    "load_program",
    "read_arrays",
    "read_bindings",
    "run_compiler",
    "safetensors_files",
]
