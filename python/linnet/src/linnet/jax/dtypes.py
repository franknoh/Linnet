"""Linnet scalar types as the NumPy and JAX dtypes `linnet.jax` uses."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..dtypes import BY_MLIR, DTYPES

# Linnet names to NumPy scalar types; bf16 is JAX's (a class of its own, not
# a NumPy scalar type, so the values are typed as classes).
NUMPY_TYPES: dict[str, type[object]] = {
    name: jnp.bfloat16 if name == "bf16" else np.dtype(info.numpy).type
    for name, info in DTYPES.items()
}
# StableHLO element types (`i1`, `ui32`, `bf16`) to the same.
MLIR_TYPES: dict[str, type[object]] = {
    mlir: NUMPY_TYPES[info.name] for mlir, info in BY_MLIR.items()
}

__all__ = ["MLIR_TYPES", "NUMPY_TYPES"]
