"""Linnet scalar types as PyTorch dtypes."""

from __future__ import annotations

import torch

from .. import ir
from ..dtypes import BY_SAFETENSORS, DTYPES

TORCH_DTYPES: dict[str, torch.dtype] = {
    name: getattr(torch, info.torch) for name, info in DTYPES.items()
}

# PyTorch dtypes back to Linnet names, for binding a dtype generic from an input.
LINNET_DTYPES: dict[torch.dtype, str] = {dtype: name for name, dtype in TORCH_DTYPES.items()}

# PyTorch dtypes as SafeTensors names them, and back.
SAFETENSORS_NAMES: dict[torch.dtype, str] = {
    TORCH_DTYPES[name]: info.safetensors for name, info in DTYPES.items()
}
FROM_SAFETENSORS: dict[str, torch.dtype] = {
    code: TORCH_DTYPES[info.name] for code, info in BY_SAFETENSORS.items()
}


def torch_dtype(bindings: ir.Bindings, dtype: ir.DType) -> torch.dtype:
    """The PyTorch dtype of a plan dtype, its generics resolved by `bindings`."""
    return TORCH_DTYPES[bindings.dtype(dtype)]
