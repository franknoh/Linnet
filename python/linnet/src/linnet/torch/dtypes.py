"""Linnet scalar types as PyTorch dtypes."""

from __future__ import annotations

import torch

from .. import ir
from ..dtypes import DTYPES

TORCH_DTYPES: dict[str, torch.dtype] = {
    name: getattr(torch, info.torch) for name, info in DTYPES.items()
}

# PyTorch dtypes back to Linnet names, for binding a dtype generic from an
# input: a packed format is held as `uint8`, which names `u8`.
LINNET_DTYPES: dict[torch.dtype, str] = {
    TORCH_DTYPES[name]: name for name, info in DTYPES.items() if not info.packed
}

# PyTorch dtypes as SafeTensors names them.
SAFETENSORS_NAMES: dict[torch.dtype, str] = {
    TORCH_DTYPES[name]: info.safetensors for name, info in DTYPES.items() if not info.packed
}


def torch_dtype(bindings: ir.Bindings, dtype: ir.DType) -> torch.dtype:
    """The PyTorch dtype of a plan dtype, its generics resolved by `bindings`."""
    return TORCH_DTYPES[bindings.dtype(dtype)]


def torch_shape(bindings: ir.Bindings, tensor: ir.TensorType) -> list[int]:
    """The shape PyTorch holds a tensor of `tensor`'s type with: its last
    axis halved for a packed format, whose bytes it holds."""
    return list(DTYPES[bindings.dtype(tensor.dtype)].storage_shape(bindings.shape(tensor.shape)))
