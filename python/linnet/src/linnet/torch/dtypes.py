"""Linnet scalar types as PyTorch dtypes."""

from __future__ import annotations

from typing import Any

import torch

from ..dtypes import BY_SAFETENSORS, DTYPES
from ..plan import Env

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


def torch_dtype(env: Env, spec: str | dict[str, Any]) -> torch.dtype:
    """The PyTorch dtype of a plan dtype, with dtype generics resolved by `env`."""
    return TORCH_DTYPES[env.dtype_name(spec)]
