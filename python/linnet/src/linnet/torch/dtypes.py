"""Linnet scalar types as PyTorch dtypes."""

from __future__ import annotations

from typing import Any

import torch

from ..plan import Env

TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "u8": torch.uint8,
    "u16": torch.uint16,
    "u32": torch.uint32,
    "u64": torch.uint64,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}


def torch_dtype(env: Env, spec: str | dict[str, Any]) -> torch.dtype:
    """The PyTorch dtype of a plan dtype, with dtype generics resolved by `env`."""
    return TORCH_DTYPES[env.dtype_name(spec)]
