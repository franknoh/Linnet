"""PyTorch implementations of standard-library semantic operations.

Each entry is keyed by the implementation name the compiler selects in a plan
(`linnet explain` lists them). An implementation receives the operation's
arguments in declaration order and must agree with the canonical `.linnet`
body up to floating-point rounding; the differential tests hold it to that.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as functional

Native = Callable[[list[Any], torch.dtype | None], Any]


def _matmul(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return torch.matmul(args[0], args[1])


def _linear(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, weight, bias = args
    return functional.linear(x, weight, bias)


def _softmax(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x = args[0]
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def _relu(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return torch.relu(args[0])


def _sigmoid(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return torch.sigmoid(args[0])


def _silu(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return functional.silu(args[0])


def _gelu(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return functional.gelu(args[0], approximate="tanh")


def _rms_norm(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, weight, eps = args
    normalized = torch.rms_norm(x.float(), [x.shape[-1]], eps=float(eps.item()))
    return normalized.to(x.dtype) * weight


def _attention(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    query, key, value, scale, mask = args
    # The canonical body computes in f32; SDPA is asked to do the same so
    # that results agree with it for low-precision inputs.
    out = functional.scaled_dot_product_attention(
        query.float(),
        key.float(),
        value.float(),
        attn_mask=None if mask is None else mask,
        scale=float(scale.item()),
    )
    return out.to(query.dtype)


NATIVE: dict[str, Native] = {
    "torch.matmul": _matmul,
    "torch.nn.functional.linear": _linear,
    "torch.softmax": _softmax,
    "torch.relu": _relu,
    "torch.sigmoid": _sigmoid,
    "torch.nn.functional.silu": _silu,
    "torch.nn.functional.gelu(tanh)": _gelu,
    "torch.rms_norm": _rms_norm,
    "torch.nn.functional.scaled_dot_product_attention": _attention,
}
