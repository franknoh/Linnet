"""PyTorch implementations of standard-library semantic operations.

Each entry is keyed by the implementation name the compiler selects in a plan
(`linnet explain` lists them). An implementation receives the operation's
arguments in declaration order and must agree with the canonical `.linnet`
body up to floating-point rounding; the differential tests hold it to that.

Names ending in `(input dtype)` are the `numerics="fast"` tier: the same
kernels without the f32 accumulation the canonical bodies specify, so
low-precision inputs may round differently.
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


def _layer_norm(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, weight, bias, eps = args
    normalized = functional.layer_norm(x.float(), [x.shape[-1]], eps=float(eps.item()))
    scaled = normalized.to(x.dtype) * weight
    return scaled if bias is None else scaled + bias


def _embedding(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    ids, table = args
    return functional.embedding(ids.long(), table)


def causal_mask(shape: list[int], device: torch.device) -> torch.Tensor:
    """`causal_mask<Q, K>()` as a boolean `tril`; a square one is tagged so the
    attention kernels can use `is_causal` instead of reading it."""
    rows, columns = shape
    mask = torch.ones(rows, columns, dtype=torch.bool, device=device).tril(columns - rows)
    mask._linnet_causal = rows == columns  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    return mask


def _sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    fast: bool,
) -> torch.Tensor:
    # A square mask produced by `causal_mask` becomes `is_causal=True`, which
    # lets PyTorch use its fused causal kernels; grouped key/value heads are
    # read once through `enable_gqa`.
    causal = mask is not None and bool(getattr(mask, "_linnet_causal", False))
    options: dict[str, Any] = {"scale": float(scale.item())}
    if causal:
        options["is_causal"] = True
    elif mask is not None:
        options["attn_mask"] = mask
    if key.shape[1] != query.shape[1]:
        options["enable_gqa"] = True
    if fast:
        return functional.scaled_dot_product_attention(query, key, value, **options)
    # The canonical body computes in f32; SDPA is asked to do the same so
    # that results agree with it for low-precision inputs.
    out = functional.scaled_dot_product_attention(
        query.float(), key.float(), value.float(), **options
    )
    return out.to(query.dtype)


def _attention(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    query, key, value, scale, mask = args
    return _sdpa(query, key, value, scale, mask, fast=False)


def _softmax_fast(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return torch.softmax(args[0], dim=-1)


def _rms_norm_fast(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, weight, eps = args
    return torch.rms_norm(x, [x.shape[-1]], eps=float(eps.item())) * weight


def _layer_norm_fast(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, weight, bias, eps = args
    scaled = functional.layer_norm(x, [x.shape[-1]], eps=float(eps.item())) * weight
    return scaled if bias is None else scaled + bias


def _attention_fast(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    query, key, value, scale, mask = args
    return _sdpa(query, key, value, scale, mask, fast=True)


NATIVE: dict[str, Native] = {
    "torch.nn.functional.embedding": _embedding,
    "torch.matmul": _matmul,
    "torch.nn.functional.linear": _linear,
    "torch.softmax": _softmax,
    "torch.relu": _relu,
    "torch.sigmoid": _sigmoid,
    "torch.nn.functional.silu": _silu,
    "torch.nn.functional.gelu(tanh)": _gelu,
    "torch.rms_norm": _rms_norm,
    "torch.nn.functional.layer_norm": _layer_norm,
    "torch.nn.functional.scaled_dot_product_attention": _attention,
    "torch.softmax(input dtype)": _softmax_fast,
    "torch.rms_norm(input dtype)": _rms_norm_fast,
    "torch.nn.functional.layer_norm(input dtype)": _layer_norm_fast,
    "torch.nn.functional.scaled_dot_product_attention(input dtype)": _attention_fast,
    "torch.nn.functional.scaled_dot_product_attention(enable_gqa)": _attention,
    "torch.nn.functional.scaled_dot_product_attention(enable_gqa)(input dtype)": _attention_fast,
}
