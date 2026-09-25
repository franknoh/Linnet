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


def _index_copy(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    cache, value, at = args
    # One position when decoding, a span of them when prefilling.
    positions = at.reshape(1).long() + torch.arange(value.shape[2], device=cache.device)
    return cache.index_copy(2, positions, value)


def _index_put(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    cache, value = args[0], args[1]
    batch, heads = cache.shape[0], cache.shape[1]
    rows = torch.arange(batch, device=cache.device)
    lanes = torch.arange(heads, device=cache.device)
    if len(args) == 3:
        # `write_rows`: row b at at[b].
        return cache.index_put(
            (rows[:, None], lanes[None, :], args[2].long()[:, None]), value[:, :, 0]
        )
    # `write_slot`: one row, a span of positions.
    slot, at = args[2], args[3]
    span = at.long() + torch.arange(value.shape[2], device=cache.device)
    return cache.index_put((slot.long().reshape(1, 1), lanes[:, None], span[None, :]), value[0])


def _matmul(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return torch.matmul(args[0], args[1])


def _linear(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, weight, bias = args
    return functional.linear(x, weight, bias)


# `torch.softmax` and `torch.rms_norm` accumulate in f32 for f16 and bf16
# inputs themselves, so these results are bit-identical to the canonical
# body's explicit casts (`tests/torch/test_native.py` checks it).
def _softmax(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return torch.softmax(args[0], dim=-1)


def _relu(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return torch.relu(args[0])


def _sigmoid(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return torch.sigmoid(args[0])


def _silu(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return functional.silu(args[0])


def _gelu(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return functional.gelu(args[0], approximate="tanh")


def _gelu_erf(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    return functional.gelu(args[0], approximate="none")


def _rms_norm(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, weight, eps = args
    return torch.rms_norm(x, [x.shape[-1]], eps=float(eps.item())) * weight


def _layer_norm(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, weight, bias, eps = args
    normalized = functional.layer_norm(x.float(), [x.shape[-1]], eps=float(eps.item()))
    scaled = normalized.to(x.dtype) * weight
    return scaled if bias is None else scaled + bias


def conv2d(args: list[Any], stride: int, pad: int) -> torch.Tensor:
    """`std.nn.conv::conv2d` as `F.conv2d`, with the call's own `Stride` and
    `Pad`. They are not recovered from the shapes: a 3x3 window taking 4
    positions to 2 fits both stride 1 without padding and stride 2 with one."""
    x, weight, bias = args
    return functional.conv2d(x, weight, bias, stride=stride, padding=pad)


def upsample_nearest2d(args: list[Any], shape: list[int]) -> torch.Tensor:
    """`std.nn.resize::upsample_nearest2d` as `F.interpolate`; the scale is
    the ratio of the shapes, as the exporters recover it."""
    x = args[0]
    return functional.interpolate(x, scale_factor=shape[2] // int(x.shape[2]), mode="nearest")


def _batch_norm(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, mean, variance, weight, bias, eps = args
    return functional.batch_norm(x, mean, variance, weight, bias, False, 0.0, float(eps.item()))


def _spatial_mean(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x = args[0]
    return x.mean(dim=(2, 3), dtype=torch.float32).to(x.dtype)


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
        # A mask per sequence ([B, Q, K]) broadcasts over the heads.
        options["attn_mask"] = mask.unsqueeze(1) if mask.dim() == 3 else mask
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
    "torch.nn.functional.batch_norm": _batch_norm,
    "torch.Tensor.mean": _spatial_mean,
    "torch.Tensor.index_copy": _index_copy,
    "torch.Tensor.index_put": _index_put,
    "torch.matmul": _matmul,
    "torch.nn.functional.linear": _linear,
    "torch.softmax": _softmax,
    "torch.relu": _relu,
    "torch.sigmoid": _sigmoid,
    "torch.nn.functional.silu": _silu,
    "torch.nn.functional.gelu(tanh)": _gelu,
    "torch.nn.functional.gelu": _gelu_erf,
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
