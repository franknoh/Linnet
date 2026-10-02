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
    """The cache writes, heads as a full slice as the generated code writes
    them (see `linnet torch`)."""
    cache, value = args[0], args[1]
    if len(args) == 3:
        # `write_rows`: row b at at[b].
        rows = torch.arange(cache.shape[0], device=cache.device)
        return torch.ops.aten.index_put(cache, [rows, None, args[2].long()], value[:, :, 0])
    # `write_slot` (one row) or `write_slots` (rows `slots`): a span each.
    slot, at = args[2], args[3]
    rows = slot.long().reshape(1, 1) if slot.dim() == 0 else slot.long()[:, None]
    span = (at.long() + torch.arange(value.shape[2], device=cache.device))[None, :]
    return torch.ops.aten.index_put(cache, [rows, None, span], value.permute(0, 2, 1, 3))


def _write_tokens(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    """`write_tokens`: token p at (rows[p], positions[p]), every head."""
    cache, value, rows, positions = args
    return torch.ops.aten.index_put(
        cache, [rows.long(), None, positions.long()], value[0].permute(1, 0, 2)
    )


_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def mxfp4_experts(
    x: torch.Tensor, blocks: torch.Tensor, scales: torch.Tensor, experts: torch.Tensor
) -> torch.Tensor:
    """`std.quant::mxfp4_experts`: Linnet's kernel for a decoding step's few
    rows on CUDA (`linnet.torch.kernels`), the chosen experts unpacked and
    multiplied in f32 otherwise."""
    rows, chosen, _ = x.shape
    if x.is_cuda and rows * chosen <= 32:
        try:
            from .kernels import mxfp4_experts as kernel
        except ImportError:
            pass
        else:
            return kernel(x, blocks, scales, experts)
    table = torch.tensor(_FP4, dtype=torch.float32, device=x.device)
    taken = blocks[experts]
    values = torch.stack([table[(taken & 15).long()], table[(taken >> 4).long()]], dim=-1)
    factor = torch.exp2(scales[experts].float() - 127)[..., None]
    weight = (values.reshape(*taken.shape[:-1], 32) * factor).reshape(*taken.shape[:-2], -1)
    return torch.einsum("rki,rkoi->rko", x.float(), weight).to(x.dtype)


def _mxfp4_experts(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, blocks, scales, experts = args
    return mxfp4_experts(x, blocks, scales, experts)


def _mxfp4_experts_shared(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, blocks, scales, experts = args
    return mxfp4_experts(x.expand(-1, experts.shape[1], -1), blocks, scales, experts)


def _one_shard(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    """`std.nn.parallel::all_reduce` in one process, which holds the whole
    model: the sum over one shard is the value itself."""
    return args[0]


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


def convolution(args: list[Any], strides: list[int], pads: list[int]) -> torch.Tensor:
    """`std.nn.conv::conv1d`, `conv2d`, and `conv2d_rect` as `F.conv1d` and
    `F.conv2d`, with the call's own strides and padding, one per spatial
    axis. They are not recovered from the shapes: a 3x3 window taking 4
    positions to 2 fits both stride 1 without padding and stride 2 with one."""
    x, weight, bias = args
    convolve = functional.conv1d if len(strides) == 1 else functional.conv2d
    return convolve(x, weight, bias, stride=tuple(strides), padding=tuple(pads))


def group_norm(args: list[Any], groups: int) -> torch.Tensor:
    """`std.nn.norm::group_norm` as `F.group_norm`, with the call's own
    `Groups`: statistics in f32, as the body computes them."""
    x, weight, bias, eps = args
    return functional.group_norm(x, groups, weight, bias, float(eps.item()))


def max_pool2d(args: list[Any], window: int, stride: int, pad: int) -> torch.Tensor:
    """`std.nn.pool::max_pool2d` as `F.max_pool2d`, with the call's own
    geometry, as for the convolution."""
    return functional.max_pool2d(args[0], window, stride=stride, padding=pad)


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


def _index_select(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    x, order = args
    return x.index_select(-1, order.long())


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


def _int4_groups_linear(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    """`std.quant::linear_int4_groups`: dequantized, then `F.linear`."""
    x, packed, scale, zero, bias = args
    out_features, groups, half = packed.shape
    q = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(out_features, groups, 2 * half)
    weight = (q.float() - zero.float()[..., None]) * scale.float()[..., None]
    y = functional.linear(x, weight.reshape(out_features, groups * 2 * half).to(x.dtype))
    return y if bias is None else y + bias


def _grouped(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether `torch._grouped_mm` runs these: CUDA, Hopper or later, bf16."""
    return (
        hasattr(torch, "_grouped_mm")
        and x.is_cuda
        and x.dtype == weight.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(x.device)[0] >= 9
        and x.shape[-1] % 8 == 0
        and weight.shape[1] % 8 == 0
    )


def _linear_experts(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    """`std.nn.moe::linear_experts`: one grouped product over the rows sorted
    by expert on CUDA in bf16, each chosen expert's weight gathered elsewhere."""
    x, weight, experts = args
    rows, chosen, width = x.shape
    count, out_features, _ = weight.shape
    flat = experts.reshape(-1).long()
    if _grouped(x, weight):
        grouped = torch._grouped_mm  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
        order = flat.argsort(stable=True)
        slots = torch.arange(count, device=x.device)
        ends = (flat[None, :] <= slots[:, None]).sum(1).to(torch.int32)
        y = grouped(x.reshape(-1, width)[order], weight.transpose(-2, -1), offs=ends)
        unsorted = torch.empty_like(y).index_copy_(0, order, y)
        return unsorted.reshape(rows, chosen, out_features)
    taken = weight[flat].reshape(rows, chosen, out_features, width)
    return torch.einsum("rki,rkoi->rko", x.float(), taken.float()).to(x.dtype)


def _linear_experts_shared(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    """`std.nn.moe::linear_experts_shared`: the grouped product over the row
    repeated for each chosen expert, or every expert and the chosen kept."""
    x, weight, experts = args
    rows, chosen = experts.shape
    count, out_features, width = weight.shape
    if _grouped(x, weight):
        return _linear_experts([x[:, None, :].expand(rows, chosen, width), weight, experts], None)
    every = functional.linear(x, weight.reshape(count * out_features, width))
    every = every.reshape(rows, count, out_features)
    return every.gather(1, experts.long()[..., None].expand(rows, chosen, out_features))


def _combine_experts(args: list[Any], _result: torch.dtype | None) -> torch.Tensor:
    """`std.nn.moe::combine_experts`: the grouped product weighed and summed,
    or the weighed inputs placed by expert and multiplied by every expert."""
    x, weight, experts, weights = args
    rows, chosen, width = x.shape
    count, out_features, _ = weight.shape
    if _grouped(x, weight):
        y = _linear_experts([x, weight, experts], None).float() * weights.float()[..., None]
        return y.sum(1).to(x.dtype)
    placed = torch.zeros(rows, count, width, dtype=x.dtype, device=x.device)
    placed.scatter_add_(
        1, experts.long()[..., None].expand(rows, chosen, width), weights[..., None] * x
    )
    joined = weight.permute(1, 0, 2).reshape(out_features, count * width)
    return functional.linear(placed.reshape(rows, count * width), joined)


NATIVE: dict[str, Native] = {
    "torch.ops.aten._weight_int4pack_mm": _int4_groups_linear,
    "torch._grouped_mm": _linear_experts,
    "torch._grouped_mm(shared)": _linear_experts_shared,
    "torch._grouped_mm(combined)": _combine_experts,
    "torch.nn.functional.embedding": _embedding,
    "torch.index_select": _index_select,
    "torch.nn.functional.batch_norm": _batch_norm,
    "torch.Tensor.mean": _spatial_mean,
    "torch.Tensor.index_copy": _index_copy,
    "torch.Tensor.index_put": _index_put,
    "torch.Tensor.index_put(tokens)": _write_tokens,
    "torch.distributed.all_reduce": _one_shard,
    "linnet.mxfp4_experts": _mxfp4_experts,
    "linnet.mxfp4_experts(shared)": _mxfp4_experts_shared,
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
