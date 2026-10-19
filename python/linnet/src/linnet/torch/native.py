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

from collections.abc import Callable, Sequence
from typing import TypedDict, cast

import torch
import torch.nn.functional as functional

# An implementation's arguments: the operation's, in declaration order, as
# tensors (scalars as 0-d ones). An optional one left out (a bias, a mask) is
# `None`, which only the implementations of operations that have one see:
# they take `OptionalArgs`.
Args = Sequence[torch.Tensor]
OptionalArgs = Sequence[torch.Tensor | None]
# An implementation: the arguments to the result.
Native = Callable[[OptionalArgs], torch.Tensor]


def _required(native: Callable[[Args], torch.Tensor]) -> Native:
    """`native`, for an operation without optional arguments: none is `None`."""

    def run(args: OptionalArgs) -> torch.Tensor:
        present = [arg for arg in args if arg is not None]
        assert len(present) == len(args)
        return native(present)

    return run


# The arguments of the operations that take an optional one, by position.
_Biased = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]  # x, weight, bias
_Norm = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]  # x, w, b, eps
# query, key, value, scale, mask
_Masked = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
# x, packed, scale, zero, bias
_Int4Linear = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]


def _index_copy(args: Args) -> torch.Tensor:
    cache, value, at = args
    # One position when decoding, a span of them when prefilling.
    positions = at.reshape(1).long() + torch.arange(value.shape[2], device=cache.device)
    return cache.index_copy(2, positions, value)


def _index_put(args: Args) -> torch.Tensor:
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


def _write_tokens(args: Args) -> torch.Tensor:
    """`write_tokens`: token p at (rows[p], positions[p]), every head."""
    cache, value, rows, positions = args
    return torch.ops.aten.index_put(
        cache, [rows.long(), None, positions.long()], value[0].permute(1, 0, 2)
    )


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
    from .moe import mxfp4_values

    weight = mxfp4_values(blocks[experts], scales[experts])
    return torch.einsum("rki,rkoi->rko", x.float(), weight).to(x.dtype)


def _mxfp4_experts(args: Args) -> torch.Tensor:
    x, blocks, scales, experts = args
    return mxfp4_experts(x, blocks, scales, experts)


def _mxfp4_experts_shared(args: Args) -> torch.Tensor:
    x, blocks, scales, experts = args
    return mxfp4_experts(x.expand(-1, experts.shape[1], -1), blocks, scales, experts)


def mxfp4_grouped(
    x: torch.Tensor,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    experts: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """`std.quant::mxfp4_linear_experts_shared` (no `weights`) and
    `mxfp4_combine_experts`: `linnet.torch.moe` on a Hopper GPU in bf16
    (each expert times the rows that chose it), every expert dequantized
    and multiplied in the bodies' dense form otherwise."""
    rows, chosen = experts.shape
    count, out_features = blocks.shape[0], blocks.shape[1]
    shared = weights is None
    from .moe import available, mxfp4_values
    from .moe import mxfp4_grouped as grouped

    if available(x):
        y = grouped(x, blocks, scales, experts, shared)
        if weights is None:
            return y
        return (y.float() * weights.float()[..., None]).sum(1).to(x.dtype)
    weight = mxfp4_values(blocks, scales).to(x.dtype)
    width = weight.shape[-1]
    if weights is None:
        every = functional.linear(x, weight.reshape(count * out_features, width))
        every = every.reshape(rows, count, out_features)
        return every.gather(1, experts[..., None].expand(rows, chosen, out_features))
    placed = torch.zeros(rows, count, width, dtype=x.dtype, device=x.device)
    placed.scatter_add_(1, experts[..., None].expand(rows, chosen, width), weights[..., None] * x)
    joined = weight.permute(1, 0, 2).reshape(out_features, count * width)
    return functional.linear(placed.reshape(rows, count * width), joined)


def _mxfp4_grouped_shared(args: Args) -> torch.Tensor:
    x, blocks, scales, experts = args
    return mxfp4_grouped(x, blocks, scales, experts)


def _mxfp4_grouped_combine(args: Args) -> torch.Tensor:
    x, blocks, scales, experts, weights = args
    return mxfp4_grouped(x, blocks, scales, experts, weights)


def _linear_cross_entropy(args: Args) -> torch.Tensor:
    from .loss import linear_cross_entropy

    hidden, weight, targets, weights = args
    return linear_cross_entropy(hidden, weight, targets, weights)


def _linear_token_log_probs(args: Args) -> torch.Tensor:
    from .loss import linear_token_log_probs

    hidden, weight, targets = args
    return linear_token_log_probs(hidden, weight, targets)


def _one_shard(args: Args) -> torch.Tensor:
    """`std.nn.parallel::all_reduce` and `all_gather` in one process, which
    holds the whole model: one shard's sum, or its slices side by side, is the
    value itself."""
    return args[0]


def _matmul(args: Args) -> torch.Tensor:
    return torch.matmul(args[0], args[1])


def _linear(args: OptionalArgs) -> torch.Tensor:
    x, weight, bias = cast(_Biased, args)
    return functional.linear(x, weight, bias)


# `torch.softmax` and `torch.rms_norm` accumulate in f32 for f16 and bf16
# inputs themselves, so these results are bit-identical to the canonical
# body's explicit casts (`tests/torch/test_native.py` checks it).
def _softmax(args: Args) -> torch.Tensor:
    return torch.softmax(args[0], dim=-1)


def _relu(args: Args) -> torch.Tensor:
    return torch.relu(args[0])


def _sigmoid(args: Args) -> torch.Tensor:
    return torch.sigmoid(args[0])


def _silu(args: Args) -> torch.Tensor:
    return functional.silu(args[0])


def _gelu(args: Args) -> torch.Tensor:
    return functional.gelu(args[0], approximate="tanh")


def _gelu_erf(args: Args) -> torch.Tensor:
    return functional.gelu(args[0], approximate="none")


def _rms_norm(args: Args) -> torch.Tensor:
    x, weight, eps = args
    return torch.rms_norm(x, [x.shape[-1]], eps=float(eps.item())) * weight


def _layer_norm(args: OptionalArgs) -> torch.Tensor:
    x, weight, bias, eps = cast(_Norm, args)
    normalized = functional.layer_norm(x.float(), [x.shape[-1]], eps=float(eps.item()))
    scaled = normalized.to(x.dtype) * weight
    return scaled if bias is None else scaled + bias


def convolution(args: OptionalArgs, strides: list[int], pads: list[int]) -> torch.Tensor:
    """`std.nn.conv::conv1d`, `conv2d`, and `conv2d_rect` as `F.conv1d` and
    `F.conv2d`, with the call's own strides and padding, one per spatial
    axis. They are not recovered from the shapes: a 3x3 window taking 4
    positions to 2 fits both stride 1 without padding and stride 2 with one."""
    x, weight, bias = cast(_Biased, args)
    convolve = functional.conv1d if len(strides) == 1 else functional.conv2d
    return convolve(x, weight, bias, stride=tuple(strides), padding=tuple(pads))


def group_norm(args: Args, groups: int) -> torch.Tensor:
    """`std.nn.norm::group_norm` as `F.group_norm`, with the call's own
    `Groups`: statistics in f32, as the body computes them."""
    x, weight, bias, eps = args
    return functional.group_norm(x, groups, weight, bias, float(eps.item()))


def max_pool2d(args: Args, window: int, stride: int, pad: int) -> torch.Tensor:
    """`std.nn.pool::max_pool2d` as `F.max_pool2d`, with the call's own
    geometry, as for the convolution."""
    return functional.max_pool2d(args[0], window, stride=stride, padding=pad)


def upsample_nearest2d(args: Args, shape: list[int]) -> torch.Tensor:
    """`std.nn.resize::upsample_nearest2d` as `F.interpolate`; the scale is
    the ratio of the shapes, as the exporters recover it."""
    x = args[0]
    return functional.interpolate(x, scale_factor=shape[2] // int(x.shape[2]), mode="nearest")


def _batch_norm(args: Args) -> torch.Tensor:
    x, mean, variance, weight, bias, eps = args
    return functional.batch_norm(x, mean, variance, weight, bias, False, 0.0, float(eps.item()))


def _spatial_mean(args: Args) -> torch.Tensor:
    x = args[0]
    return x.mean(dim=(2, 3), dtype=torch.float32).to(x.dtype)


def _embedding(args: Args) -> torch.Tensor:
    ids, table = args
    return functional.embedding(ids.long(), table)


def _index_select(args: Args) -> torch.Tensor:
    x, order = args
    return x.index_select(-1, order.long())


def causal_mask(shape: list[int], device: torch.device) -> torch.Tensor:
    """`causal_mask<Q, K>()` as a boolean `tril`; a square one is tagged so the
    attention kernels can use `is_causal` instead of reading it."""
    rows, columns = shape
    mask = torch.ones(rows, columns, dtype=torch.bool, device=device).tril(columns - rows)
    mask._linnet_causal = rows == columns  # pyright: ignore[reportAttributeAccessIssue]
    return mask


class _SdpaOptions(TypedDict, total=False):
    """The keywords `_sdpa` passes `F.scaled_dot_product_attention`."""

    scale: float
    is_causal: bool
    attn_mask: torch.Tensor
    enable_gqa: bool


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
    options: _SdpaOptions = {"scale": float(scale.item())}
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


def _attention(args: OptionalArgs) -> torch.Tensor:
    query, key, value, scale, mask = cast(_Masked, args)
    return _sdpa(query, key, value, scale, mask, fast=False)


def _sink_attention(args: Args) -> torch.Tensor:
    """`std.nn.attention::sink_attention`: two products around a softmax that
    also counts each query head's sink logit, whose share is then dropped."""
    query, key, value, sinks, scale, mask = args
    batch, heads, queries, width = query.shape
    kv_heads, keys = key.shape[1], key.shape[2]
    group = heads // kv_heads
    rows = mask.reshape(-1, queries, keys)
    grouped = query.reshape(batch, kv_heads, group * queries, width)
    scores = torch.matmul(grouped, key.transpose(-1, -2)).float() * float(scale)
    scores = scores.reshape(batch, kv_heads, group, queries, keys)
    scores = scores.masked_fill(~rows.reshape(rows.shape[0], 1, 1, queries, keys), -1e30)
    sink = sinks.float().reshape(1, kv_heads, group, 1, 1)
    peak = torch.maximum(scores.amax(-1, keepdim=True), sink)
    shifted = torch.exp(scores - peak)
    total = shifted.sum(-1, keepdim=True) + torch.exp(sink - peak)
    weights = (shifted / total).to(value.dtype).reshape(batch, kv_heads, group * queries, keys)
    return torch.matmul(weights, value).reshape(batch, heads, queries, width)


def _gated_delta_rule(args: Args) -> torch.Tensor:
    """`std.nn.recurrent::gated_delta_rule`: the chunked kernel
    (`linnet.torch.delta`)."""
    from .delta import gated_delta_rule

    query, key, value, decay, beta, initial, scale = args
    return gated_delta_rule(query, key, value, decay, beta, initial, float(scale.item()))


def _gated_delta_update(args: Args) -> torch.Tensor:
    """`std.nn.recurrent::gated_delta_update`: one pass over the state."""
    from .delta import gated_delta_update

    key, value, decay, beta, state = args
    return gated_delta_update(state, key, value, decay, beta)


def _gated_delta_rule_packed(args: Args) -> torch.Tensor:
    """`std.nn.recurrent::gated_delta_rule_packed`: the prompts as sequences
    of their own lengths (`linnet.torch.delta`)."""
    from .delta import gated_delta_rule_packed

    *tensors, scale = args
    return gated_delta_rule_packed(*tensors, float(scale.item()))


def _causal_conv_packed(args: Args) -> torch.Tensor:
    """`std.nn.recurrent::causal_conv_packed`: each tap shifted and masked."""
    from .delta import causal_conv_packed

    x, segments, weight = args
    return causal_conv_packed(x, segments, weight)


def _causal_history_packed(args: Args) -> torch.Tensor:
    """`std.nn.recurrent::causal_history_packed`: the rows' windows gathered."""
    from .delta import causal_history_packed

    x, segments, rows, history = args
    return causal_history_packed(x, segments, rows, history)


def _causal_conv(args: Args) -> torch.Tensor:
    """`std.nn.recurrent::causal_conv`: one grouped `F.conv1d`."""
    from .delta import causal_conv

    x, history, weight = args
    return causal_conv(x, history, weight)


def _layer_norm_fast(args: OptionalArgs) -> torch.Tensor:
    x, weight, bias, eps = cast(_Norm, args)
    scaled = functional.layer_norm(x, [x.shape[-1]], eps=float(eps.item())) * weight
    return scaled if bias is None else scaled + bias


def _attention_fast(args: OptionalArgs) -> torch.Tensor:
    query, key, value, scale, mask = cast(_Masked, args)
    return _sdpa(query, key, value, scale, mask, fast=True)


def _int4_groups_linear(args: OptionalArgs) -> torch.Tensor:
    """`std.quant::linear_int4_groups`: dequantized, then `F.linear`."""
    x, packed, scale, zero, bias = cast(_Int4Linear, args)
    out_features, groups, half = packed.shape
    q = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(out_features, groups, 2 * half)
    weight = (q.float() - zero.float()[..., None]) * scale.float()[..., None]
    y = functional.linear(x, weight.reshape(out_features, groups * 2 * half).to(x.dtype))
    return y if bias is None else y + bias


def _decode_fp8(args: OptionalArgs) -> torch.Tensor:
    """`std.quant::decode_fp8`: the bytes as `float8_e4m3fn`, widened."""
    (bits,) = cast(tuple[torch.Tensor], args)
    return bits.view(torch.float8_e4m3fn).float()


# x, weight, scales, bias
_EightBit = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]


def _fp8_linear(rounded: int) -> Native:
    """`std.quant::linear_fp8` as the generated code runs it
    (`linnet.torch.quantized`); `rounded` rounds a call's input to FP8 too."""

    def run(args: OptionalArgs) -> torch.Tensor:
        from . import quantized  # imports Triton where there is one

        x, weight, scale, bias = cast(_EightBit, args)
        return quantized.fp8_linear(x, quantized.fp8_operands(weight, scale, rounded), bias)

    return run


def _fp8_block_linear(rounded: int) -> Native:
    """`std.quant::linear_fp8_blocks`, as `_fp8_linear`."""

    def run(args: OptionalArgs) -> torch.Tensor:
        from . import quantized

        x, weight, scale, bias = cast(_EightBit, args)
        operands = quantized.fp8_block_operands(weight, scale, rounded)
        return quantized.fp8_block_linear(x, operands, bias)

    return run


def _int8_linear(rounded: int) -> Native:
    """`std.quant::linear_int8`, as `_fp8_linear`."""

    def run(args: OptionalArgs) -> torch.Tensor:
        from . import quantized

        x, weight, scale, bias = cast(_EightBit, args)
        return quantized.int8_linear(x, quantized.int8_operands(weight, scale, rounded), bias)

    return run


def _int8_group_linear(args: OptionalArgs) -> torch.Tensor:
    """`std.quant::linear_int8_groups`, as `_fp8_linear`."""
    from . import quantized

    x, weight, scale, bias = cast(_EightBit, args)
    return quantized.int8_group_linear(x, quantized.int8_group_operands(weight, scale), bias)


def _grouped(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether `torch._grouped_mm` runs these: bf16 on a GPU it has a kernel
    for (`moe.grouped_mm_native`)."""
    from .moe import grouped_mm_native

    return (
        x.is_cuda
        and x.dtype == weight.dtype == torch.bfloat16
        and grouped_mm_native(x.device)
        and x.shape[-1] % 8 == 0
        and weight.shape[1] % 8 == 0
    )


def _linear_experts(args: Args) -> torch.Tensor:
    """`std.nn.moe::linear_experts`: one grouped product over the rows sorted
    by expert on CUDA in bf16, each chosen expert's weight gathered elsewhere."""
    x, weight, experts = args
    rows, chosen, width = x.shape
    count, out_features, _ = weight.shape
    flat = experts.reshape(-1).long()
    from . import moe

    if moe.bf16_available(x, weight):
        return moe.bf16_experts(x, weight, experts, False)
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


def _linear_experts_shared(args: Args) -> torch.Tensor:
    """`std.nn.moe::linear_experts_shared`: the grouped product over the row
    repeated for each chosen expert, or every expert and the chosen kept."""
    x, weight, experts = args
    rows, chosen = experts.shape
    count, out_features, width = weight.shape
    from . import moe

    if moe.bf16_available(x, weight):
        return moe.bf16_experts(x, weight, experts, True)
    if _grouped(x, weight):
        return _linear_experts([x[:, None, :].expand(rows, chosen, width), weight, experts])
    every = functional.linear(x, weight.reshape(count * out_features, width))
    every = every.reshape(rows, count, out_features)
    return every.gather(1, experts.long()[..., None].expand(rows, chosen, out_features))


def _combine_experts(args: Args) -> torch.Tensor:
    """`std.nn.moe::combine_experts`: the grouped product weighed and summed,
    or the weighed inputs placed by expert and multiplied by every expert."""
    x, weight, experts, weights = args
    rows, chosen, width = x.shape
    count, out_features, _ = weight.shape
    from . import moe

    if _grouped(x, weight) or moe.bf16_available(x, weight):
        y = _linear_experts([x, weight, experts]).float() * weights.float()[..., None]
        return y.sum(1).to(x.dtype)
    placed = torch.zeros(rows, count, width, dtype=x.dtype, device=x.device)
    placed.scatter_add_(
        1, experts.long()[..., None].expand(rows, chosen, width), weights[..., None] * x
    )
    joined = weight.permute(1, 0, 2).reshape(out_features, count * width)
    return functional.linear(placed.reshape(rows, count * width), joined)


NATIVE: dict[str, Native] = {
    "torch.ops.aten._weight_int4pack_mm": _int4_groups_linear,
    "linnet.decode_fp8": _decode_fp8,
    "linnet.fp8_linear": _fp8_linear(0),
    "torch._scaled_mm": _fp8_linear(1),
    "linnet.fp8_block_linear": _fp8_block_linear(0),
    "torch._scaled_mm(blocks)": _fp8_block_linear(1),
    "linnet.int8_linear": _int8_linear(0),
    "linnet.int8_linear(w8a8)": _int8_linear(1),
    "linnet.int8_group_linear": _int8_group_linear,
    "torch._grouped_mm": _required(_linear_experts),
    "torch._grouped_mm(shared)": _required(_linear_experts_shared),
    "torch._grouped_mm(combined)": _required(_combine_experts),
    "torch.nn.functional.embedding": _required(_embedding),
    "torch.index_select": _required(_index_select),
    "torch.nn.functional.batch_norm": _required(_batch_norm),
    "torch.Tensor.mean": _required(_spatial_mean),
    "torch.Tensor.index_copy": _required(_index_copy),
    "torch.Tensor.index_put": _required(_index_put),
    "torch.Tensor.index_put(tokens)": _required(_write_tokens),
    "torch.distributed.all_reduce": _required(_one_shard),
    "torch.distributed.all_gather": _required(_one_shard),
    "linnet.sink_attention": _required(_sink_attention),
    "linnet.gated_delta_rule": _required(_gated_delta_rule),
    "torch.nn.functional.conv1d(depthwise)": _required(_causal_conv),
    "linnet.gated_delta_update": _required(_gated_delta_update),
    "linnet.gated_delta_rule_packed": _required(_gated_delta_rule_packed),
    "linnet.causal_conv_packed": _required(_causal_conv_packed),
    "linnet.causal_history_packed": _required(_causal_history_packed),
    "linnet.mxfp4_experts": _required(_mxfp4_experts),
    "linnet.mxfp4_experts(shared)": _required(_mxfp4_experts_shared),
    "linnet.mxfp4_grouped(shared)": _required(_mxfp4_grouped_shared),
    "linnet.linear_cross_entropy": _required(_linear_cross_entropy),
    "linnet.linear_token_log_probs": _required(_linear_token_log_probs),
    "linnet.mxfp4_grouped(combine)": _required(_mxfp4_grouped_combine),
    "torch.matmul": _required(_matmul),
    "torch.nn.functional.linear": _linear,
    "torch.softmax": _required(_softmax),
    "torch.relu": _required(_relu),
    "torch.sigmoid": _required(_sigmoid),
    "torch.nn.functional.silu": _required(_silu),
    "torch.nn.functional.gelu(tanh)": _required(_gelu),
    "torch.nn.functional.gelu": _required(_gelu_erf),
    "torch.rms_norm": _required(_rms_norm),
    "torch.nn.functional.layer_norm": _layer_norm,
    "torch.nn.functional.scaled_dot_product_attention": _attention,
    "torch.softmax(input dtype)": _required(_softmax),
    "torch.rms_norm(input dtype)": _required(_rms_norm),
    "torch.nn.functional.layer_norm(input dtype)": _layer_norm_fast,
    "torch.nn.functional.scaled_dot_product_attention(input dtype)": _attention_fast,
    "torch.nn.functional.scaled_dot_product_attention(enable_gqa)": _attention,
    "torch.nn.functional.scaled_dot_product_attention(enable_gqa)(input dtype)": _attention_fast,
}
