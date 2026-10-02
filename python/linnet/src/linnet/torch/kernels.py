"""Triton kernels the generated PyTorch calls for operations PyTorch has no
kernel of its own for. Each is a `torch.library.triton_op`, so it runs the
same inside `torch.compile` and a captured CUDA graph as outside them.

Importing this module needs Triton (and so a CUDA build of PyTorch); the
generated code falls back to plain PyTorch arithmetic without it.
"""

# Triton reads a kernel's annotations and resolves only its own, so the kernels
# stay unannotated.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportMissingTypeStubs=false, reportMissingParameterType=false, reportUnknownParameterType=false

from __future__ import annotations

from typing import Any

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch.library import triton_op, wrap_triton


@triton.jit
def _half(nibble):
    # An E2M1 nibble's bits placed in an fp16: its exponent bits at 11..10,
    # its mantissa bit at 9, its sign at 15. That fp16 is the E2M1 value times
    # 2 ** -14, normal and subnormal alike; the block's scale puts the 2 ** 14
    # back.
    bits = ((nibble & 7) << 9) | ((nibble & 8) << 12)
    return bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _mxfp4_experts_kernel(
    x_ptr,
    blocks_ptr,
    scales_ptr,
    experts_ptr,
    y_ptr,
    width: tl.constexpr,
    count: tl.constexpr,
    block_o: tl.constexpr,
    block_g: tl.constexpr,
):
    # One (row, chosen slot) pair and block_o of its expert's output rows per
    # program; block_g blocks of 32 weights of each row at a time.
    pair = tl.program_id(0)
    tile = tl.program_id(1)
    expert = tl.load(experts_ptr + pair)
    outs = tile * block_o + tl.arange(0, block_o)
    rows = (expert * width + outs).to(tl.int64)
    lanes = tl.arange(0, 16)
    acc = tl.zeros((block_o, block_g), dtype=tl.float32)
    for g0 in range(0, count, block_g):
        groups = g0 + tl.arange(0, block_g)
        live = groups < count
        packed = tl.load(
            blocks_ptr
            + (rows[:, None, None] * count + groups[None, :, None]) * 16
            + lanes[None, None, :],
            mask=live[None, :, None],
            other=0,
        ).to(tl.int32)
        # The inputs a block's bytes multiply: element 2j (the low nibble of
        # byte j) and 2j + 1 (its high nibble).
        pairs = tl.load(
            x_ptr
            + pair * (count * 32)
            + (
                groups[:, None, None] * 32
                + lanes[None, :, None] * 2
                + tl.arange(0, 2)[None, None, :]
            ),
            mask=live[:, None, None],
            other=0.0,
        ).to(tl.float32)
        x_low, x_high = tl.split(pairs)
        part = tl.sum(
            _half(packed & 15) * x_low[None, :, :] + _half(packed >> 4) * x_high[None, :, :], axis=2
        )
        scale = tl.load(
            scales_ptr + rows[:, None] * count + groups[None, :], mask=live[None, :], other=113
        )
        acc += part * tl.exp2(scale.to(tl.float32) - 113.0)  # 2 ** (scale - 127) * 2 ** 14
    tl.store(y_ptr + pair * width + outs, tl.sum(acc, axis=1).to(y_ptr.dtype.element_ty))


@triton_op("linnet::mxfp4_experts", mutates_args=())
def mxfp4_experts(
    x: torch.Tensor, blocks: torch.Tensor, scales: torch.Tensor, experts: torch.Tensor
) -> torch.Tensor:
    """`y[r, k, o] = sum_i x[r, k, i] * W[experts[r, k], o, i]`, with `W` the
    MXFP4 weights `blocks` ([E, Out, G, 16] u8) and `scales` ([E, Out, G] u8)
    unpacked in registers: each pair reads only its expert's bytes. Meant for
    a decoding step's few rows; with many, each pair rereads its expert."""
    rows, chosen, _width = x.shape
    _count, out_features, groups, _ = blocks.shape
    y = torch.empty(rows, chosen, out_features, dtype=x.dtype, device=x.device)
    # Measured on an H100 for gpt-oss's 5760- and 2880-wide projections.
    block_o = 16 if out_features >= 4096 else 8

    def grid(_meta: dict[str, Any]) -> tuple[int, int]:
        return (rows * chosen, triton.cdiv(out_features, block_o))

    wrap_triton(_mxfp4_experts_kernel)[grid](
        x.contiguous(),
        blocks.contiguous(),
        scales.contiguous(),
        experts.reshape(-1).contiguous(),
        y,
        width=out_features,
        count=groups,
        block_o=block_o,
        block_g=32,
        num_warps=4,
    )
    return y


@triton.jit
def _int4_linear_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    zero_ptr,
    y_ptr,
    rows,
    width,
    depth,
    group: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    # A block_m by block_n tile of the output per program, over one of the
    # split's equal stretches of the inputs (axis 2), block_k at a time. Each
    # step unpacks block_k of every weight row (block_k divides the group, so
    # one scale and zero point serve it), puts the even and odd values back in
    # order, and multiplies on tensor cores.
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)
    part = tl.program_id(2)
    split = tl.num_programs(2)
    rm = tile_m * block_m + tl.arange(0, block_m)
    rn = tile_n * block_n + tl.arange(0, block_n)
    half = tl.arange(0, block_k // 2)
    inputs = tl.arange(0, block_k)
    groups = depth // group
    span = depth // split
    live_m = rm < rows
    live_n = rn < width
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k0 in range(part * span, (part + 1) * span, block_k):
        packed = tl.load(
            packed_ptr + rn[:, None].to(tl.int64) * (depth // 2) + (k0 // 2 + half)[None, :],
            mask=live_n[:, None],
            other=0,
        ).to(tl.int32)
        step = tl.load(scale_ptr + rn * groups + k0 // group, mask=live_n, other=0).to(tl.float32)
        level = tl.load(zero_ptr + rn * groups + k0 // group, mask=live_n, other=0).to(tl.float32)
        low = ((packed & 15).to(tl.float32) - level[:, None]) * step[:, None]
        high = ((packed >> 4).to(tl.float32) - level[:, None]) * step[:, None]
        weight = tl.interleave(low, high).to(x_ptr.dtype.element_ty)
        x = tl.load(
            x_ptr + rm[:, None] * depth + (k0 + inputs)[None, :], mask=live_m[:, None], other=0.0
        )
        acc = tl.dot(x, tl.trans(weight), acc)
    out = y_ptr + part * rows * width + rm[:, None] * width + rn[None, :]
    tl.store(out, acc.to(y_ptr.dtype.element_ty), mask=live_m[:, None] & live_n[None, :])


# Tiles by row count, measured on an H100 over the joined projections of
# Llama 3.1 8B, Qwen2.5 7B, and TinyLlama: (rows up to, block_m, block_n,
# warps, stages, waves). The inputs are split while the programs stay within
# `waves` waves of the GPU's multiprocessors, so a few rows over a short
# weight still fill it.
_INT4_TILES = (
    (16, 16, 32, 4, 3, 2),
    (32, 32, 64, 4, 3, 1),
    (64, 64, 64, 4, 3, 2),
    (128, 128, 64, 8, 4, 1),
)


@triton_op("linnet::int4_linear", mutates_args=())
def int4_linear(
    x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
) -> torch.Tensor:
    """`F.linear(x, W)` for `x` [rows, In] (`bf16` or `f16`), with `W` the
    group-wise 4-bit weights of `std.quant::linear_int4_groups` -- `packed`
    ([Out, Groups, Group / 2] u8, the even value low), `scale` and `zero`
    ([Out, Groups]) -- unpacked inside the product, so the weight is read
    once, at a quarter of `bf16`'s bytes. Meant for a batch of decoding
    requests, 8 to 128 rows: fewer go faster through tinygemm, more through
    cuBLAS over the weight dequantized once for the call. The group is 16,
    32, 64, or a multiple of 128."""
    rows, depth = x.shape
    width, _groups, half = packed.shape
    group = 2 * half
    block_k = min(group, 128)
    if block_k < 16 or group % block_k or block_k & (block_k - 1):
        raise ValueError(f"int4_linear: a group of {group} is not 16, 32, 64, or a multiple of 128")
    tiles = next((tiles for tiles in _INT4_TILES if rows <= tiles[0]), _INT4_TILES[-1])
    _, block_m, block_n, warps, stages, waves = tiles
    programs = triton.cdiv(rows, block_m) * triton.cdiv(width, block_n)
    budget = waves * torch.cuda.get_device_properties(x.device).multi_processor_count
    split = 1
    while split < 4 and programs * split * 2 <= budget and depth % (2 * split * block_k) == 0:
        split *= 2
    if split == 1:
        y = torch.empty(rows, width, dtype=x.dtype, device=x.device)
    else:
        y = torch.empty(split, rows, width, dtype=torch.float32, device=x.device)

    def grid(_meta: dict[str, Any]) -> tuple[int, int, int]:
        return (triton.cdiv(rows, block_m), triton.cdiv(width, block_n), split)

    wrap_triton(_int4_linear_kernel)[grid](
        x.contiguous(),
        packed.contiguous(),
        scale.contiguous(),
        zero.contiguous(),
        y,
        rows,
        width,
        depth,
        group=group,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    return y if split == 1 else y.sum(0).to(x.dtype)


@triton.jit
def one_shot_all_reduce_kernel(
    x_ptr,
    out_ptr,
    buffers_ptr,
    flags_ptr,
    counters_ptr,
    n,
    capacity,
    rank: tl.constexpr,
    world: tl.constexpr,
    block_size: tl.constexpr,
):
    block = tl.program_id(0)
    # This block's call count, kept by it alone: the epoch the same
    # block of every process agrees on, as they all make the same calls.
    epoch = tl.load(counters_ptr + block) + 1
    tl.store(counters_ptr + block, epoch)
    # Two slots, alternating: a peer may still be reading this call's
    # predecessor's part, never the one before that, which it finished
    # before it flagged the predecessor.
    slot = (epoch % 2).to(tl.int64) * capacity
    dtype = x_ptr.dtype.element_ty
    offsets = block * block_size + tl.arange(0, block_size)
    live = offsets < n
    x = tl.load(x_ptr + offsets, mask=live, other=0.0)
    mine = tl.load(buffers_ptr + rank).to(tl.pointer_type(dtype))
    tl.store(mine + slot + offsets, x, mask=live)
    tl.debug_barrier()
    # This block's part is in place: tell every peer, in its flags at
    # (block, rank), then wait until every peer has told us.
    for peer in tl.static_range(world):
        if peer != rank:
            theirs = tl.load(flags_ptr + peer).to(tl.pointer_type(tl.int32))
            tl.atomic_xchg(theirs + block * world + rank, epoch, sem="release", scope="sys")
    own = tl.load(flags_ptr + rank).to(tl.pointer_type(tl.int32))
    for peer in tl.static_range(world):
        if peer != rank:
            seen = tl.atomic_add(own + block * world + peer, 0, sem="acquire", scope="sys")
            while seen < epoch:
                seen = tl.atomic_add(own + block * world + peer, 0, sem="acquire", scope="sys")
    tl.debug_barrier()
    total = x.to(tl.float32)
    for peer in tl.static_range(world):
        if peer != rank:
            part = tl.load(buffers_ptr + peer).to(tl.pointer_type(dtype))
            read = tl.load(part + slot + offsets, mask=live, other=0.0, volatile=True)
            total += read.to(tl.float32)
    tl.store(out_ptr + offsets, total.to(dtype), mask=live)


__all__ = ["int4_linear", "mxfp4_experts", "one_shot_all_reduce_kernel"]
