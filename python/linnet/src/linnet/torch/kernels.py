"""Triton kernels the generated PyTorch calls for operations PyTorch has no
kernel of its own for. Each is a `torch.library.triton_op`, so it runs the
same inside `torch.compile` and a captured CUDA graph as outside them.

Importing this module needs Triton (and so a CUDA build of PyTorch); the
generated code falls back to plain PyTorch arithmetic without it.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportMissingTypeStubs=false

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


__all__ = ["mxfp4_experts"]
