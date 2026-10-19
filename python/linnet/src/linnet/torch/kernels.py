"""Triton kernels the generated PyTorch calls for operations PyTorch has no
kernel of its own for. `int4_linear` and `mxfp4_experts` are
`torch.library.triton_op`s, which run the same inside `torch.compile` and a
captured CUDA graph as outside them. The 8-bit kernels (`gemv8`, `w8a16`,
`fp8_rows`, `int8_rows`, `fp8_groups`, `int8_mm`) are custom ops inside
`torch.compile` and launched directly outside it, where an op's host cost
would be most of a decoding step's; `dequantize8` is launched directly
everywhere, Dynamo tracing the launch.

Importing this module needs Triton (and so a CUDA build of PyTorch); the
generated code falls back to plain PyTorch arithmetic without it.
"""

# Triton reads a kernel's annotations and resolves only its own, so the kernels
# stay unannotated.
# pyright: reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUntypedFunctionDecorator=false

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Concatenate, ParamSpec, Protocol, TypeVar

import torch
import triton  # pyright: ignore[reportMissingTypeStubs]
import triton.language as tl  # pyright: ignore[reportMissingTypeStubs]
from torch.library import triton_op, wrap_triton

_P = ParamSpec("_P")
_R = TypeVar("_R")


class Launcher(Protocol):
    """A Triton kernel, or one wrapped for `torch.compile` (`wrap_triton`):
    indexed by its grid (a tuple, or a function of the launch's
    meta-parameters), then called."""

    def __getitem__(
        self, grid: Callable[..., tuple[int, ...]] | tuple[int, ...], /
    ) -> Callable[..., object]: ...


def _launched(
    op: Callable[_P, _R], launch: Callable[Concatenate[Launcher, _P], _R], kernel: Launcher
) -> Callable[_P, _R]:
    """`launch` run with `kernel`: the custom op `op` inside `torch.compile`,
    which keeps it opaque, and outside it the kernel launched directly,
    whose host cost is a quarter of the op's."""

    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if torch.compiler.is_compiling():
            return op(*args, **kwargs)
        return launch(kernel, *args, **kwargs)

    return run


@triton.jit
def _grouped_tile(
    pid, rows, width, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr
):
    # Program `pid`'s output tile (its row and column tile), programs
    # ordered in groups of `group_m` row tiles so that neighbours share
    # weight tiles in L2.
    tiles_m = tl.cdiv(rows, block_m)
    tiles_n = tl.cdiv(width, block_n)
    per_group = group_m * tiles_n
    first = (pid // per_group) * group_m
    size = min(tiles_m - first, group_m)
    return first + (pid % per_group) % size, (pid % per_group) // size


@triton.jit
def _e2m1_pair(spread, signs):
    # Two E2M1 nibbles as fp16s, from a word shifted so the two nibbles' low
    # three bits sit at bits 9..11 and 25..27 (`spread`) and their sign bits at
    # 15 and 31 (`signs`): masked together, each half of the word is an fp16
    # whose exponent bits are the nibble's exponent and whose top mantissa bit
    # is its mantissa bit -- the E2M1 value times 2 ** -14, normal and
    # subnormal alike; the block's scale puts the 2 ** 14 back. One `lop3`.
    # (-2147450880 is 0x80008000 as a signed 32-bit integer.)
    word = (spread & 0x0E000E00) | (signs & -2147450880)
    low, high = tl.inline_asm_elementwise(
        "mov.b32 {$0, $1}, $2;",
        "=h,=h,r",
        [word],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    return low.to(tl.float32), high.to(tl.float32)


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
    # program; block_g blocks of 32 weights of each row at a time, each block's
    # 16 bytes read as four 32-bit words of eight nibbles.
    pair = tl.program_id(0)
    tile = tl.program_id(1)
    expert = tl.load(experts_ptr + pair)
    outs = tile * block_o + tl.arange(0, block_o)
    live_o = outs < width
    rows = (expert * width + outs).to(tl.int64)
    words_ptr = blocks_ptr.to(tl.pointer_type(tl.int32))
    quads = tl.arange(0, 4)
    acc = tl.zeros((block_o, block_g), dtype=tl.float32)
    for g0 in range(0, count, block_g):
        groups = g0 + tl.arange(0, block_g)
        live = groups < count
        w = tl.load(
            words_ptr
            + (rows[:, None, None] * count + groups[None, :, None]) * 4
            + quads[None, None, :],
            mask=live_o[:, None, None] & live[None, :, None],
            other=0,
        )
        # Word j of a block holds bytes 4j..4j+3, so inputs 8j..8j+7: byte i's
        # low nibble is input 8j + 2i, its high one 8j + 2i + 1. The inputs
        # 8j + t, t = 4a + 2b + c, split by c, then b, then a, line up with
        # the pairs `_e2m1_pair` makes: the low nibbles of bytes 0 and 2
        # (t = 0, 4), of bytes 1 and 3 (t = 2, 6), and the high nibbles of
        # bytes 0 and 2 (t = 1, 5) and of bytes 1 and 3 (t = 3, 7).
        xs = tl.load(
            x_ptr
            + pair * (count * 32)
            + groups[:, None, None] * 32
            + quads[None, :, None] * 8
            + tl.arange(0, 8)[None, None, :],
            mask=live[:, None, None],
            other=0.0,
        ).to(tl.float32)
        even, odd = tl.split(tl.reshape(xs, (block_g, 4, 2, 2, 2)))
        x_low02, x_low13 = tl.split(even)
        x_high02, x_high13 = tl.split(odd)
        x0, x4 = tl.split(x_low02)
        x2, x6 = tl.split(x_low13)
        x1, x5 = tl.split(x_high02)
        x3, x7 = tl.split(x_high13)
        w0, w4 = _e2m1_pair(w << 9, w << 12)
        w2, w6 = _e2m1_pair(w << 1, w << 4)
        w1, w5 = _e2m1_pair(w << 5, w << 8)
        w3, w7 = _e2m1_pair(w >> 3, w)
        part = (
            w0 * x0[None]
            + w1 * x1[None]
            + w2 * x2[None]
            + w3 * x3[None]
            + w4 * x4[None]
            + w5 * x5[None]
            + w6 * x6[None]
            + w7 * x7[None]
        )
        scale = tl.load(
            scales_ptr + rows[:, None] * count + groups[None, :],
            mask=live_o[:, None] & live[None, :],
            other=113,
        )
        acc += tl.sum(part, axis=2) * tl.exp2(
            scale.to(tl.float32) - 113.0
        )  # 2 ** (scale - 127) * 2 ** 14
    tl.store(
        y_ptr + pair * width + outs, tl.sum(acc, axis=1).to(y_ptr.dtype.element_ty), mask=live_o
    )


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
    # Measured on an H100 for gpt-oss's 5760- and 2880-wide projections, one
    # row and four: about 2.1 TB/s of weights read, against 1.4 unpacking
    # each nibble on its own.
    block_o = 16

    def grid(_meta: dict[str, object]) -> tuple[int, int]:
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
        num_stages=2,
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
    split = _split(programs, depth, block_k, x.device, waves, most=4)
    if split == 1:
        y = torch.empty(rows, width, dtype=x.dtype, device=x.device)
    else:
        y = torch.empty(split, rows, width, dtype=torch.float32, device=x.device)

    def grid(_meta: dict[str, object]) -> tuple[int, int, int]:
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
def _widen8(w, fp8: tl.constexpr):
    # One byte of weight as f32: FP8 E4M3 bits through the hardware's
    # conversion, int8 as 2 ** 23 + 128 + w read as an f32 less 2 ** 23 + 128
    # (integer and float adds instead of a conversion).
    if fp8:
        return w.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return (w.to(tl.int32) + 0x4B000080).to(tl.float32, bitcast=True) - 8388736.0


@triton.jit
def _gemv8_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    y_ptr,
    depth,
    scale_columns,
    fp8: tl.constexpr,
    group: tl.constexpr,
    row_block: tl.constexpr,
    chunk: tl.constexpr,
    unroll: tl.constexpr,
):
    # One output per program, one warp, on CUDA cores: `chunk` bytes of the
    # weight row at a time, `unroll` chunks a step, widened to f32 in
    # registers. `group` 0: one scale a row, applied to the sum; otherwise
    # the scale of rows `n // row_block` and inputs `k // group`, applied to
    # each chunk (`chunk` divides `group`).
    n = tl.program_id(0)
    inputs = tl.arange(0, chunk)
    row = weight_ptr + n.to(tl.int64) * depth + inputs
    acc = tl.zeros((chunk,), dtype=tl.float32)
    for k0 in range(0, depth, chunk * unroll):
        for j in tl.static_range(unroll):  # pyright: ignore[reportGeneralTypeIssues]
            k = k0 + j * chunk
            w = _widen8(tl.load(row + k, eviction_policy="evict_first"), fp8)
            x = tl.load(x_ptr + k + inputs).to(tl.float32)
            if group == 0:
                acc += w * x
            else:
                step = tl.load(scale_ptr + (n // row_block) * scale_columns + k // group)
                acc += w * (x * step.to(tl.float32))
    y = tl.sum(acc, axis=0)
    if group == 0:
        y = y * tl.load(scale_ptr + n).to(tl.float32)
    tl.store(y_ptr + n, y.to(y_ptr.dtype.element_ty))


def _gemv8(
    kernel: Launcher,
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    fp8: bool,
    group: int,
    row_block: int,
) -> torch.Tensor:
    """`F.linear(x, W)` for one row `x` [1, In] (`bf16` or `f16`), with `W`
    8-bit weights [Out, In] (FP8 E4M3 bits as `u8`, or `i8`) times `scale`:
    one a row ([Out]) when `group` is 0, else [Out / row_block, In / group]
    (int8 groups: `row_block` 1; FP8 blocks: both 128). The weight is read
    once and `x` stays unrounded: a decoding step of one sequence, at the
    memory's speed (measured on an H100: 2.2 to 3.0 TB/s over Llama 3.1 8B's
    projections). In and the group are multiples of 16."""
    depth = x.shape[-1]
    width = weight.shape[0]
    # Measured: one row a program in one warp, 256 inputs a chunk (fewer
    # where the group or In needs it), four chunks a step where they divide
    # In.
    chunk = next(c for c in (256, 128, 64, 32, 16) if depth % c == 0 and (group or depth) % c == 0)
    unroll = next(u for u in (4, 2, 1) if depth % (chunk * u) == 0)
    y = torch.empty(1, width, dtype=x.dtype, device=x.device)

    def grid(_meta: dict[str, object]) -> tuple[int]:
        return (width,)

    kernel[grid](
        x.contiguous(),
        weight,
        scale,
        y,
        depth,
        scale.shape[-1] if group else 1,
        fp8=fp8,
        group=group,
        row_block=row_block,
        chunk=chunk,
        unroll=unroll,
        num_warps=1,
        num_stages=1,
    )
    return y


@triton_op("linnet::gemv8", mutates_args=())
def _gemv8_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    fp8: bool,
    group: int,
    row_block: int,
) -> torch.Tensor:
    return _gemv8(wrap_triton(_gemv8_kernel), x, weight, scale, fp8, group, row_block)


# See `_gemv8`.
gemv8 = _launched(_gemv8_op, _gemv8, _gemv8_kernel)


@triton.jit
def _w8a16_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    y_ptr,
    rows,
    width,
    depth,
    span,
    scale_columns,
    fp8: tl.constexpr,
    group: tl.constexpr,
    row_block: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
):
    # A block_m by block_n tile of the output per program, over one of the
    # split's equal stretches of the inputs (axis 1); programs ordered in
    # groups of `group_m` row tiles so neighbours share weight tiles in L2.
    # Each step widens block_k of every weight row to `x`'s dtype (exact:
    # every FP8 and int8 value is a bf16 and an f16), scaled first when the
    # scales are per group (block_k divides the group), and multiplies on
    # tensor cores.
    part = tl.program_id(1)
    tile_m, tile_n = _grouped_tile(tl.program_id(0), rows, width, block_m, block_n, group_m)
    rm = tile_m * block_m + tl.arange(0, block_m)
    rn = tile_n * block_n + tl.arange(0, block_n)
    rk = tl.arange(0, block_k)
    live_m = rm < rows
    live_n = rn < width
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    # This part's stretch of the inputs, `span` (a multiple of block_k)
    # from its start: the loop runs from zero, so Triton reads in wide loads.
    first_k = part * span
    xs = x_ptr + first_k + rm[:, None] * depth + rk[None, :]
    ws = weight_ptr + first_k + rn[:, None].to(tl.int64) * depth + rk[None, :]
    for k0 in range(0, span, block_k):
        x = tl.load(xs + k0, mask=live_m[:, None], other=0.0)
        w = _widen8(tl.load(ws + k0, mask=live_n[:, None], other=0), fp8)
        if group != 0:
            step = tl.load(
                scale_ptr + (rn // row_block) * scale_columns + (first_k + k0) // group,
                mask=live_n,
                other=0.0,
            )
            w = w * step.to(tl.float32)[:, None]
        acc = tl.dot(x, tl.trans(w.to(x_ptr.dtype.element_ty)), acc)
    if group == 0:
        acc = acc * tl.load(scale_ptr + rn, mask=live_n, other=0.0).to(tl.float32)[None, :]
    out = y_ptr + part * rows * width + rm[:, None] * width + rn[None, :]
    tl.store(out, acc.to(y_ptr.dtype.element_ty), mask=live_m[:, None] & live_n[None, :])


# Tiles by row count, measured on an H100 over Llama 3.1 8B's joined
# projections: (rows up to, block_m, block_n, block_k, warps, stages). Past
# 64 rows cuBLAS over the weight dequantized for the call is faster.
_W8A16_TILES = (
    (16, 16, 64, 128, 8, 4),
    (32, 32, 64, 128, 4, 4),
    (1 << 62, 64, 64, 128, 4, 4),
)


@functools.cache
def _processors(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _split(
    programs: int, depth: int, block_k: int, device: torch.device, waves: int = 4, most: int = 8
) -> int:
    """Parts the inputs split into so that a few rows' tiles fill the GPU's
    multiprocessors: doubled, up to `most`, while twice the programs still
    fit in `waves` waves of them and the parts' stretches stay whole
    `block_k`s (each part sums a stretch of the inputs, and the parts are
    added after)."""
    split = 1
    while (
        split < most
        and programs * split * 2 <= waves * _processors(device)
        and depth % (2 * split * block_k) == 0
    ):
        split *= 2
    return split


def _w8a16(
    kernel: Launcher,
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    fp8: bool,
    group: int,
    row_block: int,
) -> torch.Tensor:
    """`F.linear(x, W)` for `x` [rows, In] (`bf16` or `f16`) and the 8-bit
    weights of `_gemv8`, widened to `x`'s dtype inside the product: the
    weight is read once, at half of `bf16`'s bytes, and `x` is not rounded.
    In is a multiple of 128 and of the group. Meant for up to 64 rows."""
    rows, depth = x.shape
    width = weight.shape[0]
    _, block_m, block_n, block_k, warps, stages = next(
        tiles for tiles in _W8A16_TILES if rows <= tiles[0]
    )
    if group:
        block_k = min(block_k, group)
    programs = triton.cdiv(rows, block_m) * triton.cdiv(width, block_n)
    split = _split(programs, depth, block_k, x.device)
    if split == 1:
        y = torch.empty(rows, width, dtype=x.dtype, device=x.device)
    else:
        y = torch.empty(split, rows, width, dtype=torch.float32, device=x.device)

    def grid(_meta: dict[str, object]) -> tuple[int, int]:
        return (programs, split)

    kernel[grid](
        x.contiguous(),
        weight,
        scale,
        y,
        rows,
        width,
        depth,
        depth // split,
        scale.shape[-1] if group else 1,
        fp8=fp8,
        group=group,
        row_block=row_block,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=8,
        num_warps=warps,
        num_stages=stages,
    )
    return y if split == 1 else y.sum(0).to(x.dtype)


@triton_op("linnet::w8a16", mutates_args=())
def _w8a16_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    fp8: bool,
    group: int,
    row_block: int,
) -> torch.Tensor:
    return _w8a16(wrap_triton(_w8a16_kernel), x, weight, scale, fp8, group, row_block)


# See `_w8a16`.
w8a16 = _launched(_w8a16_op, _w8a16, _w8a16_kernel)


@triton.jit
def _dequantize8_kernel(
    weight_ptr,
    scale_ptr,
    out_ptr,
    depth,
    scale_columns,
    fp8: tl.constexpr,
    group: tl.constexpr,
    row_block: tl.constexpr,
    block: tl.constexpr,
):
    # `block` weights of one row per program, widened and scaled in f32, then
    # rounded to the output's dtype, as `std.quant`'s bodies dequantize.
    row = tl.program_id(0)
    cols = tl.program_id(1) * block + tl.arange(0, block)
    live = cols < depth
    w = _widen8(tl.load(weight_ptr + row.to(tl.int64) * depth + cols, mask=live, other=0), fp8)
    if group == 0:
        step = tl.load(scale_ptr + row).to(tl.float32)
    else:
        step = tl.load(
            scale_ptr + (row // row_block) * scale_columns + cols // group, mask=live, other=0.0
        ).to(tl.float32)
    out = out_ptr + row.to(tl.int64) * depth + cols
    tl.store(out, (w * step).to(out_ptr.dtype.element_ty), mask=live)


def dequantize8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    fp8: bool,
    group: int,
    row_block: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The weight of `_gemv8` dequantized to `dtype` in one kernel: for a
    call of many rows, which cuBLAS multiplies faster than Linnet's kernel
    widens the weight. Called outside `torch.compile` and inside, where
    Dynamo traces the launch."""
    kernel = wrap_triton(_dequantize8_kernel) if torch.compiler.is_compiling() else None
    return _dequantize8(kernel or _dequantize8_kernel, weight, scale, fp8, group, row_block, dtype)


def _dequantize8(
    kernel: Launcher,
    weight: torch.Tensor,
    scale: torch.Tensor,
    fp8: bool,
    group: int,
    row_block: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    width, depth = weight.shape
    out = torch.empty(width, depth, dtype=dtype, device=weight.device)

    def grid(_meta: dict[str, object]) -> tuple[int, int]:
        return (width, triton.cdiv(depth, 1024))

    kernel[grid](
        weight,
        scale,
        out,
        depth,
        scale.shape[-1] if group else 1,
        fp8=fp8,
        group=group,
        row_block=row_block,
        block=1024,
        num_warps=4,
    )
    return out


@triton.jit
def _rows8_kernel(
    x_ptr, q_ptr, scale_ptr, depth, fp8: tl.constexpr, block: tl.constexpr, whole: tl.constexpr
):
    # One row per program: its largest magnitude, then the row over its
    # scale (that magnitude over the type's largest value: 448 for FP8, 127
    # for int8) rounded to the type, int8 to the nearest level. A row that
    # fits in `block` (`whole`) is read once.
    row = tl.program_id(0).to(tl.int64)
    base = row * depth
    largest = 448.0 if fp8 else 127.0
    if whole:
        offsets = tl.arange(0, block)
        live = offsets < depth
        x = tl.load(x_ptr + base + offsets, mask=live, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12) / largest
        x = x / scale
        if fp8:
            q = tl.clamp(x, -448.0, 448.0).to(q_ptr.dtype.element_ty)
        else:
            q = tl.clamp(tl.where(x >= 0, x + 0.5, x - 0.5), -127.0, 127.0).to(tl.int8)
        tl.store(q_ptr + base + offsets, q, mask=live)
    else:
        top = tl.zeros((block,), dtype=tl.float32)
        for k0 in range(0, depth, block):
            offsets = k0 + tl.arange(0, block)
            x = tl.load(x_ptr + base + offsets, mask=offsets < depth, other=0.0).to(tl.float32)
            top = tl.maximum(top, tl.abs(x))
        scale = tl.maximum(tl.max(top, axis=0), 1e-12) / largest
        for k0 in range(0, depth, block):
            offsets = k0 + tl.arange(0, block)
            live = offsets < depth
            x = tl.load(x_ptr + base + offsets, mask=live, other=0.0).to(tl.float32) / scale
            if fp8:
                q = tl.clamp(x, -448.0, 448.0).to(q_ptr.dtype.element_ty)
            else:
                q = tl.clamp(tl.where(x >= 0, x + 0.5, x - 0.5), -127.0, 127.0).to(tl.int8)
            tl.store(q_ptr + base + offsets, q, mask=live)
    tl.store(scale_ptr + row, scale)


def _rows8(kernel: Launcher, x: torch.Tensor, fp8: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row of `x` [rows, In] rounded to FP8 E4M3 or int8 with a scale
    of its own: the rounded rows and the scales ([rows, 1], f32), in one
    kernel, as `torch._scaled_mm` and `int8_mm` take them."""
    rows, depth = x.shape
    dtype = torch.float8_e4m3fn if fp8 else torch.int8
    q = torch.empty(rows, depth, dtype=dtype, device=x.device)
    scale = torch.empty(rows, 1, dtype=torch.float32, device=x.device)
    # A row of up to 16384 inputs in registers, read once.
    whole = depth <= 16384
    block = triton.next_power_of_2(depth) if whole else 4096

    def grid(_meta: dict[str, object]) -> tuple[int]:
        return (rows,)

    kernel[grid](
        x.contiguous(),
        q,
        scale,
        depth,
        fp8=fp8,
        block=block,
        whole=whole,
        num_warps=4 if block <= 4096 else 8,
    )
    return q, scale


@triton_op("linnet::rows8", mutates_args=())
def _rows8_op(x: torch.Tensor, fp8: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return _rows8(wrap_triton(_rows8_kernel), x, fp8)


_rows8_launched = _launched(_rows8_op, _rows8, _rows8_kernel)


def fp8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """See `_rows8`."""
    return _rows8_launched(x, True)


def int8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """See `_rows8`."""
    return _rows8_launched(x, False)


@triton.jit
def _fp8_groups_kernel(
    x_ptr, q_ptr, scale_ptr, rows, padded, depth, group: tl.constexpr, groups: tl.constexpr
):
    # `groups` groups of one row per program: each group's largest magnitude
    # over 448, the group over it rounded to FP8. Rows past `rows`, up to
    # `padded`, are zeros with a scale of one. The scales are written column
    # by column ([In / group, padded]), the layout `torch._scaled_mm` reads.
    row = tl.program_id(0)
    first = tl.program_id(1) * groups
    g = first + tl.arange(0, groups)
    offsets = g[:, None] * group + tl.arange(0, group)[None, :]
    base = row.to(tl.int64) * depth
    live = (g < depth // group)[:, None]
    x = tl.load(x_ptr + base + offsets, mask=live & (row < rows), other=0.0).to(tl.float32)
    top = tl.max(tl.abs(x), axis=1)
    scale = tl.where(row < rows, tl.maximum(top, 1e-12) / 448.0, 1.0)
    q = tl.clamp(x / scale[:, None], -448.0, 448.0).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + base + offsets, q, mask=live)
    tl.store(scale_ptr + g * padded + row, scale, mask=g < depth // group)


def _fp8_groups(
    kernel: Launcher, x: torch.Tensor, group: int, padded: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each `group` inputs of each row of `x` [rows, In] rounded to FP8 E4M3
    with a scale of their own, the rows padded with zeros to `padded`: the
    rounded rows [padded, In] and the scales [padded, In / group] (f32,
    column-major), as `torch._scaled_mm` takes them with block weights."""
    rows, depth = x.shape
    q = torch.empty(padded, depth, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(depth // group, padded, dtype=torch.float32, device=x.device)

    def grid(_meta: dict[str, object]) -> tuple[int, int]:
        return (padded, triton.cdiv(depth // group, 8))

    kernel[grid](x.contiguous(), q, scale, rows, padded, depth, group=group, groups=8, num_warps=4)
    return q, scale.t()


@triton_op("linnet::fp8_groups", mutates_args=())
def _fp8_groups_op(x: torch.Tensor, group: int, padded: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _fp8_groups(wrap_triton(_fp8_groups_kernel), x, group, padded)


# See `_fp8_groups`.
fp8_groups = _launched(_fp8_groups_op, _fp8_groups, _fp8_groups_kernel)


@triton.jit
def _int8_mm_kernel(
    x_ptr,
    weight_ptr,
    x_scale_ptr,
    scale_ptr,
    y_ptr,
    rows,
    width,
    depth,
    span,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
):
    # int8 rows times int8 weight rows on tensor cores, summed in int32
    # (exactly) over one of the split's stretches of the inputs (axis 1),
    # then both scales and the output's dtype in one epilogue.
    part = tl.program_id(1)
    tile_m, tile_n = _grouped_tile(tl.program_id(0), rows, width, block_m, block_n, group_m)
    rm = tile_m * block_m + tl.arange(0, block_m)
    rn = tile_n * block_n + tl.arange(0, block_n)
    rk = tl.arange(0, block_k)
    live_m = rm < rows
    live_n = rn < width
    acc = tl.zeros((block_m, block_n), dtype=tl.int32)
    first_k = part * span
    xs = x_ptr + first_k + rm[:, None] * depth + rk[None, :]
    ws = weight_ptr + first_k + rn[None, :].to(tl.int64) * depth + rk[:, None]
    for k0 in range(0, span, block_k):
        x = tl.load(xs + k0, mask=live_m[:, None], other=0)
        w = tl.load(ws + k0, mask=live_n[None, :], other=0)
        acc = tl.dot(x, w, acc, out_dtype=tl.int32)
    x_scale = tl.load(x_scale_ptr + rm, mask=live_m, other=0.0)
    scale = tl.load(scale_ptr + rn, mask=live_n, other=0.0).to(tl.float32)
    y = acc.to(tl.float32) * x_scale[:, None] * scale[None, :]
    out = y_ptr + part * rows * width + rm[:, None] * width + rn[None, :]
    tl.store(out, y.to(y_ptr.dtype.element_ty), mask=live_m[:, None] & live_n[None, :])


# Tiles by row count, measured as `_W8A16_TILES`: (rows up to, block_m,
# block_n, block_k, warps, stages).
_INT8_TILES = (
    (64, 16, 64, 128, 4, 4),
    (256, 64, 64, 128, 4, 4),
    (512, 64, 128, 128, 4, 3),
    (1 << 62, 128, 128, 128, 4, 3),
)


def _int8_mm(
    kernel: Launcher,
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """`(x @ weight.T) * x_scale * scale` in `dtype`, for int8 rows `x`
    [rows, In] with their scales ([rows, 1], f32, from `int8_rows`) and int8
    weights [Out, In] with one scale a row ([Out]). In is a multiple of
    128."""
    rows, depth = x.shape
    width = weight.shape[0]
    _, block_m, block_n, block_k, warps, stages = next(
        tiles for tiles in _INT8_TILES if rows <= tiles[0]
    )
    programs = triton.cdiv(rows, block_m) * triton.cdiv(width, block_n)
    split = _split(programs, depth, block_k, x.device)
    if split == 1:
        y = torch.empty(rows, width, dtype=dtype, device=x.device)
    else:
        y = torch.empty(split, rows, width, dtype=torch.float32, device=x.device)

    def grid(_meta: dict[str, object]) -> tuple[int, int]:
        return (programs, split)

    kernel[grid](
        x,
        weight,
        x_scale,
        scale,
        y,
        rows,
        width,
        depth,
        depth // split,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=8,
        num_warps=warps,
        num_stages=stages,
    )
    return y if split == 1 else y.sum(0).to(dtype)


@triton_op("linnet::int8_mm", mutates_args=())
def _int8_mm_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _int8_mm(wrap_triton(_int8_mm_kernel), x, weight, x_scale, scale, dtype)


# See `_int8_mm`.
int8_mm = _launched(_int8_mm_op, _int8_mm, _int8_mm_kernel)


@triton.jit
def _delta_update_kernel(
    state_in,
    state_out,
    key,
    value,
    decay,
    beta,
    key_heads,
    value_heads,
    width: tl.constexpr,
    columns: tl.constexpr,
    block: tl.constexpr,
):
    # One program per (row, value head, block of value columns): its
    # [width, block] slice of the state read once, updated, written back.
    row_head = tl.program_id(0)
    row = row_head // value_heads
    group = (row_head % value_heads) // (value_heads // key_heads)
    ks = tl.arange(0, width)
    vs = tl.program_id(1) * block + tl.arange(0, block)
    at = row_head.to(tl.int64) * width * columns + ks[:, None] * columns + vs[None, :]
    s = tl.load(state_in + at)
    k = tl.load(key + (row * key_heads + group) * width + ks).to(tl.float32)
    v = tl.load(value + row_head * columns + vs).to(tl.float32)
    s = s * tl.exp(tl.load(decay + row_head))
    predicted = tl.sum(s * k[:, None], axis=0)
    correction = (v - predicted) * tl.load(beta + row_head)
    tl.store(state_out + at, s + k[:, None] * correction[None, :])


def _delta_update(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> None:
    rows, heads, width, columns = state_in.shape
    block = 32 if columns % 32 == 0 else columns
    wrap_triton(_delta_update_kernel)[(rows * heads, columns // block)](
        state_in,
        state_out,
        key,
        value,
        decay,
        beta,
        key.shape[1],
        heads,
        width=width,
        columns=columns,
        block=block,
    )


@triton_op("linnet::delta_update", mutates_args=())
def delta_update(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """One token of the gated delta rule per row (`std.nn.recurrent::
    gated_delta_update`): `state` [B, Hv, K, V] f32 contiguous, `key` [B,
    Hk, K], `value` [B, Hv, V], `decay`/`beta` [B, Hv] f32; the new state."""
    out = torch.empty_like(state)
    _delta_update(state, out, key, value, decay, beta)
    return out


@triton_op("linnet::delta_update_", mutates_args={"state"})
def delta_update_(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> None:
    """`delta_update` written into `state` itself."""
    _delta_update(state, state, key, value, decay, beta)


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
    # (block, rank), then wait until every peer has told us. (`static_range`
    # unrolls as Triton compiles; its Python type is not iterable.)
    for peer in tl.static_range(world):  # pyright: ignore[reportGeneralTypeIssues]
        if peer != rank:
            theirs = tl.load(flags_ptr + peer).to(tl.pointer_type(tl.int32))
            tl.atomic_xchg(theirs + block * world + rank, epoch, sem="release", scope="sys")
    own = tl.load(flags_ptr + rank).to(tl.pointer_type(tl.int32))
    for peer in tl.static_range(world):  # pyright: ignore[reportGeneralTypeIssues]
        if peer != rank:
            seen = tl.atomic_add(own + block * world + peer, 0, sem="acquire", scope="sys")
            while seen < epoch:
                seen = tl.atomic_add(own + block * world + peer, 0, sem="acquire", scope="sys")
    tl.debug_barrier()
    total = x.to(tl.float32)
    for peer in tl.static_range(world):  # pyright: ignore[reportGeneralTypeIssues]
        if peer != rank:
            part = tl.load(buffers_ptr + peer).to(tl.pointer_type(dtype))
            read = tl.load(part + slot + offsets, mask=live, other=0.0, volatile=True)
            total += read.to(tl.float32)
    tl.store(out_ptr + offsets, total.to(dtype), mask=live)


__all__ = [
    "Launcher",
    "dequantize8",
    "fp8_groups",
    "fp8_rows",
    "gemv8",
    "int4_linear",
    "int8_mm",
    "int8_rows",
    "mxfp4_experts",
    "one_shot_all_reduce_kernel",
    "w8a16",
]
