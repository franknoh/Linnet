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

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch.library import triton_op, wrap_triton


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
    budget = waves * torch.cuda.get_device_properties(x.device).multi_processor_count
    split = 1
    while split < 4 and programs * split * 2 <= budget and depth % (2 * split * block_k) == 0:
        split *= 2
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
def _fp8_gemv_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    y_ptr,
    width,
    depth,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    # One input row times block_n weight rows per program, over one of the
    # split's equal stretches of the inputs (axis 1), on CUDA cores: each
    # step widens block_k FP8 bytes of every row to f32 in registers. Tensor
    # cores, which want 16 rows, ran these slower.
    tile_n = tl.program_id(0)
    part = tl.program_id(1)
    span = depth // tl.num_programs(1)
    rn = tile_n * block_n + tl.arange(0, block_n)
    inputs = tl.arange(0, block_k)
    live = rn < width
    acc = tl.zeros((block_n, block_k), dtype=tl.float32)
    for k0 in range(part * span, (part + 1) * span, block_k):
        bits = tl.load(
            weight_ptr + rn[:, None].to(tl.int64) * depth + (k0 + inputs)[None, :],
            mask=live[:, None],
            other=0,
        )
        x = tl.load(x_ptr + k0 + inputs).to(tl.float32)
        acc += bits.to(tl.float8e4nv, bitcast=True).to(tl.float32) * x[None, :]
    y = tl.sum(acc, axis=1) * tl.load(scale_ptr + rn, mask=live, other=0.0)
    tl.store(y_ptr + part * width + rn, y, mask=live)


@triton_op("linnet::fp8_gemv", mutates_args=())
def fp8_gemv(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """`F.linear(x, W)` for one row `x` [1, In] (`bf16` or `f16`), with `W`
    the per-row FP8 E4M3 weights of `std.quant::linear_fp8` -- `weight`
    ([Out, In] bytes) and `scale` ([Out]) -- read once, at half of `bf16`'s
    bytes, and multiplied without rounding `x`: a decoding step of one
    sequence. In is a multiple of 512."""
    depth = x.shape[-1]
    width = weight.shape[0]
    # Tiles measured on an H100 over Llama 3.1 8B's projections: small
    # weights take wider steps. Each program covers at most 2048 inputs, and
    # the inputs split further while there are fewer than 1024 programs.
    small = width <= 4096 and depth <= 4096
    block_n, block_k = (8, 512) if small else (4, 256)
    programs = triton.cdiv(width, block_n)
    steps = depth // block_k
    if depth % block_k:
        raise ValueError(f"fp8_gemv: {depth} inputs are not a multiple of {block_k}")
    parts = [d for d in range(1, steps + 1) if steps % d == 0]
    split = next(d for d in parts if steps // d * block_k <= 2048)
    for d in parts:
        if split < d <= 16 and programs * split < 1024:
            split = d
    y = torch.empty(split, width, dtype=torch.float32, device=x.device)

    def grid(_meta: dict[str, object]) -> tuple[int, int]:
        return (programs, split)

    wrap_triton(_fp8_gemv_kernel)[grid](
        x.contiguous(),
        weight.contiguous(),
        scale.float().contiguous(),
        y,
        width,
        depth,
        block_n=block_n,
        block_k=block_k,
        num_warps=1 if small else 4,
        num_stages=4 if small else 2,
    )
    return torch.sum(y, dim=0, dtype=x.dtype).reshape(1, width)


@triton.jit
def _fp8_rows_kernel(x_ptr, q_ptr, scale_ptr, depth, block: tl.constexpr):
    # One row per program: its largest magnitude, then the row over its
    # scale (that magnitude over 448, FP8's largest value) rounded to FP8.
    row = tl.program_id(0).to(tl.int64)
    base = row * depth
    top = tl.zeros((block,), dtype=tl.float32)
    for k0 in range(0, depth, block):
        offsets = k0 + tl.arange(0, block)
        x = tl.load(x_ptr + base + offsets, mask=offsets < depth, other=0.0).to(tl.float32)
        top = tl.maximum(top, tl.abs(x))
    scale = tl.maximum(tl.max(top, axis=0), 1e-12) / 448.0
    for k0 in range(0, depth, block):
        offsets = k0 + tl.arange(0, block)
        live = offsets < depth
        x = tl.load(x_ptr + base + offsets, mask=live, other=0.0).to(tl.float32)
        q = tl.clamp(x / scale, -448.0, 448.0).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + base + offsets, q, mask=live)
    tl.store(scale_ptr + row, scale)


@triton_op("linnet::fp8_rows", mutates_args=())
def fp8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row of `x` [rows, In] rounded to FP8 E4M3 with a scale of its
    own: the rounded rows and the scales ([rows, 1], f32), in one kernel,
    as `torch._scaled_mm` takes them."""
    rows, depth = x.shape
    q = torch.empty(rows, depth, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(rows, 1, dtype=torch.float32, device=x.device)

    def grid(_meta: dict[str, object]) -> tuple[int]:
        return (rows,)

    wrap_triton(_fp8_rows_kernel)[grid](x.contiguous(), q, scale, depth, block=1024, num_warps=4)
    return q, scale


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


__all__ = ["int4_linear", "mxfp4_experts", "one_shot_all_reduce_kernel"]
