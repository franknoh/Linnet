"""`std.quant`'s 8-bit linear layers as the generated PyTorch runs them:
`linear_fp8` (a scale a row), `linear_fp8_blocks` (a scale a block of 128 by
128), `linear_int8` (a scale a row) and `linear_int8_groups` (a scale per
group of inputs).

Each layer is two calls. `*_operands` does the weight-only work -- the
weight's bytes as the kernels read them, the scales in f32 -- which
`prepare` runs once at load; `*_linear` multiplies. With Triton on CUDA,
in bf16 or f16:

- one row (a decoding step of one sequence): `kernels.gemv8`, the weight
  read once and widened in registers on CUDA cores;
- up to 64 rows: `kernels.w8a16`, the weight widened to the input's dtype
  tile by tile on tensor cores;
- more rows: the weight dequantized for the call (`kernels.dequantize8`),
  then cuBLAS;
- past 32 rows and `rounded` (`numerics="fast"`): the input rounded to 8
  bits too and the two multiplied in 8 bits: FP8 by `torch._scaled_mm` (a
  scale per row of the input, or per 128 of its inputs against block
  weights), int8 by `kernels.int8_mm`. Up to 32 rows, reading the weight
  bounds the product either way.

Anywhere else the weight is dequantized for the call and multiplied by
`F.linear`.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as functional

from ..quant import FEW_ROWS, TILED_ROWS

try:
    from . import kernels
except ImportError:  # no Triton: the dequantized weights everywhere
    kernels = None

_HALF = (torch.bfloat16, torch.float16)


class _ScaledMM(Protocol):
    """`torch._scaled_mm`: `a @ b` of FP8 operands times their scales."""

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor: ...


_scaled_mm: _ScaledMM = getattr(torch, "_scaled_mm")  # noqa: B009 - private

# Each kind's operands: the weight, what the kernels read alongside it, and
# two flags (`rounded`: the input may be rounded to 8 bits; the last one
# says whether the kernels can run, as far as the weights decide it).
Fp8Operands = tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]
Fp8BlockOperands = tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]
Int8Operands = tuple[torch.Tensor, torch.Tensor, int, int]
Int8GroupOperands = tuple[torch.Tensor, torch.Tensor, int, int]


def _capability(weight: torch.Tensor) -> tuple[int, int]:
    if kernels is None or not weight.is_cuda:
        return (0, 0)
    return torch.cuda.get_device_capability(weight.device)


def _fp8_kernels(weight: torch.Tensor) -> int:
    """2: Linnet's FP8 kernels and FP8 products (`torch._scaled_mm`) run
    here; 1: the kernels only (compute capability 8.9); 0: neither."""
    capability = _capability(weight)
    out_features, in_features = weight.shape
    if in_features % 128 or out_features % 16:
        return 0
    return 2 if capability >= (9, 0) else 1 if capability >= (8, 9) else 0


def _int8_kernels(weight: torch.Tensor) -> int:
    out_features, in_features = weight.shape
    return int(_capability(weight) >= (8, 0) and in_features % 128 == 0 and out_features % 16 == 0)


def _weight_only(
    flat: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    fp8: bool,
    group: int = 0,
    row_block: int = 1,
) -> torch.Tensor:
    assert kernels is not None
    rows = flat.shape[0]
    if rows == 1:
        return kernels.gemv8(flat, weight, scale, fp8, group, row_block)
    if rows <= TILED_ROWS:
        return kernels.w8a16(flat, weight, scale, fp8, group, row_block)
    dequantized = kernels.dequantize8(weight, scale, fp8, group, row_block, flat.dtype)
    return functional.linear(flat, dequantized)


def _finish(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    y = y.reshape(*x.shape[:-1], y.shape[-1])
    return y if bias is None else y + bias


def fp8_operands(weight: torch.Tensor, scale: torch.Tensor, rounded: int) -> Fp8Operands:
    """`weight` [Out, In] FP8 E4M3 bytes (`u8`), `scale` [Out]: the bytes,
    the same read as FP8 and transposed (`torch._scaled_mm`'s operand), the
    scales as f32 [1, Out]."""
    weight = weight.contiguous()
    transposed = weight.view(torch.float8_e4m3fn).t()
    row = scale.float().reshape(1, -1).contiguous()
    return weight, transposed, row, rounded, _fp8_kernels(weight)


def fp8_linear(x: torch.Tensor, operands: Fp8Operands, bias: torch.Tensor | None) -> torch.Tensor:
    weight, transposed, scale, rounded, fast = operands
    flat = x.reshape(-1, x.shape[-1])
    if fast and kernels is not None and flat.dtype in _HALF:
        if rounded and fast == 2 and flat.shape[0] > FEW_ROWS:
            q, rows = kernels.fp8_rows(flat)
            y = _scaled_mm(q, transposed, scale_a=rows, scale_b=scale, out_dtype=flat.dtype)
        else:
            y = _weight_only(flat, weight, scale, True)
    else:
        decoded = weight.view(torch.float8_e4m3fn).float() * scale.reshape(-1, 1)
        y = functional.linear(flat, decoded.to(flat.dtype))
    return _finish(x, y, bias)


def fp8_block_operands(weight: torch.Tensor, scale: torch.Tensor, rounded: int) -> Fp8BlockOperands:
    """`weight` [Out, In] FP8 E4M3 bytes, `scale` [Out / B, In / B]: the
    bytes, the same as FP8 transposed, the scales as f32, and B."""
    weight = weight.contiguous()
    transposed = weight.view(torch.float8_e4m3fn).t()
    block = weight.shape[0] // scale.shape[0]
    fast = _fp8_kernels(weight) if weight.shape[1] == block * scale.shape[1] else 0
    # PyTorch's block-scaled FP8 product takes blocks of 128 only.
    fast = min(fast, 1) if block != 128 else fast
    return weight, transposed, scale.float().contiguous(), block, rounded, fast


def fp8_block_linear(
    x: torch.Tensor, operands: Fp8BlockOperands, bias: torch.Tensor | None
) -> torch.Tensor:
    weight, transposed, scale, block, rounded, fast = operands
    flat = x.reshape(-1, x.shape[-1])
    if fast and kernels is not None and flat.dtype in _HALF:
        rows = flat.shape[0]
        if rounded and fast == 2 and rows > FEW_ROWS:
            # cuBLAS takes the block-scaled product of rows in fours.
            padded = -(-rows // 4) * 4
            q, scales = kernels.fp8_groups(flat, block, padded)
            y = _scaled_mm(q, transposed, scale_a=scales, scale_b=scale.t(), out_dtype=flat.dtype)
            y = y[:rows] if padded != rows else y
        else:
            y = _weight_only(flat, weight, scale, True, block, block)
    else:
        out_features, in_features = weight.shape
        decoded = weight.view(torch.float8_e4m3fn).float()
        decoded = decoded.reshape(out_features // block, block, in_features // block, block)
        decoded = decoded * scale[:, None, :, None]
        y = functional.linear(flat, decoded.reshape(out_features, in_features).to(flat.dtype))
    return _finish(x, y, bias)


def int8_operands(weight: torch.Tensor, scale: torch.Tensor, rounded: int) -> Int8Operands:
    """`weight` [Out, In] `i8`, `scale` [Out]: the weight and the scales as
    f32."""
    weight = weight.contiguous()
    return weight, scale.float().reshape(-1).contiguous(), rounded, _int8_kernels(weight)


def int8_linear(x: torch.Tensor, operands: Int8Operands, bias: torch.Tensor | None) -> torch.Tensor:
    weight, scale, rounded, fast = operands
    flat = x.reshape(-1, x.shape[-1])
    if fast and kernels is not None and flat.dtype in _HALF:
        if rounded and flat.shape[0] > FEW_ROWS:
            q, rows = kernels.int8_rows(flat)
            y = kernels.int8_mm(q, weight, rows, scale, flat.dtype)
        else:
            y = _weight_only(flat, weight, scale, False)
    else:
        y = functional.linear(flat, (weight.float() * scale[:, None]).to(flat.dtype))
    return _finish(x, y, bias)


def int8_group_operands(weight: torch.Tensor, scale: torch.Tensor) -> Int8GroupOperands:
    """`weight` [Out, In] `i8`, `scale` [Out, In / group]: the weight, the
    scales as f32, and the group."""
    weight = weight.contiguous()
    group = weight.shape[1] // scale.shape[1]
    fast = _int8_kernels(weight) if group % 16 == 0 else 0
    return weight, scale.float().contiguous(), group, fast


def int8_group_linear(
    x: torch.Tensor, operands: Int8GroupOperands, bias: torch.Tensor | None
) -> torch.Tensor:
    weight, scale, group, fast = operands
    flat = x.reshape(-1, x.shape[-1])
    if fast and kernels is not None and flat.dtype in _HALF:
        y = _weight_only(flat, weight, scale, False, group)
    else:
        out_features, in_features = weight.shape
        grouped = weight.float().reshape(out_features, in_features // group, group)
        decoded = (grouped * scale[:, :, None]).reshape(out_features, in_features)
        y = functional.linear(flat, decoded.to(flat.dtype))
    return _finish(x, y, bias)
