"""The storage formats MLX has no dtype for, as `linnet mlx` holds them: the
4-bit formats (`i4`, `u4`, `f4e2m1`) as bytes, two elements a byte along the
last axis as in `linnet.torch.packed`, and FP8 E5M2 as its byte, which is
the high byte of an f16. (FP8 E4M3 is MLX's own `from_fp8` and `to_fp8`.)

`unpack4` widens: `i4` to `int8`, `u4` to `uint8`, `f4e2m1` to `float32`.
`pack4` narrows an array of any dtype: an integer format wraps (a float
rounding toward zero first), FP4 E2M1 rounds to the nearest value, ties to
even, saturates at 6, and takes NaN to zero.
"""

from __future__ import annotations

import mlx.core as mx

# FP4 E2M1 by code: a sign bit, two exponent bits biased by 1, one mantissa
# bit, and no infinities or NaN.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FORMATS = ("i4", "u4", "f4e2m1")


def unpack4(packed: mx.array, kind: str) -> mx.array:
    """`packed`'s elements of format `kind`, widened."""
    if kind not in _FORMATS:
        raise ValueError(f"`{kind}` is not a 4-bit format ({', '.join(_FORMATS)})")
    pairs = mx.stack([packed & 15, packed >> 4], axis=-1)
    codes = mx.reshape(pairs, (*packed.shape[:-1], -1))
    if kind == "u4":
        return codes
    if kind == "i4":
        return (codes.astype(mx.int8) ^ 8) - 8
    # A mantissa bit `m` over exponent bits `e` is `m / 2` when `e` is 0,
    # else `(2 + m) * 2 ** (e - 2)`.
    wide = codes.astype(mx.int32)
    exponent = (wide >> 1) & 3
    mantissa = (wide & 1).astype(mx.float32)
    normal = (mantissa + 2) * mx.power(2.0, (exponent - 2).astype(mx.float32))
    magnitude = mx.where(exponent == 0, mantissa * 0.5, normal)
    return mx.where((wide & 8) != 0, -magnitude, magnitude)


def _round_e2m1(x: mx.array) -> mx.array:
    """The E2M1 code of each element of `x`."""
    wide = x.astype(mx.float32)
    nan = mx.isnan(wide)
    magnitude = mx.minimum(mx.where(nan, 0.0, mx.abs(wide)), 6.0)
    grid = mx.array(_E2M1, dtype=mx.float32)
    below = mx.sum(magnitude[..., None] >= grid, axis=-1).astype(mx.int32) - 1
    below = mx.minimum(below, len(_E2M1) - 2)
    low, high = mx.take(grid, below), mx.take(grid, below + 1)
    # The nearer of the two; at the midpoint, the even code (mantissa 0).
    up = (magnitude - low > high - magnitude) | (
        (magnitude - low == high - magnitude) & (below % 2 == 1)
    )
    code = mx.minimum(below + up.astype(mx.int32), len(_E2M1) - 1)
    negative = ((wide.view(mx.uint32) >> 31) == 1) & ~nan
    return mx.where(negative, code | 8, code)


def pack4(x: mx.array, kind: str) -> mx.array:
    """`x` narrowed to format `kind`, as bytes: its last axis (even) halved."""
    if kind not in _FORMATS:
        raise ValueError(f"`{kind}` is not a 4-bit format ({', '.join(_FORMATS)})")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"a 4-bit array's last axis is even, found shape {tuple(x.shape)}")
    if kind == "f4e2m1":
        codes = _round_e2m1(x)
    elif mx.issubdtype(x.dtype, mx.floating):
        codes = mx.trunc(x).astype(mx.int64) & 15
    else:
        codes = x.astype(mx.int64) & 15
    pairs = mx.reshape(codes.astype(mx.uint8), (*codes.shape[:-1], -1, 2))
    return pairs[..., 0] | (pairs[..., 1] << 4)


def from_e5m2(data: mx.array) -> mx.array:
    """FP8 E5M2 bytes as the f16 values they are."""
    return (data.astype(mx.uint16) << 8).view(mx.float16)


def to_e5m2(x: mx.array) -> mx.array:
    """`x` as FP8 E5M2 bytes: rounded to f16, then to E5M2's two mantissa
    bits, to nearest, ties to even; past the largest value, infinity."""
    half = x.astype(mx.float16)
    bits = half.view(mx.uint16)
    rounded = (bits + 0x7F + ((bits >> 8) & 1)) >> 8
    return mx.where(mx.isnan(half), 0x7E, rounded).astype(mx.uint8)


__all__ = ["from_e5m2", "pack4", "to_e5m2", "unpack4"]
