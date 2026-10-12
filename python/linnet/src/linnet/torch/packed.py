"""The 4-bit storage formats (`std.dtypes`: `i4`, `u4`, `f4e2m1`) as PyTorch
holds them: their bytes, `uint8` with the last axis halved, element `2k`
in the low four bits and `2k + 1` in the high four (spec §3.1).

`unpack4` widens: `i4` to `int8`, `u4` to `uint8`, `f4e2m1` to `float32`.
`pack4` narrows a tensor of any dtype: an integer format wraps (a float
rounding toward zero first), FP4 E2M1 rounds to the nearest value, ties to
even, saturates at 6, and takes NaN to zero.
"""

from __future__ import annotations

import torch

# FP4 E2M1 by code: a sign bit, two exponent bits biased by 1, one mantissa
# bit, and no infinities or NaN.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FORMATS = ("i4", "u4", "f4e2m1")


def _codes(packed: torch.Tensor) -> torch.Tensor:
    """Each byte's two codes, the low first, along the last axis."""
    pairs = torch.stack((packed & 15, packed >> 4), dim=-1)
    return pairs.flatten(-2)


def unpack4(packed: torch.Tensor, kind: str) -> torch.Tensor:
    """`packed`'s elements of format `kind`, widened."""
    if kind not in _FORMATS:
        raise ValueError(f"`{kind}` is not a 4-bit format ({', '.join(_FORMATS)})")
    codes = _codes(packed)
    if kind == "u4":
        return codes
    if kind == "i4":
        return (codes.to(torch.int8) ^ 8) - 8
    # From the bits, with no table to copy to the device each call: a
    # mantissa bit `m` over exponent bits `e` is `m / 2` when `e` is 0, else
    # `(2 + m) * 2 ** (e - 2)`.
    wide = codes.to(torch.int32)
    exponent = (wide >> 1) & 3
    mantissa = (wide & 1).to(torch.float32)
    normal = (mantissa + 2) * torch.exp2((exponent - 2).to(torch.float32))
    magnitude = torch.where(exponent == 0, mantissa * 0.5, normal)
    return torch.where((wide & 8) != 0, -magnitude, magnitude)


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """The E2M1 code of each element of `x`."""
    magnitude = torch.nan_to_num(x.float().abs(), nan=0.0).clamp(max=6.0)
    grid = torch.tensor(_E2M1, dtype=torch.float32, device=x.device)
    below = (torch.bucketize(magnitude, grid, right=True) - 1).clamp(max=len(_E2M1) - 2)
    low, high = grid[below], grid[below + 1]
    # The nearer of the two; at the midpoint, the even code (mantissa 0).
    up = (magnitude - low > high - magnitude) | (
        (magnitude - low == high - magnitude) & (below % 2 == 1)
    )
    code = (below + up.long()).clamp(max=len(_E2M1) - 1)
    negative = torch.signbit(x) & ~torch.isnan(x)
    return torch.where(negative, code | 8, code)


def pack4(x: torch.Tensor, kind: str) -> torch.Tensor:
    """`x` narrowed to format `kind`, as bytes: its last axis (even) halved."""
    if kind not in _FORMATS:
        raise ValueError(f"`{kind}` is not a 4-bit format ({', '.join(_FORMATS)})")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"a 4-bit tensor's last axis is even, found shape {tuple(x.shape)}")
    if kind == "f4e2m1":
        codes = _round_e2m1(x)
    elif x.is_floating_point():
        codes = x.trunc().to(torch.int64) & 15
    else:
        codes = x.to(torch.int64) & 15
    pairs = codes.to(torch.uint8).reshape(*codes.shape[:-1], -1, 2)
    return pairs[..., 0] | (pairs[..., 1] << 4)


__all__ = ["pack4", "unpack4"]
