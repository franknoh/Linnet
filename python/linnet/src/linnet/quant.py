"""Group-wise 4-bit weights for `std.quant::Int4GroupLinear`.

`quantize_int4_groups` rounds one `[Out, In]` weight to the nearest of 16
levels per group of `group` consecutive inputs (asymmetric: a scale and a
zero point per group), in the layout the block declares:

    weight  [Out, In / group, group / 2]  u8, two values a byte, low nibble first
    scale   [Out, In / group]             the model's float dtype
    zero    [Out, In / group]             u8, in 0..15

`quantize_checkpoint` does that to the linear weights of a SafeTensors
checkpoint and writes the result under Linnet parameter paths, next to every
other tensor, copied as it is. A block declared with `Int4GroupLinear` where
it had `Linear` then loads it with the same paths:

    from linnet.quant import quantize_checkpoint

    quantize_checkpoint("model.safetensors", "model-int4.safetensors",
                        patterns=["*_proj.weight"], group=128, dtype="bf16",
                        bindings="bindings.json")

Rounding to nearest is the plainest scheme; checkpoints quantized with
calibration (GPTQ, AWQ) repack into the same layout.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from .compiler import LinnetError
from .weights import iter_safetensors, read_bindings, write_safetensors

_FLOATS = {"F32": np.float32, "F16": np.float16, "F64": np.float64}


def quantize_int4_groups(
    weight: np.ndarray, group: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(packed, scale, zero)` for a `[Out, In]` weight; `scale` is f32."""
    if weight.ndim != 2:
        raise LinnetError(f"a linear weight is [Out, In]; this one is {list(weight.shape)}")
    out_features, in_features = weight.shape
    if group <= 0 or group % 2 != 0 or in_features % group != 0:
        raise LinnetError(f"groups of {group} must be even and divide the {in_features} inputs")
    grouped = weight.astype(np.float32).reshape(out_features, in_features // group, group)
    low = grouped.min(axis=-1)
    high = grouped.max(axis=-1)
    scale = (high - low) / 15.0
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    zero = np.clip(np.round(-low / scale), 0, 15).astype(np.uint8)
    q = np.clip(np.round(grouped / scale[..., None]) + zero[..., None], 0, 15).astype(np.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    return packed, scale, zero


def dequantize_int4_groups(packed: np.ndarray, scale: np.ndarray, zero: np.ndarray) -> np.ndarray:
    """The f32 `[Out, In]` weight the block computes with."""
    out_features, groups, half = packed.shape
    q = np.stack([packed & 15, packed >> 4], axis=-1).reshape(out_features, groups, 2 * half)
    weight = (q.astype(np.float32) - zero[..., None].astype(np.float32)) * scale[..., None]
    return weight.reshape(out_features, groups * 2 * half)


def _to_f32(dtype: str, data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    if dtype == "BF16":
        bits = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).reshape(shape)
    if dtype not in _FLOATS:
        raise LinnetError(f"cannot quantize a {dtype} tensor")
    return np.frombuffer(data, dtype=_FLOATS[dtype]).astype(np.float32).reshape(shape)


def _from_f32(values: np.ndarray, dtype: str) -> tuple[str, bytes]:
    if dtype == "bf16":
        bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> 16
        return "BF16", rounded.astype(np.uint16).tobytes()
    names = {"f32": ("F32", np.float32), "f16": ("F16", np.float16)}
    if dtype not in names:
        raise LinnetError(f"scales are stored as bf16, f16, or f32, not {dtype}")
    name, numpy_dtype = names[dtype]
    return name, np.ascontiguousarray(values, dtype=numpy_dtype).tobytes()


def quantize_checkpoint(
    weights: str | Path,
    output: str | Path,
    *,
    patterns: Iterable[str],
    group: int = 128,
    dtype: str = "bf16",
    bindings: str | Path | Mapping[str, str] | None = None,
) -> list[str]:
    """Writes `output` with every tensor of `weights` under its Linnet path
    (`bindings` maps paths to checkpoint names, as for loading), quantizing
    the `.weight` tensors whose path matches one of `patterns` into
    `.weight`, `.scale`, and `.zero`. Returns the quantized paths."""
    mapping = (
        dict(bindings)
        if isinstance(bindings, Mapping)
        else read_bindings(bindings)
        if bindings is not None
        else {}
    )
    path_of = {name: path for path, name in mapping.items()}
    chosen = list(patterns)
    out: list[tuple[str, str, tuple[int, ...], bytes]] = []
    quantized: list[str] = []
    for tensor in iter_safetensors(weights):
        path = path_of.get(tensor.name, tensor.name)
        if path.endswith(".weight") and any(fnmatch.fnmatchcase(path, p) for p in chosen):
            values = _to_f32(tensor.dtype, tensor.data, tensor.shape)
            packed, scale, zero = quantize_int4_groups(values, group)
            prefix = path.removesuffix(".weight")
            scale_dtype, scale_bytes = _from_f32(scale, dtype)
            out.append((prefix + ".weight", "U8", tuple(packed.shape), packed.tobytes()))
            out.append((prefix + ".scale", scale_dtype, tuple(scale.shape), scale_bytes))
            out.append((prefix + ".zero", "U8", tuple(zero.shape), zero.tobytes()))
            quantized.append(path)
        else:
            out.append((path, tensor.dtype, tensor.shape, tensor.data))
    if not quantized:
        raise LinnetError("no weight matched " + ", ".join(chosen))
    write_safetensors(output, out, metadata={"format": "pt"})
    return quantized


__all__ = ["dequantize_int4_groups", "quantize_checkpoint", "quantize_int4_groups"]
