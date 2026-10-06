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

Rounding to nearest is the plainest scheme. Checkpoints quantized with
calibration repack into the same layout: `import_quantized` reads a 4-bit
GPTQ or AWQ checkpoint (its `config.json`'s `quantization_config` says
which) and writes the same tensors, plus `.order` for a GPTQ checkpoint in
activation order, whose groups were formed over the inputs in another
order than their own:

    from linnet.quant import import_quantized

    import_quantized("Llama-3.1-8B-Instruct-GPTQ-INT4/", "model-int4.safetensors",
                     bindings="bindings.json")
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import fnmatch
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .compiler import LinnetError
from .dtypes import DTYPES
from .weights import (
    RawTensor,
    decode_floats,
    encode_floats,
    iter_safetensors,
    read_bindings,
    write_safetensors,
)


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
    paths_of = _paths_of(bindings)
    chosen = list(patterns)
    out: list[tuple[str, str, tuple[int, ...], bytes]] = []
    quantized: list[str] = []
    for tensor in iter_safetensors(weights):
        for path in paths_of.get(tensor.name, [tensor.name]):
            if path.endswith(".weight") and any(fnmatch.fnmatchcase(path, p) for p in chosen):
                values = decode_floats(tensor.dtype, tensor.data).reshape(tensor.shape)
                packed, scale, zero = quantize_int4_groups(values, group)
                out += _int4_tensors(path.removesuffix(".weight"), packed, scale, zero, dtype)
                quantized.append(path)
            else:
                out.append((path, tensor.dtype, tensor.shape, tensor.data))
    if not quantized:
        raise LinnetError("no weight matched " + ", ".join(chosen))
    write_safetensors(output, out, metadata={"format": "pt"})
    return quantized


def _paths_of(bindings: str | Path | Mapping[str, str] | None) -> dict[str, list[str]]:
    """Checkpoint name -> the Linnet paths bound to it (a tied output head
    and embedding share one)."""
    mapping = (
        dict(bindings)
        if isinstance(bindings, Mapping)
        else read_bindings(bindings)
        if bindings is not None
        else {}
    )
    paths_of: dict[str, list[str]] = {}
    for path, name in mapping.items():
        paths_of.setdefault(name, []).append(path)
    return paths_of


def _int4_tensors(
    prefix: str, packed: np.ndarray, scale: np.ndarray, zero: np.ndarray, dtype: str
) -> list[tuple[str, str, tuple[int, ...], bytes]]:
    if dtype not in ("bf16", "f16", "f32"):
        raise LinnetError(f"scales are stored as bf16, f16, or f32, not {dtype}")
    scale_dtype = DTYPES[dtype].safetensors
    scale_bytes = encode_floats(scale, scale_dtype)
    return [
        (prefix + ".weight", "U8", tuple(packed.shape), packed.tobytes()),
        (prefix + ".scale", scale_dtype, tuple(scale.shape), scale_bytes),
        (prefix + ".zero", "U8", tuple(zero.shape), zero.tobytes()),
    ]


# AWQ packs eight outputs to an int32 interleaved: bits `4k` hold output
# `_AWQ_ORDER[k]` of the eight. Taking them back is this inverse.
_AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
_AWQ_REVERSE = tuple(_AWQ_ORDER.index(k) for k in range(8))


def _nibbles(words: np.ndarray, axis: int) -> np.ndarray:
    """The eight 4-bit values of each int32, lowest bits first, laid along
    `axis` (the packed axis grows eightfold)."""
    bits = words.astype(np.int64) & 0xFFFFFFFF
    shifts = np.arange(0, 32, 4, dtype=np.int64)
    values = (np.expand_dims(bits, -1) >> shifts) & 15  # [..., 8]
    values = np.moveaxis(values, -1, axis + 1)
    shape = list(words.shape)
    shape[axis] *= 8
    return values.reshape(shape).astype(np.uint8)


def pack_int4_groups(q: np.ndarray, group: int) -> np.ndarray:
    """`[Out, In]` values in `0..15` as `Int4GroupLinear`'s `[Out, In /
    group, group / 2]` bytes, low nibble first."""
    out_features, in_features = q.shape
    grouped = q.reshape(out_features, in_features // group, group).astype(np.uint8)
    return (grouped[..., 0::2] | (grouped[..., 1::2] << 4)).astype(np.uint8)


def import_quantized(
    weights: str | Path,
    output: str | Path,
    *,
    bindings: str | Path | Mapping[str, str] | None = None,
    dtype: str = "bf16",
    config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Writes `output` with a 4-bit GPTQ or AWQ checkpoint's quantized linear
    layers repacked for `Int4GroupLinear` (`.weight`, `.scale` in `dtype`,
    `.zero`, and `.order` when a GPTQ layer's groups follow activation
    order) and every other tensor as it is, all under Linnet paths.
    `config` is the checkpoint's `quantization_config`; by default it is
    read from the `config.json` beside `weights`. Returns the quantized
    paths, as `.weight`."""
    if config is None:
        location = Path(weights)
        directory = location if location.is_dir() else location.parent
        try:
            document = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise LinnetError(f"no config.json beside {weights}; pass `config`") from None
        config = document.get("quantization_config")
        if not isinstance(config, Mapping):
            raise LinnetError(f"{directory / 'config.json'} has no quantization_config")
    method = str(config.get("quant_method", "")).lower()
    if method not in ("gptq", "awq"):
        raise LinnetError(f"a {method or 'unnamed'} checkpoint is not GPTQ or AWQ")
    if int(config.get("bits", config.get("w_bit", 4))) != 4:
        raise LinnetError("only 4-bit GPTQ and AWQ checkpoints repack into Int4GroupLinear")
    if method == "awq" and str(config.get("version", "gemm")).lower() != "gemm":
        raise LinnetError(f"AWQ's {config.get('version')} packing is not read; GEMM's is")
    group = int(config.get("group_size", config.get("q_group_size", 128)))
    # GPTQ's first checkpoint format stores each zero point less one.
    zero_offset = 1 if method == "gptq" and config.get("checkpoint_format", "gptq") == "gptq" else 0
    tensors: dict[str, RawTensor] = {tensor.name: tensor for tensor in iter_safetensors(weights)}
    paths_of = _paths_of(bindings)
    out: list[tuple[str, str, tuple[int, ...], bytes]] = []
    quantized: list[str] = []
    parts = (".qweight", ".qzeros", ".scales", ".g_idx")
    layers = sorted(name.removesuffix(".qweight") for name in tensors if name.endswith(".qweight"))
    taken = {layer + part for layer in layers for part in parts}
    for layer in layers:
        qweight = tensors[layer + ".qweight"]
        qzeros = tensors[layer + ".qzeros"]
        stored = tensors[layer + ".scales"]
        scales = decode_floats(stored.dtype, stored.data).reshape(stored.shape)
        words = np.frombuffer(qweight.data, dtype=np.int32).reshape(qweight.shape)
        zero_words = np.frombuffer(qzeros.data, dtype=np.int32).reshape(qzeros.shape)
        order: np.ndarray | None = None
        if method == "gptq":
            q = _nibbles(words, 0)  # [In, Out]
            zero = _nibbles(zero_words, 1).astype(np.int32) + zero_offset  # [Groups, Out]
            in_features = q.shape[0]
            group_of = np.arange(in_features) // (in_features if group <= 0 else group)
            if layer + ".g_idx" in tensors:
                g_idx = tensors[layer + ".g_idx"]
                ids = np.frombuffer(g_idx.data, dtype=np.int32).reshape(g_idx.shape)
                if not np.array_equal(ids, group_of):
                    # Activation order: the groups hold inputs in another
                    # order. Taken in that order, each group is contiguous.
                    order = np.argsort(ids, kind="stable").astype(np.int32)
                    if not np.array_equal(ids[order], group_of):
                        raise LinnetError(f"{layer}: its groups are not all the same size")
                    q = q[order]
        else:
            columns = np.arange(words.shape[1] * 8).reshape(-1, 8)[:, _AWQ_REVERSE].reshape(-1)
            q = _nibbles(words, 1)[:, columns]  # [In, Out]
            zero = _nibbles(zero_words, 1)[:, columns].astype(np.int32)
        in_features, out_features = q.shape
        size = in_features if group <= 0 else group
        if in_features % size != 0 or zero.shape != (in_features // size, out_features):
            raise LinnetError(f"{layer}: {in_features} inputs do not make groups of {size}")
        if zero.max(initial=0) > 255:
            raise LinnetError(f"{layer}: a zero point past 255")
        packed = pack_int4_groups(np.ascontiguousarray(q.T), size)
        weight_name = layer + ".weight"
        for path in paths_of.get(weight_name, [weight_name]):
            prefix = path.removesuffix(".weight")
            out += _int4_tensors(
                prefix,
                packed,
                np.ascontiguousarray(scales.T),
                np.ascontiguousarray(zero.T).astype(np.uint8),
                dtype,
            )
            if order is not None:
                out.append((prefix + ".order", "I32", (in_features,), order.tobytes()))
            quantized.append(path)
    for name, tensor in tensors.items():
        if name in taken:
            continue
        for path in paths_of.get(name, [name]):
            out.append((path, tensor.dtype, tensor.shape, tensor.data))
    if not quantized:
        raise LinnetError(f"no {method.upper()} layer (`.qweight`) in {weights}")
    write_safetensors(output, out, metadata={"format": "pt"})
    return quantized


__all__ = [
    "dequantize_int4_groups",
    "import_quantized",
    "pack_int4_groups",
    "quantize_checkpoint",
    "quantize_int4_groups",
]
