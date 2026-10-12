"""Group-wise 4-bit weights for `std.quant::Int4GroupLinear`, and 8-bit
ones for `std.quant::Int8GroupLinear`.

`quantize_int4_groups` rounds one `[Out, In]` weight to the nearest of 16
levels per group of `group` consecutive inputs (asymmetric: a scale and a
zero point per group), in the layout the block declares:

    weight  [Out, In / group, group / 2]  u8, two values a byte, low nibble first
    scale   [Out, In / group]             the model's float dtype
    zero    [Out, In / group]             u8, in 0..15

`quantize_int8_groups` rounds one to symmetric int8 instead, a scale per
group:

    weight  [Out, In]                     i8, in -127..127
    scale   [Out, In / group]             the model's float dtype

`quantize_checkpoint` does that to the linear weights of a SafeTensors
checkpoint and writes the result under Linnet parameter paths, next to every
other tensor, copied as it is. A block declared with `Int4GroupLinear` (or
`Int8GroupLinear`, with `bits=8`) where it had `Linear` then loads it with
the same paths:

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

A bitsandbytes NF4 checkpoint (QLoRA's) imports the same way, for
`std.quant::Nf4Linear`:

    weight  [Out, In]           nf4, two codes a byte, low nibble first
    absmax  [Out, In / Block]   f32, each block's scale
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import SupportsInt, cast

import numpy as np

from .compiler import LinnetError
from .dtypes import DTYPES
from .weights import (
    RawTensor,
    decode_floats,
    encode_floats,
    iter_safetensors,
    paths_by_tensor,
    read_bindings,
    write_safetensors,
)

# Where PyTorch switches 8-bit linear layers' kernels (`linnet.torch.quantized`),
# measured on an H100 over Llama 3.1 8B's projections, and read by the
# resource model too: up to `FEW_ROWS` a call's rows multiply the weight as it
# is (rounding them to 8 bits pays only past it), and up to `TILED_ROWS`
# Linnet's tensor-core kernel beats cuBLAS over the weight dequantized for
# the call.
FEW_ROWS = 32
TILED_ROWS = 64


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
    q = np.clip(np.round(grouped / scale[..., None]) + zero[..., None], 0, 15)
    return pack_int4_groups(q.reshape(out_features, in_features), group), scale, zero


def quantize_int8_groups(
    weight: np.ndarray, group: int = 128, dtype: str = "f32"
) -> tuple[np.ndarray, np.ndarray]:
    """`(q, scale)` for a `[Out, In]` weight: the largest magnitude of each
    group over 127, rounded to `dtype` (the dtype it is stored in, which the
    levels are then rounded against), and `q` the nearest levels; `scale`
    is f32."""
    if weight.ndim != 2:
        raise LinnetError(f"a linear weight is [Out, In]; this one is {list(weight.shape)}")
    out_features, in_features = weight.shape
    if group <= 0 or in_features % group != 0:
        raise LinnetError(f"groups of {group} must divide the {in_features} inputs")
    grouped = weight.astype(np.float32).reshape(out_features, in_features // group, group)
    scale = np.abs(grouped).max(axis=-1) / 127.0
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    stored = DTYPES[dtype].safetensors
    scale = decode_floats(stored, encode_floats(scale, stored)).reshape(scale.shape)
    q = np.clip(np.round(grouped / scale[..., None]), -127, 127).astype(np.int8)
    return q.reshape(out_features, in_features), scale


def dequantize_int8_groups(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The f32 `[Out, In]` weight `Int8GroupLinear` computes with."""
    out_features, in_features = q.shape
    groups = scale.shape[1]
    grouped = q.astype(np.float32).reshape(out_features, groups, in_features // groups)
    return (grouped * scale[..., None]).reshape(out_features, in_features)


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
    bits: int = 4,
) -> list[str]:
    """Writes `output` with every tensor of `weights` under its Linnet path
    (`bindings` maps paths to checkpoint names, as for loading), quantizing
    the `.weight` tensors whose path matches one of `patterns`: into
    `.weight`, `.scale` and `.zero` for `Int4GroupLinear` (`bits` 4), or
    `.weight` and `.scale` for `Int8GroupLinear` (`bits` 8). Returns the
    quantized paths."""
    if bits not in (4, 8):
        raise LinnetError(f"weights are quantized to 4 or 8 bits, not {bits}")
    if dtype not in ("bf16", "f16", "f32"):
        raise LinnetError(f"scales are stored as bf16, f16, or f32, not {dtype}")
    paths_of = _paths_of(bindings)
    chosen = list(patterns)
    out: list[tuple[str, str, tuple[int, ...], bytes]] = []
    quantized: list[str] = []
    for tensor in iter_safetensors(weights):
        for path in paths_of.get(tensor.name, [tensor.name]):
            if path.endswith(".weight") and any(fnmatch.fnmatchcase(path, p) for p in chosen):
                values = decode_floats(tensor.dtype, tensor.data).reshape(tensor.shape)
                prefix = _prefix(path)
                if bits == 8:
                    q, scale = quantize_int8_groups(values, group, dtype)
                    out.append((prefix + "weight", "I8", tuple(q.shape), q.tobytes()))
                    out.append(_scale_tensor(prefix, scale, dtype))
                else:
                    packed, scale, zero = quantize_int4_groups(values, group)
                    out += _int4_tensors(prefix, packed, scale, zero, dtype)
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
    return paths_by_tensor(mapping)


def _prefix(path: str) -> str:
    """What names a weight's siblings: `proj.` for `proj.weight`, nothing
    for a root block's own `weight`."""
    if path == "weight":
        return ""
    return path.removesuffix("weight") if path.endswith(".weight") else path + "."


def _scale_tensor(
    prefix: str, scale: np.ndarray, dtype: str
) -> tuple[str, str, tuple[int, ...], bytes]:
    scale_dtype = DTYPES[dtype].safetensors
    return (prefix + "scale", scale_dtype, tuple(scale.shape), encode_floats(scale, scale_dtype))


def _int4_tensors(
    prefix: str, packed: np.ndarray, scale: np.ndarray, zero: np.ndarray, dtype: str
) -> list[tuple[str, str, tuple[int, ...], bytes]]:
    if dtype not in ("bf16", "f16", "f32"):
        raise LinnetError(f"scales are stored as bf16, f16, or f32, not {dtype}")
    return [
        (prefix + "weight", "U8", tuple(packed.shape), packed.tobytes()),
        _scale_tensor(prefix, scale, dtype),
        (prefix + "zero", "U8", tuple(zero.shape), zero.tobytes()),
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


def _setting(config: Mapping[str, object], key: str, alias: str, default: int) -> int:
    """`config[key]` (or `config[alias]`, its AWQ name, or `default`) as an integer."""
    value = config.get(key, config.get(alias, default))
    if not isinstance(value, str | SupportsInt):
        raise LinnetError(f"the quantization_config's `{key}` is {value!r}, not a number")
    return int(value)


def import_quantized(
    weights: str | Path,
    output: str | Path,
    *,
    bindings: str | Path | Mapping[str, str] | None = None,
    dtype: str = "bf16",
    config: Mapping[str, object] | None = None,
) -> list[str]:
    """Writes `output` with a 4-bit GPTQ or AWQ checkpoint's quantized linear
    layers repacked for `Int4GroupLinear` (`.weight`, `.scale` in `dtype`,
    `.zero`, and `.order` when a GPTQ layer's groups follow activation
    order), or a bitsandbytes NF4 checkpoint's for `Nf4Linear` (`.weight`,
    `.absmax`), and every other tensor as it is, all under Linnet paths.
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
    if method == "bitsandbytes":
        return _import_bitsandbytes(weights, output, config, _paths_of(bindings))
    if method not in ("gptq", "awq"):
        raise LinnetError(f"a {method or 'unnamed'} checkpoint is not GPTQ, AWQ or bitsandbytes")
    if _setting(config, "bits", "w_bit", 4) != 4:
        raise LinnetError("only 4-bit GPTQ and AWQ checkpoints repack into Int4GroupLinear")
    if method == "awq" and str(config.get("version", "gemm")).lower() != "gemm":
        raise LinnetError(f"AWQ's {config.get('version')} packing is not read; GEMM's is")
    group = _setting(config, "group_size", "q_group_size", 128)
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
            group_of = np.arange(in_features) // (in_features if group <= 0 else group)  # pyright: ignore[reportUnknownVariableType]
            if layer + ".g_idx" in tensors:
                g_idx = tensors[layer + ".g_idx"]
                ids = np.frombuffer(g_idx.data, dtype=np.int32).reshape(g_idx.shape)
                if not np.array_equal(ids, group_of):  # pyright: ignore[reportUnknownArgumentType]
                    # Activation order: the groups hold inputs in another
                    # order. Taken in that order, each group is contiguous.
                    order = np.argsort(ids, kind="stable").astype(np.int32)
                    if not np.array_equal(ids[order], group_of):  # pyright: ignore[reportUnknownArgumentType]
                        raise LinnetError(f"{layer}: its groups are not all the same size")
                    q = q[order]
        else:
            columns = np.arange(words.shape[1] * 8).reshape(-1, 8)[:, _AWQ_REVERSE].reshape(-1)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
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
            prefix = _prefix(path)
            out += _int4_tensors(
                prefix,
                packed,
                np.ascontiguousarray(scales.T),
                np.ascontiguousarray(zero.T).astype(np.uint8),
                dtype,
            )
            if order is not None:
                out.append((prefix + "order", "I32", (in_features,), order.tobytes()))
            quantized.append(path)
    if not quantized:
        raise LinnetError(f"no {method.upper()} layer (`.qweight`) in {weights}")
    _write_with_the_rest(output, out, tensors, taken, paths_of)
    return quantized


def _write_with_the_rest(
    output: str | Path,
    out: list[tuple[str, str, tuple[int, ...], bytes]],
    tensors: Mapping[str, RawTensor],
    taken: set[str],
    paths_of: Mapping[str, list[str]],
) -> None:
    """Writes `out` and every tensor not `taken`, as it is, under its paths."""
    for name, tensor in tensors.items():
        if name in taken:
            continue
        for path in paths_of.get(name, [name]):
            out.append((path, tensor.dtype, tensor.shape, tensor.data))
    write_safetensors(output, out, metadata={"format": "pt"})


# NF4's values by code, as bitsandbytes stores them (`std.dtypes::nf4`).
NF4 = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.2461123019456863,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)

# A bitsandbytes 4-bit layer's tensors beside `.weight`; the state names the
# quantization type.
_BNB_PARTS = (".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map")
_BNB_STATE = ".quant_state.bitsandbytes__"


def _floats(tensor: RawTensor) -> np.ndarray:
    return decode_floats(tensor.dtype, tensor.data).reshape(tensor.shape).astype(np.float32)


def _bnb_state(tensor: RawTensor) -> dict[str, object]:
    """A layer's quantization state: JSON, its bytes held in a `U8` tensor."""
    state: object = json.loads(bytes(tensor.data).decode("utf-8"))
    if not isinstance(state, dict):
        raise LinnetError(f"`{tensor.name}` holds no quantization state")
    return {str(key): value for key, value in cast("dict[object, object]", state).items()}


def _bnb_nf4(
    weight: str, state: Mapping[str, object], tensors: Mapping[str, RawTensor]
) -> tuple[np.ndarray, np.ndarray]:
    """bitsandbytes NF4 layer `weight` as `Nf4Linear` holds it: codes
    `[Out, In / 2]`, low nibble first, and absmax `[Out, In / Block]`.

    bitsandbytes quantizes the weight flattened, in blocks of `blocksize`,
    and packs the first code of each pair in the high nibble. With nested
    (double) quantization each block's absmax is itself an 8-bit code:
    `nested_quant_map[code] * nested_absmax[block / nested_blocksize] +
    nested_offset`, in `f32` as bitsandbytes computes it."""
    shape = state.get("shape")
    block = state.get("blocksize")
    dims = (
        [d for d in cast("list[object]", shape) if isinstance(d, int)]
        if isinstance(shape, list)
        else []
    )
    if len(dims) != 2 or not isinstance(block, int):
        raise LinnetError(f"{weight}: its quantization state has no 2-d shape and block size")
    out_features, in_features = dims
    if in_features % block != 0 or block % 2 != 0:
        raise LinnetError(f"{weight}: {in_features} inputs do not make blocks of {block}")
    quant_map = _floats(tensors[weight + ".quant_map"])
    if quant_map.shape != (16,) or not np.allclose(quant_map, NF4, rtol=0, atol=1e-6):
        raise LinnetError(f"{weight}: its quantization map is not NF4's")
    absmax = tensors[weight + ".absmax"]
    if weight + ".nested_absmax" in tensors:
        codes = np.frombuffer(absmax.data, dtype=np.uint8)
        nested = _floats(tensors[weight + ".nested_absmax"]).reshape(-1)
        nested_map = _floats(tensors[weight + ".nested_quant_map"]).reshape(-1)
        nested_block = state.get("nested_blocksize")
        offset = state.get("nested_offset")
        if not isinstance(nested_block, int) or not isinstance(offset, int | float):
            raise LinnetError(f"{weight}: its nested quantization state is incomplete")
        scale = nested_map[codes] * nested[np.arange(codes.size) // nested_block]
        scale = (scale + np.float32(offset)).astype(np.float32)
    else:
        scale = _floats(absmax).reshape(-1)
    if scale.size != out_features * in_features // block:
        raise LinnetError(f"{weight}: {scale.size} scales for blocks of {block}")
    packed = np.frombuffer(tensors[weight].data, dtype=np.uint8)
    if packed.size * 2 != out_features * in_features:
        raise LinnetError(
            f"{weight}: {packed.size} bytes for a [{out_features}, {in_features}] weight"
        )
    swapped = ((packed >> 4) | (packed << 4)).astype(np.uint8)
    return (
        swapped.reshape(out_features, in_features // 2),
        scale.reshape(out_features, in_features // block),
    )


def _import_bitsandbytes(
    weights: str | Path,
    output: str | Path,
    config: Mapping[str, object],
    paths_of: Mapping[str, list[str]],
) -> list[str]:
    """`import_quantized` for a bitsandbytes 4-bit NF4 checkpoint."""
    four_bit = config.get("load_in_4bit", config.get("_load_in_4bit", False))
    kind = str(config.get("bnb_4bit_quant_type", "fp4")).lower()
    if not four_bit:
        raise LinnetError("only 4-bit bitsandbytes checkpoints import (`load_in_4bit`)")
    if kind != "nf4":
        raise LinnetError(f"bitsandbytes' {kind} type does not import; NF4 does")
    tensors: dict[str, RawTensor] = {tensor.name: tensor for tensor in iter_safetensors(weights)}
    states = sorted(name for name in tensors if _BNB_STATE in name)
    out: list[tuple[str, str, tuple[int, ...], bytes]] = []
    taken: set[str] = set()
    quantized: list[str] = []
    for name in states:
        weight, kind = name.split(_BNB_STATE)
        if kind != "nf4":
            raise LinnetError(f"{weight}: bitsandbytes' {kind} type does not import; NF4 does")
        codes, absmax = _bnb_nf4(weight, _bnb_state(tensors[name]), tensors)
        taken |= {name, weight, *(weight + part for part in _BNB_PARTS)}
        for path in paths_of.get(weight, [weight]):
            prefix = _prefix(path)
            out.append((prefix + "weight", "U8", tuple(codes.shape), codes.tobytes()))
            out.append((prefix + "absmax", "F32", tuple(absmax.shape), absmax.tobytes()))
            quantized.append(path)
    if not quantized:
        raise LinnetError(f"no bitsandbytes 4-bit layer (`{_BNB_STATE}nf4`) in {weights}")
    _write_with_the_rest(output, out, tensors, taken, paths_of)
    return quantized


__all__ = [
    "FEW_ROWS",
    "NF4",
    "TILED_ROWS",
    "dequantize_int4_groups",
    "dequantize_int8_groups",
    "import_quantized",
    "pack_int4_groups",
    "quantize_checkpoint",
    "quantize_int4_groups",
    "quantize_int8_groups",
]
