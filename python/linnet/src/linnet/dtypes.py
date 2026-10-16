"""Linnet's scalar types, and what each file format and framework calls
them: one record per dtype, which every table in the package is built from.

`bool` takes a byte per element everywhere Linnet runs or stores tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .compiler import LinnetError

Kind = Literal["bool", "int", "uint", "float"]


@dataclass(frozen=True, slots=True)
class DType:
    """A scalar type: its Linnet name, width, kind, and its name in
    SafeTensors, ONNX (`TensorProto` element type), MLIR, NumPy (`bfloat16`
    and the FP8 and 4-bit formats from `ml_dtypes`) and PyTorch
    (`torch.<name>`). A storage format (`std.dtypes`) is cast to and from but
    computes nothing; a 4-bit one is packed, two elements a byte along a
    tensor's last axis, which PyTorch holds as `uint8` bytes and SafeTensors
    as `U8`. One held as `byte` (E8M0) is its byte in every framework, whose
    names here are `u8`'s: its casts are integer operations on it."""

    name: str
    bits: int
    kind: Kind
    safetensors: str
    onnx: int
    mlir: str
    numpy: str
    torch: str
    storage: bool = False
    byte: bool = False

    @property
    def element_bytes(self) -> int:
        """Whole bytes an element takes (a packed format's, rounded up:
        `bytes_for` counts them exactly)."""
        return (self.bits + 7) // 8

    def bytes_for(self, elements: int) -> int:
        """Bytes `elements` elements take: two a byte when packed."""
        return -(-elements * self.bits // 8)

    @property
    def packed(self) -> bool:
        """Whether two elements share a byte along a tensor's last axis."""
        return self.bits < 8

    def storage_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        """The shape of a tensor of `shape` as bytes: its last axis halved
        when packed."""
        if not self.packed or not shape:
            return tuple(shape)
        return (*shape[:-1], shape[-1] // 2)

    @property
    def is_float(self) -> bool:
        """Whether it is a float that computes."""
        return self.kind == "float" and not self.storage


DTYPES: dict[str, DType] = {
    d.name: d
    for d in (
        DType("bool", 8, "bool", "BOOL", 9, "i1", "bool", "bool"),
        DType("i8", 8, "int", "I8", 3, "i8", "int8", "int8"),
        DType("i16", 16, "int", "I16", 5, "i16", "int16", "int16"),
        DType("i32", 32, "int", "I32", 6, "i32", "int32", "int32"),
        DType("i64", 64, "int", "I64", 7, "i64", "int64", "int64"),
        DType("u8", 8, "uint", "U8", 2, "ui8", "uint8", "uint8"),
        DType("u16", 16, "uint", "U16", 4, "ui16", "uint16", "uint16"),
        DType("u32", 32, "uint", "U32", 12, "ui32", "uint32", "uint32"),
        DType("u64", 64, "uint", "U64", 13, "ui64", "uint64", "uint64"),
        DType("f16", 16, "float", "F16", 10, "f16", "float16", "float16"),
        DType("bf16", 16, "float", "BF16", 16, "bf16", "bfloat16", "bfloat16"),
        DType("f32", 32, "float", "F32", 1, "f32", "float32", "float32"),
        DType("f64", 64, "float", "F64", 11, "f64", "float64", "float64"),
        DType(
            "f8e4m3", 8, "float", "F8_E4M3", 17, "f8E4M3FN", "float8_e4m3fn", "float8_e4m3fn", True
        ),
        DType("f8e5m2", 8, "float", "F8_E5M2", 19, "f8E5M2", "float8_e5m2", "float8_e5m2", True),
        DType("i4", 4, "int", "", 22, "i4", "int4", "uint8", True),
        DType("u4", 4, "uint", "", 21, "ui4", "uint4", "uint8", True),
        DType("f4e2m1", 4, "float", "", 23, "f4E2M1FN", "float4_e2m1fn", "uint8", True),
        DType("f8e8m0", 8, "float", "F8_E8M0", 2, "ui8", "uint8", "uint8", True, True),
    )
}

FLOATS = frozenset(d.name for d in DTYPES.values() if d.is_float)
INTEGERS = frozenset(d.name for d in DTYPES.values() if d.kind in ("int", "uint"))
STORAGE = frozenset(d.name for d in DTYPES.values() if d.storage)
# The dtypes each class of dtype generic admits (`T: Float`).
CLASSES: dict[str, frozenset[str]] = {
    "float": FLOATS,
    "integer": INTEGERS,
    "numeric": FLOATS | INTEGERS,
    "any": FLOATS | INTEGERS | STORAGE | {"bool"},
}

# The packed formats have no SafeTensors name: a checkpoint stores their bytes.
BY_SAFETENSORS: dict[str, DType] = {d.safetensors: d for d in DTYPES.values() if d.safetensors}
PACKED = frozenset(d.name for d in DTYPES.values() if d.packed)
# Checkpoint dtypes whose bytes a `u8` parameter may also hold, as bits
# (FP8 that `std.quant::decode_fp8` decodes); a `U8` tensor binds to an FP8
# or E8M0 parameter the same way (`linnet.weights.as_bits`).
AS_BITS = frozenset(("F8_E4M3", "F8_E5M2", "F8_E8M0"))
# A dtype held as its byte has `u8`'s names, which name `u8`.
BY_ONNX: dict[int, DType] = {d.onnx: d for d in DTYPES.values() if not d.byte}
BY_MLIR: dict[str, DType] = {d.mlir: d for d in DTYPES.values() if not d.byte}
BY_NUMPY: dict[str, DType] = {d.numpy: d for d in DTYPES.values() if not d.byte}


def from_safetensors(code: str) -> str | None:
    """The Linnet name of SafeTensors dtype `code` (`BF16` -> `bf16`)."""
    found = BY_SAFETENSORS.get(code)
    return None if found is None else found.name


def dtype(name: str) -> DType:
    """The dtype Linnet calls `name`."""
    found = DTYPES.get(name)
    if found is None:
        raise LinnetError(f"`{name}` is not a Linnet dtype ({', '.join(DTYPES)})")
    return found


__all__ = [
    "AS_BITS",
    "BY_MLIR",
    "BY_NUMPY",
    "BY_ONNX",
    "BY_SAFETENSORS",
    "CLASSES",
    "DTYPES",
    "FLOATS",
    "INTEGERS",
    "PACKED",
    "STORAGE",
    "DType",
    "Kind",
    "dtype",
    "from_safetensors",
]
