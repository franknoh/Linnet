"""Linnet's scalar types, and what each file format and framework calls
them: one record per dtype, which every table in the package is built from.

`bool` takes a byte per element everywhere Linnet runs or stores tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .compiler import LinnetError

Kind = Literal["bool", "int", "uint", "float"]


@dataclass(frozen=True, slots=True)
class DType:
    """A scalar type: its Linnet name, width, kind, and its name in
    SafeTensors, ONNX (`TensorProto` element type), MLIR, NumPy (`bfloat16`
    from `ml_dtypes`) and PyTorch (`torch.<name>`)."""

    name: str
    bits: int
    kind: Kind
    safetensors: str
    onnx: int
    mlir: str
    numpy: str
    torch: str

    @property
    def element_bytes(self) -> int:
        return (self.bits + 7) // 8

    @property
    def is_float(self) -> bool:
        return self.kind == "float"


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
    )
}

FLOATS = frozenset(d.name for d in DTYPES.values() if d.kind == "float")
INTEGERS = frozenset(d.name for d in DTYPES.values() if d.kind in ("int", "uint"))
# The dtypes each class of dtype generic admits (`T: Float`).
CLASSES: dict[str, frozenset[str]] = {
    "float": FLOATS,
    "integer": INTEGERS,
    "numeric": FLOATS | INTEGERS,
    "any": FLOATS | INTEGERS | {"bool"},
}

BY_SAFETENSORS: dict[str, DType] = {d.safetensors: d for d in DTYPES.values()}
BY_ONNX: dict[int, DType] = {d.onnx: d for d in DTYPES.values()}
BY_MLIR: dict[str, DType] = {d.mlir: d for d in DTYPES.values()}
BY_NUMPY: dict[str, DType] = {d.numpy: d for d in DTYPES.values()}


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
    "BY_MLIR",
    "BY_NUMPY",
    "BY_ONNX",
    "BY_SAFETENSORS",
    "CLASSES",
    "DTYPES",
    "FLOATS",
    "INTEGERS",
    "DType",
    "Kind",
    "dtype",
    "from_safetensors",
]
