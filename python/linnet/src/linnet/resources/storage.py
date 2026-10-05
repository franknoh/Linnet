"""How many bytes a tensor of a dtype occupies.

Every dtype Linnet has is one table entry here; a new one (an 8-bit float,
a sub-byte integer) is added in this one place. Quantized weights in Linnet
are ordinary tensors: an int4 layer is a `u8` tensor of packed pairs beside
its `f32` or `bf16` scales and zero points, each declared as a `param`. Their
storage is therefore exact from the manifest, metadata included, rather than
a logical bit width times an element count.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import expr as ex


@dataclass(frozen=True, slots=True)
class DTypeStorage:
    """Bits per element as stored, and the bytes an element takes."""

    name: str
    bits: int

    @property
    def element_bytes(self) -> int:
        return (self.bits + 7) // 8


DTYPES: dict[str, DTypeStorage] = {
    d.name: d
    for d in (
        # `bool` is stored a byte per element by every backend Linnet targets.
        DTypeStorage("bool", 8),
        DTypeStorage("i8", 8),
        DTypeStorage("i16", 16),
        DTypeStorage("i32", 32),
        DTypeStorage("i64", 64),
        DTypeStorage("u8", 8),
        DTypeStorage("u16", 16),
        DTypeStorage("u32", 32),
        DTypeStorage("u64", 64),
        DTypeStorage("f16", 16),
        DTypeStorage("bf16", 16),
        DTypeStorage("f32", 32),
        DTypeStorage("f64", 64),
    )
}

FLOAT_DTYPES = frozenset({"f16", "bf16", "f32", "f64"})


def storage(dtype: str) -> DTypeStorage:
    if dtype not in DTYPES:
        raise KeyError(f"no storage is known for dtype `{dtype}`")
    return DTYPES[dtype]


def tensor_bytes(dtype: str, elements: ex.Expr) -> ex.Expr:
    """Bytes of `elements` elements of `dtype`, whole bytes per element; a
    sub-byte dtype would pack, rounding the tensor up to a byte."""
    info = storage(dtype)
    if info.bits % 8 == 0:
        return ex.mul(ex.const(info.element_bytes), elements)
    return ex.ceildiv(ex.mul(ex.const(info.bits), elements), ex.const(8))


def is_float(dtype: str) -> bool:
    return dtype in FLOAT_DTYPES
