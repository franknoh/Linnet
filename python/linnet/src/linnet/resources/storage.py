"""How many bytes a tensor of a dtype occupies.

Every dtype's width is in `linnet.dtypes`. Quantized weights in Linnet are
ordinary tensors: an int4 layer is a `u8` tensor of packed pairs beside its
`f32` or `bf16` scales and zero points, each declared as a `param`. Their
storage is therefore exact from the manifest, metadata included, rather than
a logical bit width times an element count.
"""

from __future__ import annotations

from .. import dtypes
from . import expr as ex


def tensor_bytes(dtype: str, elements: ex.Expr) -> ex.Expr:
    """Bytes of `elements` elements of `dtype`, whole bytes per element; a
    sub-byte dtype would pack, rounding the tensor up to a byte."""
    info = dtypes.dtype(dtype)
    if info.bits % 8 == 0:
        return ex.mul(ex.const(info.element_bytes), elements)
    return ex.ceildiv(ex.mul(ex.const(info.bits), elements), ex.const(8))
