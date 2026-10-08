"""Fully sharded parameters in generated JAX.

`linnet jax --fully-shard <block>` has a block gather each weight where it
first uses it: `gather(part, shape, dtype)`. Under `shard_map` over a mesh
axis (`gathering(axis)`, set while the step is traced), the part is this
device's and is all-gathered whole in `dtype`, the dtype the entry computes
in; its gradient is reduce-scattered back in f32. Anywhere else the weight
is whole already and is only cast. With `--remat` on the same blocks, the
backward pass gathers each block's weights again rather than keeping them.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Generator, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from .. import lora
from ..parallel import units

_axis: contextvars.ContextVar[str | None] = contextvars.ContextVar("linnet_fsdp_axis", default=None)


@contextlib.contextmanager
def gathering(axis: str) -> Generator[None, None, None]:
    """While tracing inside `shard_map`: `gather` all-gathers over `axis`."""
    token = _axis.set(axis)
    try:
        yield
    finally:
        _axis.reset(token)


def gather(part: Any, shape: Sequence[int], dtype: Any) -> Any:
    """`part` whole (`shape`) in `dtype`: gathered over the mesh axis when it
    is one device's part, only cast when it is whole already."""
    axis = _axis.get()
    if axis is None or tuple(part.shape) == tuple(shape):
        return part.astype(dtype)
    dim = next(
        d for d, (have, want) in enumerate(zip(part.shape, shape, strict=True)) if have != want
    )
    return gather_as(axis, dim, dtype)(part)


def gather_as(axis: str, dim: int, dtype: Any) -> Callable[[Any], Any]:
    """Inside `shard_map`: a part gathered whole along `dim` in `dtype`, and
    its gradient reduce-scattered back in f32 (in the part's own dtype when
    that is wider), then given the part's dtype."""

    @jax.custom_vjp
    def gather_part(part: Any) -> Any:
        return jax.lax.all_gather(part.astype(dtype), axis, axis=dim, tiled=True)

    def forward(part: Any) -> Any:
        return gather_part(part), jnp.zeros((0,), part.dtype)

    def backward(kept: Any, grad: Any) -> Any:
        wide = jnp.promote_types(kept.dtype, jnp.float32)
        summed = jax.lax.psum_scatter(grad.astype(wide), axis, scatter_dimension=dim, tiled=True)
        return (summed.astype(kept.dtype),)

    gather_part.defvjp(forward, backward)
    return gather_part


def gathered_by_code(path: str, sharded: Sequence[str]) -> bool:
    """Whether generated code gathers the parameter at `path` itself: it
    belongs to a sharded block and is not an adapter (adapters are added by
    the code generator, outside the blocks' gathering)."""
    return not lora.is_adapter(path) and any(path.startswith(unit + ".") for unit in sharded)


__all__ = ["gather", "gather_as", "gathered_by_code", "gathering", "units"]
