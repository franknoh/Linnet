"""Routed experts in JAX: each (row, choice) pair's input times its expert,
the experts held as one `[E, Out, In]` array (MXFP4 experts dequantized once,
by `mxfp4_weight`, which generated code keeps as a prepared value).

A few pairs -- a decoding step -- gather their experts' weights into the
product, which XLA fuses into one reduction over the bytes they need. Many
pairs -- a prompt, a serving step -- are sorted by expert and multiplied by
`grouped`, a Pallas kernel on the GPU that reads each expert's weight for the
tiles of rows that chose it. XLA's own `ragged_dot` multiplies every row by
every expert and masks the result, as the canonical bodies do, so it is what
runs where the kernel cannot (the CPU, widths it does not tile).
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false, reportMissingImports=false

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np
    from jax.experimental.hijax import TransformedRef
    from jax.experimental.pallas import Slice
    from jax.typing import DTypeLike

# At most this many pairs gather their experts' weights.
GATHERED = 64


class _Indexer(Protocol):
    """A kernel operand's `at`: a view of part of it, for `load` and
    `store`."""

    def __getitem__(self, index: tuple[slice | Slice, slice | Slice], /) -> TransformedRef: ...


class _KernelRef(Protocol):
    """One of the kernel's operands, as Pallas hands it to the kernel."""

    @property
    def at(self) -> _Indexer: ...

    @property
    def dtype(self) -> np.dtype[np.generic]: ...

    def __getitem__(
        self, index: jax.Array | tuple[jax.Array | slice | Slice, ...], /
    ) -> jax.Array: ...


def mxfp4_weight(blocks: jax.Array, scales: jax.Array, dtype: DTypeLike) -> jax.Array:
    """MXFP4 experts (`blocks` [E, Out, G, 16] u8, `scales` [E, Out, G] u8)
    as `std.quant::dequantize_mxfp4` reads them: [E, Out, G * 32] in
    `dtype`, every value exact."""
    doubled = jnp.asarray([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=dtype)
    nibbles = jnp.stack([blocks & 15, blocks >> 4], axis=-1).astype(jnp.int32)
    exponent = jnp.clip(scales.astype(jnp.int32) - 127, -62, 62).astype(jnp.float32)
    factor = (jnp.exp2(exponent) * 0.5).astype(dtype)
    values = doubled[nibbles] * factor[..., None, None]
    count, out, groups = scales.shape
    return values.reshape(count, out, groups * 32)


def experts(
    x: jax.Array, weight: jax.Array, chosen: jax.Array, shared: bool, dtype: DTypeLike | None = None
) -> jax.Array:
    """`y[r, k] = input @ weight[chosen[r, k]].T`, [R, K, Out]: the input is
    row `r`'s when `shared` (`x` [R, In]) and the pair's own otherwise (`x`
    [R, K, In]). Accumulates in f32; the result is `dtype` (`x`'s unless
    given)."""
    rows, picks = chosen.shape
    flat = chosen.reshape(-1).astype(jnp.int32)
    result = x.dtype if dtype is None else dtype
    if rows * picks <= GATHERED:
        # A product and a sum in f32, as the canonical bodies write it: XLA
        # reads each chosen weight once inside the reduction. As a dot it
        # would copy them first.
        taken = weight[flat].reshape(rows, picks, *weight.shape[1:]).astype(jnp.float32)
        inputs = (x[:, None, None, :] if shared else x[:, :, None, :]).astype(jnp.float32)
        return jnp.sum(inputs * taken, axis=-1).astype(result)
    order = jnp.argsort(flat, stable=True).astype(jnp.int32)
    counts = jnp.bincount(flat, length=weight.shape[0]).astype(jnp.int32)
    inputs = x[order // picks] if shared else x.reshape(rows * picks, -1)[order]
    product = grouped(inputs, weight, counts, result)
    # Back to the pairs' own order: `order[i]` is the pair at sorted place `i`.
    places = jnp.zeros_like(order).at[order].set(jnp.arange(order.shape[0], dtype=jnp.int32))
    return product[places].reshape(rows, picks, weight.shape[1])


def experts_combined(
    x: jax.Array, weight: jax.Array, chosen: jax.Array, weights: jax.Array
) -> jax.Array:
    """The pairs' products (`x` [R, K, In]) weighed by `weights` [R, K] and
    summed per row, [R, Out]."""
    y = experts(x, weight, chosen, False, jnp.float32)
    return jnp.einsum("rk,rko->ro", weights.astype(jnp.float32), y).astype(x.dtype)


def grouped(x: jax.Array, weight: jax.Array, counts: jax.Array, dtype: DTypeLike) -> jax.Array:
    """`y[i] = x[i] @ weight[e].T` for rows sorted by expert, `counts[e]` of
    them for expert `e` (`x` [P, In], `weight` [E, Out, In]), [P, Out] in
    `dtype`."""
    tiles = _tiling(weight.shape[1], weight.shape[2], x.dtype)
    if jax.default_backend() != "gpu" or tiles is None or x.dtype != weight.dtype:
        product = jax.lax.ragged_dot(
            x, jnp.swapaxes(weight, 1, 2), counts, preferred_element_type=jnp.float32
        )
        return product.astype(dtype)
    return _grouped_pallas(x, weight, counts, dtype, *tiles)


def _tiling(out: int, width: int, dtype: DTypeLike) -> tuple[int, int, int, int, int] | None:
    """The kernel's tile (rows, outputs, inputs) and its warps and pipeline
    stages for these widths, or `None` where it does not tile them."""
    if jnp.dtype(dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        return None
    block_n = next((b for b in (128, 64, 32, 16) if out % b == 0), None)
    block_k = next((b for b in (64, 32, 16) if width % b == 0), None)
    if block_n is None or block_k is None:
        return None
    return 64, block_n, block_k, 4, 3


def _tiles_of(
    counts: jax.Array, pairs: int, block_m: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each row tile's expert and its rows' span in the sorted order: an
    expert's rows split into tiles of `block_m`, the last one partial. There
    are at most `ceil(pairs / block_m) + E` tiles; the spare ones are empty."""
    count = counts.shape[0]
    most = -(-pairs // block_m) + count
    starts = jnp.cumsum(counts) - counts
    per = (counts + block_m - 1) // block_m
    ends = jnp.cumsum(per)
    tile = jnp.arange(most, dtype=jnp.int32)
    owner = jnp.searchsorted(ends, tile, side="right").astype(jnp.int32)
    live = owner < count
    owner = jnp.minimum(owner, count - 1)
    first = starts[owner] + (tile - (ends[owner] - per[owner])) * block_m
    last = jnp.minimum(first + block_m, starts[owner] + counts[owner])
    return owner, jnp.where(live, first, 0), jnp.where(live, last, 0)


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def _grouped_pallas(
    x: jax.Array,
    weight: jax.Array,
    counts: jax.Array,
    dtype: DTypeLike,
    block_m: int,
    block_n: int,
    block_k: int,
    warps: int,
    stages: int,
) -> jax.Array:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    pairs, width = x.shape
    _, out, _ = weight.shape
    owner, first, last = _tiles_of(counts, pairs, block_m)

    def kernel(
        owner_ref: _KernelRef,
        first_ref: _KernelRef,
        last_ref: _KernelRef,
        x_ref: _KernelRef,
        w_ref: _KernelRef,
        y_ref: _KernelRef,
    ) -> None:
        tile = pl.program_id(0)
        columns = pl.ds(pl.program_id(1) * block_n, block_n)
        expert = owner_ref[tile]
        start = first_ref[tile]
        stop = last_ref[tile]

        @pl.when(start < stop)
        def _() -> None:
            rows = pl.ds(start, block_m)
            live = (start + jnp.arange(block_m)) < stop

            def step(k: jax.Array, acc: jax.Array) -> jax.Array:
                inputs = pl.ds(k * block_k, block_k)
                a = plgpu.load(x_ref.at[rows, inputs], mask=live[:, None], other=0.0)
                b = w_ref[expert, columns, inputs]
                return acc + jax.lax.dot_general(
                    a, b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
                )

            acc = jax.lax.fori_loop(
                0, width // block_k, step, jnp.zeros((block_m, block_n), jnp.float32)
            )
            plgpu.store(y_ref.at[rows, columns], acc.astype(y_ref.dtype), mask=live[:, None])

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((pairs, out), dtype),
        grid=(owner.shape[0], out // block_n),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
    )(owner, first, last, x, weight)


__all__ = ["GATHERED", "experts", "experts_combined", "grouped", "mxfp4_weight"]
