"""`linnet.jax.moe`: rows sorted by expert, each multiplied by its expert's
weight, by the Pallas kernel on a GPU and `ragged_dot` elsewhere."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from linnet.jax import moe


def _case(counts: list[int], out: int, width: int, dtype: DTypeLike) -> tuple[jax.Array, ...]:
    rng = np.random.default_rng(0)
    rows = sum(counts)
    x = jnp.asarray(rng.standard_normal((rows, width)), dtype=dtype)
    weight = jnp.asarray(rng.standard_normal((len(counts), out, width)) * 0.1, dtype=dtype)
    return x, weight, jnp.asarray(counts, dtype=jnp.int32)


def _expected(x: jax.Array, weight: jax.Array, counts: jax.Array) -> np.ndarray:
    owners = np.repeat(np.arange(counts.shape[0]), np.asarray(counts))
    xs = np.asarray(x, dtype=np.float32)
    ws = np.asarray(weight, dtype=np.float32)
    return np.einsum("pi,poi->po", xs, ws[owners])


# Empty experts, groups that are not a whole number of tiles, and a last
# tile past the end of the rows.
COUNTS = [0, 1, 63, 64, 65, 0, 130, 7]


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="the kernel runs on a GPU")
@pytest.mark.parametrize(("out", "width"), [(128, 128), (96, 160), (5760, 2880)])
def test_the_kernel_multiplies_each_group_by_its_expert(out: int, width: int) -> None:
    x, weight, counts = _case(COUNTS, out, width, jnp.bfloat16)
    assert moe._tiling(out, width, x.dtype) is not None  # pyright: ignore[reportPrivateUsage]
    got = np.asarray(moe.grouped(x, weight, counts, jnp.float32))
    want = _expected(x, weight, counts)
    assert np.abs(got - want).max() <= 1e-2 * np.abs(want).max()


def test_the_fallback_multiplies_each_group_by_its_expert() -> None:
    x, weight, counts = _case(COUNTS, 24, 40, jnp.float32)
    got = np.asarray(moe.grouped(x, weight, counts, jnp.float32))
    tolerance = 3e-3 if jax.default_backend() == "gpu" else 1e-5
    want = _expected(x, weight, counts)
    assert np.abs(got - want).max() <= tolerance * np.abs(want).max()


def test_mxfp4_weight_reads_every_code() -> None:
    codes = np.arange(256, dtype=np.uint8).reshape(1, 1, 1, 16, 16)[0, 0]  # [1, 16, 16]
    blocks = jnp.asarray(codes.reshape(1, 16, 1, 16))
    scales = jnp.full((1, 16, 1), 128, dtype=jnp.uint8)
    got = np.asarray(moe.mxfp4_weight(blocks, scales, jnp.float32))
    fp4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    pairs = np.stack([fp4[codes & 15], fp4[codes >> 4]], axis=-1).reshape(1, 16, 32)
    np.testing.assert_array_equal(got, pairs * 2.0)
