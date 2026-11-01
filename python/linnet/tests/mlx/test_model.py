"""`linnet.mlx`: entries as generated `mlx.core` code over one copy of the
weights, against PyTorch's; and the formats MLX has no dtype for, against
`linnet.torch.packed` and PyTorch's FP8."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from linnet.torch import load
from linnet.torch.packed import pack4, unpack4
from tests.helpers import STDLIB

pytest.importorskip("mlx.core")

import mlx.core as mx

from linnet.mlx import load_model
from linnet.mlx.formats import from_e5m2, to_e5m2
from linnet.mlx.formats import pack4 as mlx_pack4
from linnet.mlx.formats import unpack4 as mlx_unpack4

SOURCE = """\
module tests.mlx_model

use std.nn.linear::{Linear}

pub block Model<N: Dim, H: Dim> {
    sub proj: Linear<N, H, f32>
    param scale: Tensor[H; f32]
    state total: Tensor[H; f32]

    // A linear layer whose bias the checkpoint lacks, and a state.
    pub entry forward(x: Tensor[2, N; f32]) -> Tensor[2, H; f32] {
        let y = proj.forward(x)
        let out[b, h] = y[b, h] * scale[h]
        let added[h] = sum[b] out[b, h]
        total = total + added
        return out
    }

    // A runtime loop, read back each iteration.
    pub entry grow(x: Tensor[H; f32], limit: i32) -> Tensor[H; f32] {
        var value = x
        var count: i32 = 0
        while count < limit && all[h] (value[h] < 100.0) {
            value = value * scale
            count = count + 1
        }
        return value
    }

    // A scan, and integer arithmetic rounding toward zero.
    pub entry scan(
        xs: Tensor[4, H; f32],
        n: Tensor[4; i32],
    ) -> (Tensor[4, H; f32], Tensor[4; i32]) {
        var h = xs[0]
        let hs = for t in 0..4 {
            h = tanh(h * 0.5 + xs[t])
            yield h
        }
        let q[i] = n[i] / 3 + n[i] % 3
        return (hs, q)
    }
}
"""

N, H = 5, 4
GENERICS = {"N": N, "H": H}


@pytest.fixture
def model(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "model.linnet"
    source.write_text(SOURCE)
    weights = tmp_path / "model.safetensors"
    draws = torch.Generator().manual_seed(0)
    save_file(
        {
            "proj.weight": torch.randn(H, N, generator=draws),
            "scale": torch.tensor([1.5, 2.0, 0.5, 3.0]),
        },
        str(weights),
    )
    return source, weights


def _host(value: object) -> list[np.ndarray]:
    """An entry's results as NumPy arrays."""
    values = value if isinstance(value, tuple) else (value,)
    return [
        np.asarray(torch.as_tensor(v)) if isinstance(v, torch.Tensor) else np.asarray(v)
        for v in values
    ]


@pytest.mark.parametrize("compile", [True, False])
def test_entries_compute_as_pytorch_does(model: tuple[Path, Path], compile: bool) -> None:
    source, weights = model
    reference = load(source, generics=GENERICS, std_root=STDLIB, weights=weights)
    loaded = load_model(
        source, generics=GENERICS, weights=weights, std_root=STDLIB, compile=compile
    )
    x = np.linspace(-1, 1, 2 * N, dtype=np.float32).reshape(2, N)
    for _ in range(2):  # the state carries from call to call
        expected = _host(reference.run_entry("forward", [torch.from_numpy(x)]))
        got = _host(loaded.run_entry("forward", [x]))
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(loaded.state["total"]),
        dict(reference.named_buffers())["root.total"].numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    start = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    for limit in (0, 2, 50):
        bound = torch.tensor(limit, dtype=torch.int32)
        expected = _host(reference.run_entry("grow", [torch.from_numpy(start), bound]))
        got = _host(loaded.run_entry("grow", [start, np.int32(limit)]))
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-6)
    xs = np.linspace(-2, 2, 4 * H, dtype=np.float32).reshape(4, H)
    n = np.array([-7, -1, 4, 8], dtype=np.int32)
    expected = _host(reference.run_entry("scan", [torch.from_numpy(xs), torch.from_numpy(n)]))
    got = _host(loaded.run_entry("scan", [xs, n]))
    np.testing.assert_allclose(got[0], expected[0], rtol=1e-5, atol=1e-6)
    assert got[1].tolist() == expected[1].tolist() == [-3, -1, 2, 4]


def test_a_weight_is_read_once(model: tuple[Path, Path]) -> None:
    """Every entry shares the weights it reads; one compiled per shape."""
    source, weights = model
    loaded = load_model(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    loaded.run_entry("forward", [np.zeros((2, N), np.float32)])
    loaded.run_entry("grow", [np.ones(H, np.float32), np.int32(1)])
    assert sorted(tensor for tensor, _ in loaded._weights) == ["proj.weight", "scale"]  # pyright: ignore[reportPrivateUsage]
    first = loaded._weights[("scale", "f32")]  # pyright: ignore[reportPrivateUsage]
    loaded.run_entry("grow", [np.ones(H, np.float32), np.int32(3)])
    assert loaded._weights[("scale", "f32")] is first  # pyright: ignore[reportPrivateUsage]


VALUES = np.array(
    [0.24, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.5, 7.0, -100.0, -0.0, 0.1, -2.9, 1e9],
    dtype=np.float32,
)


@pytest.mark.parametrize("kind", ["i4", "u4", "f4e2m1"])
def test_four_bits_as_pytorch_holds_them(kind: str) -> None:
    codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
    expected = unpack4(torch.from_numpy(codes), kind).numpy()
    got = np.asarray(mlx_unpack4(mx.array(codes), kind))
    assert got.dtype == expected.dtype and np.array_equal(got, expected)
    values = np.concatenate([VALUES, -VALUES, VALUES * 0.5, [np.nan, np.inf]]).astype(np.float32)
    values = values[: values.size // 2 * 2]
    packed = pack4(torch.from_numpy(values), kind).numpy()
    assert np.array_equal(np.asarray(mlx_pack4(mx.array(values), kind)), packed)


def test_e5m2_as_pytorch_rounds() -> None:
    every = np.arange(256, dtype=np.uint8)
    wide = torch.from_numpy(every).view(torch.float8_e5m2).float().numpy()
    got = np.asarray(from_e5m2(mx.array(every)).astype(mx.float32))
    assert np.array_equal(got, wide, equal_nan=True)
    values = np.concatenate(
        [VALUES, -VALUES, np.linspace(-60000, 60000, 1001), [65536.0, -1e6, np.inf]]
    ).astype(np.float16)
    expected = torch.from_numpy(values).to(torch.float8_e5m2).view(torch.uint8).numpy()
    assert np.array_equal(np.asarray(to_e5m2(mx.array(values))), expected)


RANDOM = """\
module tests.mlx_random

use std.random::{bits, fold_in, normal, split, uniform}

pub block Rng<N: Dim> {
    param unused: Tensor[1; f32]

    pub entry draws(key: Tensor[2; i64]) -> (Tensor[N, 2; i64], Tensor[N; i64], Tensor[N; f32]) {
        return (split<N>(key), bits<N>(key), uniform<N, f32>(key))
    }

    pub entry more(key: Tensor[2; i64], data: i32) -> (Tensor[2; i64], Tensor[N; f32]) {
        return (fold_in(key, data), normal<N, f32>(key))
    }
}
"""


def test_random_draws_as_pytorch_does(tmp_path: Path) -> None:
    """`std.random`'s Threefry, unsigned shifts and all: the same keys, bits
    and uniforms bit for bit (PyTorch's match JAX's), normals to rounding."""
    source = tmp_path / "rng.linnet"
    source.write_text(RANDOM)
    generics = {"N": 64}
    reference = load(source, generics=generics, std_root=STDLIB)
    loaded = load_model(
        source, generics=generics, weights={"unused": np.zeros(1, np.float32)}, std_root=STDLIB
    )
    for seed in (0, 42, 123456):
        key = np.array([0, seed], dtype=np.int64)
        expected = _host(reference.run_entry("draws", [torch.from_numpy(key)]))
        got = _host(loaded.run_entry("draws", [key]))
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_array_equal(g, e)
        data = torch.tensor(7, dtype=torch.int32)
        expected = _host(reference.run_entry("more", [torch.from_numpy(key), data]))
        got = _host(loaded.run_entry("more", [key, np.int32(7)]))
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_allclose(got[1], expected[1], rtol=1e-6, atol=1e-6)
