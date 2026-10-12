"""Codebook formats (`std.dtypes::nf4`): a tensor holds its carrier's codes
(`u4`, two a byte), a cast widens each code to its value and narrows each
value to the nearest code (a tie to the lower). The same on PyTorch,
compiled and interpreted, JAX, ONNX and StableHLO, from a checkpoint's
packed bytes."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from safetensors.torch import save_file

from linnet.compiler import find_compiler
from linnet.torch import load
from tests.helpers import STDLIB

SOURCE = """\
module tests.codebooks

use std.dtypes::{nf4}

pub block Model<R: Dim, N: Dim> {
    param q: Tensor[R, N; nf4]
    param absmax: Tensor[R; f32]

    pub entry dequantize(x: Tensor[N; f32]) -> Tensor[R, N; f32] {
        let w = cast<f32>(q)
        let out[r, n] = w[r, n] * absmax[r] + x[n]
        return out
    }

    // Each value to the nearest NF4 value.
    pub entry round(x: Tensor[2, N; f32]) -> Tensor[2, N; f32] {
        return cast<f32>(cast<nf4>(x))
    }
}

pub entry round_values<N: Dim>(x: Tensor[N; f32]) -> Tensor[N; f32] {
    return cast<f32>(cast<nf4>(x))
}
"""

# NF4's values as bitsandbytes has them, by code.
TABLE = np.array(
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
    ]
)
NF4 = TABLE.astype(np.float32)
# The midpoints between neighbours, as the source's values give them.
MIDPOINTS = ((TABLE[1:] + TABLE[:-1]) / 2).astype(np.float32)
R, N = 3, 8
# Ties at midpoints, beyond the range, NaN, and values in between.
VALUES = np.array(
    [
        MIDPOINTS[0],
        MIDPOINTS[7],
        2.0,
        -2.0,
        float("nan"),
        0.03,
        -0.3,
        0.9,
        0.5,
        -0.6,
        0.1,
        0.0,
        -1.0,
        1.0,
        0.39,
        -0.12,
    ],
    dtype=np.float32,
)


def _nearest(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Each of `x` as NF4 rounds it: the code is how many midpoints lie
    below it."""
    codes = (x[..., None] > MIDPOINTS).sum(-1)
    return NF4[codes]


def _bytes(codes: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)


@pytest.fixture
def model(tmp_path: Path) -> tuple[Path, Path, NDArray[np.float32]]:
    source = tmp_path / "codebooks.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generator = np.random.default_rng(0)
    codes = generator.integers(0, 16, (R, N), dtype=np.uint8)
    absmax = generator.uniform(0.5, 2.0, R).astype(np.float32)
    weights = tmp_path / "codebooks.safetensors"
    save_file(
        {"q": torch.from_numpy(_bytes(codes)), "absmax": torch.from_numpy(absmax)}, str(weights)
    )
    return source, weights, NF4[codes] * absmax[:, None]


def _input() -> NDArray[np.float32]:
    return np.linspace(-1.0, 1.0, N, dtype=np.float32)


@pytest.mark.parametrize("compile", [False, True])
def test_torch_widens_and_narrows(
    model: tuple[Path, Path, NDArray[np.float32]], compile: bool
) -> None:
    source, weights, wide = model
    loaded = load(
        source, generics={"R": R, "N": N}, std_root=STDLIB, weights=weights, compile=compile
    )
    got = loaded.run_entry("dequantize", [torch.from_numpy(_input())])
    assert isinstance(got, torch.Tensor)
    np.testing.assert_allclose(got.numpy(), wide + _input(), rtol=1e-6, atol=1e-6)
    rows = VALUES.reshape(2, N)
    rounded = loaded.run_entry("round", [torch.from_numpy(rows)])
    assert isinstance(rounded, torch.Tensor)
    assert np.array_equal(rounded.numpy(), _nearest(rows))


def test_jax_widens(model: tuple[Path, Path, NDArray[np.float32]]) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load as load_jax
    from linnet.jax import load_source

    source, weights, wide = model
    for make in (load_jax, load_source):
        function = make(
            source, generics={"R": R, "N": N}, weights=weights, std_root=STDLIB, entry="dequantize"
        )
        got = np.asarray(function(_input()))
        np.testing.assert_allclose(got, wide + _input(), rtol=1e-6, atol=1e-6)


def test_onnx_widens_and_narrows(model: tuple[Path, Path, NDArray[np.float32]]) -> None:
    pytest.importorskip("onnxruntime")
    from linnet.onnx import load_model

    source, weights, wide = model
    loaded = load_model(source, generics={"R": R, "N": N}, weights=weights, std_root=STDLIB)
    got = np.asarray(loaded.run_entry("dequantize", [_input()]))
    np.testing.assert_allclose(got, wide + _input(), rtol=1e-6, atol=1e-6)
    rows = VALUES.reshape(2, N)
    assert np.array_equal(np.asarray(loaded.run_entry("round", [rows])), _nearest(rows))


def test_stablehlo_narrows(model: tuple[Path, Path, NDArray[np.float32]]) -> None:
    jax = pytest.importorskip("jax")
    import jax.extend as jex

    source, _, _ = model
    command = [find_compiler(), "stablehlo", "--std", str(STDLIB), "--entry", "round_values"]
    completed = subprocess.run(
        [*command, "--bind", f"N={VALUES.size}", str(source)],
        capture_output=True,
        text=True,
        check=True,
    )
    options = importlib.import_module("jaxlib._jax")
    backend = jex.backend.get_backend()
    device = backend.local_devices()[0]
    executable = backend.compile_and_load(
        completed.stdout, options.DeviceList((device,)), options.CompileOptions()
    )
    (got,) = executable.execute([jax.device_put(VALUES, device)])
    assert np.array_equal(np.asarray(got), _nearest(VALUES))
