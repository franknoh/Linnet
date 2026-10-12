"""The 4-bit formats (`std.dtypes`: `i4`, `u4`, `f4e2m1`), two elements a
byte along the last axis: PyTorch holds their bytes and casts as
`ml_dtypes` does; a checkpoint's packed `U8` bytes bind to a parameter of
one on PyTorch (compiled and interpreted) and JAX alike; StableHLO casts
them natively; ONNX export refuses them."""

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

from linnet.compiler import LinnetError, find_compiler
from linnet.torch import load
from linnet.torch.packed import pack4, unpack4
from tests.helpers import STDLIB

ml_dtypes = pytest.importorskip("ml_dtypes")

SOURCE = """\
module tests.packed

use std.dtypes::{f4e2m1, i4}

pub block Model<R: Dim, N: Dim> {
    param w: Tensor[R, N; f4e2m1]
    param q: Tensor[R, N; i4]
    param scale: Tensor[R; f32]

    pub entry forward(x: Tensor[N; f32]) -> Tensor[R, N; f32] {
        let wide = cast<f32>(w) + cast<f32>(q)
        let out[r, n] = wide[r, n] * scale[r] + x[n]
        return out
    }

    // Rounded to FP4 and back.
    pub entry round(x: Tensor[2, N; f32]) -> Tensor[2, N; f32] {
        return cast<f32>(cast<f4e2m1>(x))
    }
}

// The same, outside the block.
pub entry round_values<N: Dim>(x: Tensor[N; f32]) -> Tensor[N; f32] {
    return cast<f32>(cast<f4e2m1>(x))
}
"""

R, N = 3, 8
VALUES = np.array(
    [0.24, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.5, 7.0, -100.0, -0.0, 0.1, -2.9, 1e9],
    dtype=np.float32,
)
CODES = np.arange(16, dtype=np.uint8)


def _bytes(codes: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Codes along the last axis packed two a byte, the low first."""
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)


def test_codes_widen_and_values_narrow_as_ml_dtypes_does() -> None:
    fp4 = CODES.view(ml_dtypes.float4_e2m1fn).astype(np.float32)
    packed = torch.from_numpy(_bytes(CODES))
    assert np.array_equal(unpack4(packed, "f4e2m1").numpy(), fp4)
    assert np.array_equal(unpack4(packed, "u4").numpy(), CODES)
    signed = np.where(CODES >= 8, CODES.astype(np.int16) - 16, CODES)
    assert np.array_equal(unpack4(packed, "i4").numpy(), signed)
    expected = VALUES.astype(ml_dtypes.float4_e2m1fn).astype(np.float32)
    rounded = unpack4(pack4(torch.from_numpy(VALUES), "f4e2m1"), "f4e2m1").numpy()
    assert np.array_equal(rounded, expected)
    assert np.array_equal(np.signbit(rounded), np.signbit(expected))
    nan = unpack4(pack4(torch.tensor([float("nan"), 1.0]), "f4e2m1"), "f4e2m1")
    assert nan.tolist() == [0.0, 1.0]
    ints = torch.tensor([7, 8, -9, 15, 16, -1, 2.7, -9.5])
    for kind, numpy_dtype in (("i4", ml_dtypes.int4), ("u4", ml_dtypes.uint4)):
        got = unpack4(pack4(ints, kind), kind).numpy().astype(np.int64)
        want = ints.numpy().astype(np.int64).astype(numpy_dtype).astype(np.int64)
        assert np.array_equal(got, want), kind


@pytest.fixture
def model(tmp_path: Path) -> tuple[Path, Path, NDArray[np.float32]]:
    """The source, a checkpoint of packed bytes, and the weight they stand for."""
    source = tmp_path / "packed.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generator = np.random.default_rng(0)
    fp4 = generator.integers(0, 16, (R, N), dtype=np.uint8)
    int4 = generator.integers(0, 16, (R, N), dtype=np.uint8)
    scale = generator.standard_normal(R).astype(np.float32)
    weights = tmp_path / "packed.safetensors"
    save_file(
        {
            "w": torch.from_numpy(_bytes(fp4)),
            "q": torch.from_numpy(_bytes(int4)),
            "scale": torch.from_numpy(scale),
        },
        str(weights),
    )
    wide = fp4.view(ml_dtypes.float4_e2m1fn).astype(np.float32)
    wide = wide + np.where(int4 >= 8, int4.astype(np.float32) - 16, int4)
    return source, weights, (wide * scale[:, None]).astype(np.float32)


def _input() -> NDArray[np.float32]:
    return np.linspace(-1.0, 1.0, N, dtype=np.float32)


@pytest.mark.parametrize("compile", [False, True])
def test_torch_binds_the_bytes_and_widens(
    model: tuple[Path, Path, NDArray[np.float32]], compile: bool
) -> None:
    source, weights, wide = model
    loaded = load(
        source, generics={"R": R, "N": N}, std_root=STDLIB, weights=weights, compile=compile
    )
    assert dict(loaded.named_parameters())["root.w"].shape == (R, N // 2)
    got = loaded.run_entry("forward", [torch.from_numpy(_input())])
    assert isinstance(got, torch.Tensor)
    np.testing.assert_allclose(got.numpy(), wide + _input(), rtol=1e-6, atol=1e-6)
    rows = VALUES.reshape(2, N)
    rounded = loaded.run_entry("round", [torch.from_numpy(rows)])
    assert isinstance(rounded, torch.Tensor)
    expected = rows.astype(ml_dtypes.float4_e2m1fn).astype(np.float32)
    assert np.array_equal(rounded.numpy(), expected)


def test_jax_unpacks_the_bytes(model: tuple[Path, Path, NDArray[np.float32]]) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load as load_jax
    from linnet.jax import load_source

    source, weights, wide = model
    for make in (load_jax, load_source):
        function = make(
            source, generics={"R": R, "N": N}, weights=weights, std_root=STDLIB, entry="forward"
        )
        got = np.asarray(function(_input()))
        np.testing.assert_allclose(got, wide + _input(), rtol=1e-6, atol=1e-6)


def test_stablehlo_casts_natively(model: tuple[Path, Path, NDArray[np.float32]]) -> None:
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
    expected = VALUES.astype(ml_dtypes.float4_e2m1fn).astype(np.float32)
    assert np.array_equal(np.asarray(got), expected)


def test_onnx_refuses_four_bits(model: tuple[Path, Path, NDArray[np.float32]]) -> None:
    pytest.importorskip("onnx")
    from linnet.onnx import export_model

    source, weights, _ = model
    with pytest.raises(LinnetError, match="4-bit"):
        export_model(
            source, generics={"R": R, "N": N}, weights=weights, std_root=STDLIB, entry="forward"
        )
