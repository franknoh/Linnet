"""E8M0 (`std.dtypes::f8e8m0`), an MX block's power-of-two scale, held as
its byte on every backend: each byte widens to `2 ** (byte - 127)` (NaN for
255) and values narrow to the power PyTorch's `float8_e8m0fnu` rounds a
normal `f32` to, on PyTorch (compiled and interpreted), JAX, ONNX and
StableHLO alike, save that XLA flushes byte 0's subnormal `2 ** -127` to
zero; a checkpoint's `U8` (or `F8_E8M0`) bytes bind to it."""

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
module tests.e8m0

use std.dtypes::{f8e8m0}

pub block Model<N: Dim> {
    param scale: Tensor[N; f8e8m0]

    pub entry widen() -> Tensor[N; f32] {
        return cast<f32>(scale)
    }

    // Rounded to a power of two and back.
    pub entry round(x: Tensor[N; f32]) -> Tensor[N; f32] {
        return cast<f32>(cast<f8e8m0>(x))
    }

    // The bytes themselves.
    pub entry codes(x: Tensor[N; f32]) -> Tensor[N; f8e8m0] {
        return cast<f8e8m0>(x)
    }
}

pub entry round_values<N: Dim>(x: Tensor[N; f32]) -> Tensor[N; f32] {
    return cast<f32>(cast<f8e8m0>(x))
}
"""

BYTES = np.arange(256, dtype=np.uint8)
# Ties at 1.5 times a power round up; zero and NaN, infinities and values
# from 1.5 * 2 ** 127 up are the edges.
SPECIAL = [
    1.0, 1.5, 1.4999999, 3.0, 0.75, 0.7499999, -3.0, -0.1, 0.0, -0.0,
    1e38, 2.5e38, 2.6e38, 3.4e38, float("inf"), float("-inf"), float("nan"),
    1.1754944e-38, 1.7e-38, 2.0 ** -100, 2.0 ** 100, 6.0, 7.0, 1e-3,
]  # fmt: skip


def _values() -> NDArray[np.float32]:
    """The special values, then normal ones spread over the whole range, 256
    in all (the length `N` the model is loaded with)."""
    generator = np.random.default_rng(0)
    exponents = generator.uniform(-125.0, 127.0, 256 - len(SPECIAL))
    signs = generator.choice([-1.0, 1.0], exponents.size)
    spread = signs * np.exp2(exponents)
    return np.concatenate([np.array(SPECIAL), spread]).astype(np.float32)


def _widened(codes: NDArray[np.uint8]) -> NDArray[np.float32]:
    exact = np.ldexp(np.float64(1.0), codes.astype(np.int64) - 127)
    return np.where(codes == 255, np.nan, exact).astype(np.float32)


def _codes(values: NDArray[np.float32]) -> NDArray[np.uint8]:
    """The byte each value narrows to (`std.dtypes::f8e8m0`)."""
    magnitude = np.abs(values.astype(np.float64))
    mantissa, exponent = np.frexp(magnitude)  # magnitude = mantissa * 2 ** exponent
    up = (2 * mantissa >= 1.5).astype(np.int64)
    code = np.clip(exponent.astype(np.int64) - 1 + 127 + up, 0, 255)
    code = np.where(magnitude == 0, 0, code)
    return np.where(np.isfinite(magnitude), code, 255).astype(np.uint8)


def test_the_reference_rounds_as_pytorch() -> None:
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("PyTorch 2.7 or later has float8_e8m0fnu")
    values = _values()
    normal = np.isfinite(values) & (np.abs(values) >= np.float32(2.0**-126))
    wanted = torch.from_numpy(values[normal]).to(torch.float8_e8m0fnu).view(torch.uint8)
    assert np.array_equal(_codes(values[normal]), wanted.numpy())
    widened = torch.from_numpy(BYTES).view(torch.float8_e8m0fnu).float().numpy()
    np.testing.assert_array_equal(_widened(BYTES), widened)


@pytest.fixture
def model(tmp_path: Path) -> tuple[Path, Path]:
    """The source, and a checkpoint of every byte as `U8`."""
    source = tmp_path / "e8m0.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    weights = tmp_path / "e8m0.safetensors"
    save_file({"scale": torch.from_numpy(BYTES)}, str(weights))
    return source, weights


def _flushed(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """`values` as XLA holds them: byte 0's `2 ** -127`, a subnormal `f32`,
    flushed to zero."""
    return np.where(values == np.float32(2.0**-127), np.float32(0.0), values)


def _check_round(rounded: NDArray[np.float32], flush: bool = False) -> None:
    expected = _widened(_codes(_values()))
    np.testing.assert_array_equal(rounded, _flushed(expected) if flush else expected)


@pytest.mark.parametrize("compile", [False, True])
def test_torch_holds_the_byte(model: tuple[Path, Path], compile: bool) -> None:
    source, weights = model
    loaded = load(source, generics={"N": 256}, std_root=STDLIB, weights=weights, compile=compile)
    assert dict(loaded.named_parameters())["root.scale"].dtype == torch.uint8
    widened = loaded.run_entry("widen", [])
    assert isinstance(widened, torch.Tensor)
    np.testing.assert_array_equal(widened.numpy(), _widened(BYTES))
    rounded = loaded.run_entry("round", [torch.from_numpy(_values())])
    assert isinstance(rounded, torch.Tensor)
    _check_round(rounded.numpy())
    codes = loaded.run_entry("codes", [torch.from_numpy(_values())])
    assert isinstance(codes, torch.Tensor)
    assert codes.dtype == torch.uint8
    assert np.array_equal(codes.numpy(), _codes(_values()))


def test_torch_binds_f8_e8m0_checkpoints(tmp_path: Path) -> None:
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("PyTorch 2.7 or later has float8_e8m0fnu")
    source = tmp_path / "e8m0.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    weights = tmp_path / "e8m0.safetensors"
    save_file({"scale": torch.from_numpy(BYTES).view(torch.float8_e8m0fnu)}, str(weights))
    loaded = load(source, generics={"N": 256}, std_root=STDLIB, weights=weights)
    widened = loaded.run_entry("widen", [])
    assert isinstance(widened, torch.Tensor)
    np.testing.assert_array_equal(widened.numpy(), _widened(BYTES))


def test_jax_holds_the_byte(model: tuple[Path, Path]) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load as load_jax
    from linnet.jax import load_source

    source, weights = model
    for make in (load_jax, load_source):
        widen = make(source, generics={"N": 256}, weights=weights, std_root=STDLIB, entry="widen")
        np.testing.assert_array_equal(np.asarray(widen()), _flushed(_widened(BYTES)))
        round_ = make(source, generics={"N": 256}, weights=weights, std_root=STDLIB, entry="round")
        _check_round(np.asarray(round_(_values())), flush=True)


def test_stablehlo_holds_the_byte(model: tuple[Path, Path]) -> None:
    jax = pytest.importorskip("jax")
    import jax.extend as jex

    source, _ = model
    command = [find_compiler(), "stablehlo", "--std", str(STDLIB), "--entry", "round_values"]
    completed = subprocess.run(
        [*command, "--bind", "N=256", str(source)], capture_output=True, text=True, check=True
    )
    options = importlib.import_module("jaxlib._jax")
    backend = jex.backend.get_backend()
    device = backend.local_devices()[0]
    executable = backend.compile_and_load(
        completed.stdout, options.DeviceList((device,)), options.CompileOptions()
    )
    (got,) = executable.execute([jax.device_put(_values(), device)])
    _check_round(np.asarray(got), flush=True)


def test_onnx_holds_the_byte(model: tuple[Path, Path]) -> None:
    pytest.importorskip("onnxruntime")
    from linnet.onnx import load_model

    source, weights = model
    loaded = load_model(source, generics={"N": 256}, weights=weights, std_root=STDLIB)
    np.testing.assert_array_equal(np.asarray(loaded.run_entry("widen", [])), _widened(BYTES))
    _check_round(np.asarray(loaded.run_entry("round", [_values()])))
