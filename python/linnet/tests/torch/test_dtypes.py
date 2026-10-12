"""Storage dtypes (`std.dtypes`): an FP8 parameter binds an `F8_E4M3`
checkpoint tensor as it is and casts to the value PyTorch gives each byte,
in the interpreter and in the generated source; and a cast to FP8 rounds as
PyTorch rounds."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

from linnet.torch import load
from tests.helpers import STDLIB

SOURCE = """\
module tests.dtypes

use std.dtypes::{f8e4m3, f8e5m2}

pub block Scaled<N: Dim> {
    param weight: Tensor[N; f8e4m3]

    pub entry forward(x: Tensor[N; f32]) -> Tensor[N; f32] {
        return cast<f32>(weight) * x
    }
}

pub entry round<N: Dim>(x: Tensor[N; f32]) -> (Tensor[N; f8e4m3], Tensor[N; f8e5m2]) {
    return (cast<f8e4m3>(x), cast<f8e5m2>(x))
}
"""


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "dtypes.linnet"
    path.write_text(SOURCE, encoding="utf-8")
    return path


@pytest.mark.parametrize("compile", [False, True])
def test_an_fp8_parameter_binds_and_casts(tmp_path: Path, source: Path, compile: bool) -> None:
    weight = (torch.arange(8, dtype=torch.float32) - 3.5).to(torch.float8_e4m3fn)
    save_file({"weight": weight}, str(tmp_path / "model.safetensors"))
    model = load(
        source,
        generics={"N": 8},
        root="Scaled",
        std_root=STDLIB,
        weights=tmp_path / "model.safetensors",
        compile=compile,
    )
    x = torch.linspace(-1.0, 1.0, 8)
    result: object = model.run_entry("forward", [x])
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, weight.float() * x)


@pytest.mark.parametrize("compile", [False, True])
def test_a_cast_to_fp8_rounds_as_pytorch_does(source: Path, compile: bool) -> None:
    from linnet.torch import load_function

    round_ = load_function(source, "round", std_root=STDLIB, compile=compile)
    x = torch.tensor([0.1, -0.3, 1.7, 300.0, -2.5e4])
    e4m3, e5m2 = round_(x)
    assert e4m3.dtype == torch.float8_e4m3fn and e5m2.dtype == torch.float8_e5m2
    torch.testing.assert_close(e4m3.float(), x.to(torch.float8_e4m3fn).float())
    torch.testing.assert_close(e5m2.float(), x.to(torch.float8_e5m2).float())
