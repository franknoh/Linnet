"""Convolution as each export target's own convolution: StableHLO's
`convolution` under XLA, and ONNX's `Conv`, must compute what `F.conv2d`
does, including where the shapes alone could not say which geometry is meant.

Before they had one, both exports wrote the canonical gather, whose index
table for a VAE's first convolution has 2.4 billion entries."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from safetensors.torch import save_file  # type: ignore[import-untyped]

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.convolve

use std.nn.conv::{conv2d}

pub block Model<Cin: Dim, Cout: Dim, H: Dim, K: Dim, Stride: Dim, Pad: Dim, T: Float = f32>
where
    Stride > 0,
    K > 0
{
    param weight: Tensor[Cout, Cin, K, K; T]
    param bias: Tensor[Cout; T]

    pub entry forward<B: Dim>(
        x: Tensor[B, Cin, H, H; T],
    ) -> Tensor[B, Cout, (H + 2 * Pad - K) / Stride + 1, (H + 2 * Pad - K) / Stride + 1; T] {
        return conv2d<B, Cin, Cout, H, H, K, Stride, Pad, T>(x, weight, some(bias))
    }
}
"""

# (H, K, Stride, Pad): a ResNet stem, a residual block, a downsampling block,
# and the geometry whose shapes fit two answers (stride 1 with no padding).
GEOMETRIES = [(16, 7, 2, 3), (16, 3, 1, 1), (16, 3, 2, 1), (4, 3, 2, 1)]


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


def _case(
    tmp_path: Path, height: int, kernel: int, stride: int, pad: int
) -> tuple[Path, Path, torch.Tensor, torch.Tensor, dict[str, int | str]]:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    source = tmp_path / "convolve.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    weight, bias = torch.randn(4, 3, kernel, kernel), torch.randn(4)
    weights = tmp_path / "model.safetensors"
    save_file({"weight": weight, "bias": bias}, str(weights))
    x = torch.randn(2, 3, height, height)
    expected = functional.conv2d(x, weight, bias, stride=stride, padding=pad)
    generics: dict[str, int | str] = {
        "Cin": 3,
        "Cout": 4,
        "H": height,
        "K": kernel,
        "Stride": stride,
        "Pad": pad,
    }
    return source, weights, x, expected, generics


@pytest.mark.parametrize(("height", "kernel", "stride", "pad"), GEOMETRIES)
def test_xla_convolution_is_f_conv2d(
    tmp_path: Path, height: int, kernel: int, stride: int, pad: int
) -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from linnet.jax import load as load_jax

    source, weights, x, expected, generics = _case(tmp_path, height, kernel, stride, pad)
    function = load_jax(source, generics=generics, weights=weights, std_root=STDLIB)
    got = np.asarray(function(jnp.asarray(x.numpy())))  # pyright: ignore[reportUnknownMemberType]
    np.testing.assert_allclose(got, expected.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(("height", "kernel", "stride", "pad"), GEOMETRIES)
def test_onnx_conv_is_f_conv2d(
    tmp_path: Path, height: int, kernel: int, stride: int, pad: int
) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_model

    source, weights, x, expected, generics = _case(tmp_path, height, kernel, stride, pad)
    exported = export_model(source, generics={**generics, "B": 2}, weights=weights, std_root=STDLIB)
    text = exported.model.SerializeToString()
    session = onnxruntime.InferenceSession(text, providers=["CPUExecutionProvider"])
    (got,) = session.run(None, {session.get_inputs()[0].name: x.numpy()})
    np.testing.assert_allclose(got, expected.numpy(), atol=1e-5, rtol=1e-5)
    assert any(node.op_type == "Conv" for node in exported.model.graph.node)
