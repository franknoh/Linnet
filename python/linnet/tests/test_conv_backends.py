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


BATCH_NORM_SOURCE = """\
module tests.convolve_norm

use std.nn.conv::{conv2d}
use std.nn.norm::{batch_norm}

pub block Model<T: Float = f32> {
    param weight: Tensor[4, 3, 3, 3; T]
    param mean: Tensor[4; T]
    param variance: Tensor[4; T]
    param scale: Tensor[4; T]
    param shift: Tensor[4; T]
    param unused: Tensor[2; T]

    pub entry forward<B: Dim>(x: Tensor[B, 3, 8, 8; T]) -> Tensor[B, 4, 8, 8; T] {
        let y = conv2d<B, 3, 4, 8, 8, 3, 1, 1, T>(x, weight, none)
        return batch_norm<B, 4, 8, 8, T>(y, mean, variance, scale, shift)
    }
}
"""


def test_onnx_runtime_folds_batch_norm_into_the_convolution(tmp_path: Path) -> None:
    """A batch norm is ONNX's `BatchNormalization`, and an entry without
    state takes the weights as initializers, so ONNX Runtime can fold one
    into the other -- still computing what `F.batch_norm` does."""
    pytest.importorskip("onnxruntime")
    from linnet.onnx import export_model, load_model

    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    source = tmp_path / "convolve_norm.linnet"
    source.write_text(BATCH_NORM_SOURCE, encoding="utf-8")
    tensors = {
        "weight": torch.randn(4, 3, 3, 3),
        "mean": torch.randn(4),
        "variance": torch.rand(4) + 0.5,
        "scale": torch.randn(4),
        "shift": torch.randn(4),
        "unused": torch.randn(2),
    }
    weights = tmp_path / "model.safetensors"
    save_file(tensors, str(weights))
    x = torch.randn(2, 3, 8, 8)
    expected = functional.batch_norm(
        functional.conv2d(x, tensors["weight"], padding=1),
        tensors["mean"],
        tensors["variance"],
        tensors["scale"],
        tensors["shift"],
    )
    exported = export_model(source, generics={"B": 2}, weights=weights, std_root=STDLIB)
    assert any(node.op_type == "BatchNormalization" for node in exported.model.graph.node)
    model = load_model(
        source, generics={}, weights=weights, std_root=STDLIB, providers=["CPUExecutionProvider"]
    )
    got = model.run_entry("forward", [x.numpy()])
    np.testing.assert_allclose(got, expected.numpy(), atol=1e-5, rtol=1e-5)
    # An input placed once, as a benchmark or a server keeps one, is bound
    # as it is.
    placed = model.run_entry("forward", [model.place(x.numpy(), "f32")])
    np.testing.assert_array_equal(placed, got)
    kept = model.run_entry("forward", [x.numpy()], keep_on_device=True)
    np.testing.assert_array_equal(kept.numpy(), got)


SHAPES_SOURCE = """\
module tests.conv_shapes

use std.nn.conv::{Conv1d, Conv2dRect, conv1d, conv2d_rect}

pub entry signal<B: Dim, L: Dim>(
    x: Tensor[B, 3, L; f32],
    weight: Tensor[4, 3, 5; f32],
    bias: Tensor[4; f32],
) -> Tensor[B, 4, (L + 2 * 2 - 5) / 2 + 1; f32] {
    return conv1d<B, 3, 4, L, 5, 2, 2, f32>(x, weight, some(bias))
}

pub entry rectangle<B: Dim, H: Dim, W: Dim>(
    x: Tensor[B, 3, H, W; f32],
    weight: Tensor[4, 3, 3, 5; f32],
) -> Tensor[B, 4, (H + 2 * 1 - 3) / 1 + 1, (W + 2 * 2 - 5) / 2 + 1; f32] {
    return conv2d_rect<B, 3, 4, H, W, 3, 5, 1, 2, 1, 2, f32>(x, weight, none)
}

pub block Model {
    sub signal_conv: Conv1d<3, 4, 5, 2, 2, f32>
    sub image_conv: Conv2dRect<3, 4, 3, 5, 1, 2, 1, 2, f32>

    pub entry forward<B: Dim, L: Dim, H: Dim, W: Dim>(
        signal: Tensor[B, 3, L; f32],
        image: Tensor[B, 3, H, W; f32],
    ) -> (
        Tensor[B, 4, (L + 2 * 2 - 5) / 2 + 1; f32],
        Tensor[B, 4, (H + 2 * 1 - 3) / 1 + 1, (W + 2 * 2 - 5) / 2 + 1; f32],
    ) {
        return (signal_conv.forward(signal), image_conv.forward(image))
    }
}
"""


def _shapes_case(
    tmp_path: Path,
) -> tuple[Path, dict[str, torch.Tensor], dict[str, tuple[tuple[np.ndarray, ...], np.ndarray]]]:
    """The source, the block's weights, and per function its inputs and what
    `F.conv1d` / `F.conv2d` give for them."""
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    source = tmp_path / "conv_shapes.linnet"
    source.write_text(SHAPES_SOURCE, encoding="utf-8")
    weights = {
        "signal_conv.weight": torch.randn(4, 3, 5),
        "signal_conv.bias": torch.randn(4),
        "image_conv.weight": torch.randn(4, 3, 3, 5),
    }
    signal, image = torch.randn(2, 3, 11), torch.randn(2, 3, 6, 9)
    one = functional.conv1d(
        signal, weights["signal_conv.weight"], weights["signal_conv.bias"], stride=2, padding=2
    )
    two = functional.conv2d(image, weights["image_conv.weight"], stride=(1, 2), padding=(1, 2))
    cases = {
        "signal": (
            (
                signal.numpy(),
                weights["signal_conv.weight"].numpy(),
                weights["signal_conv.bias"].numpy(),
            ),
            one.numpy(),
        ),
        "rectangle": ((image.numpy(), weights["image_conv.weight"].numpy()), two.numpy()),
    }
    return source, weights, cases


@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize("name", ["signal", "rectangle"])
def test_conv1d_and_rectangular_conv2d_in_pytorch(tmp_path: Path, compile: bool, name: str) -> None:
    """Interpreted, `numerics="exact"` runs the canonical gather; generated
    under the default numerics, `F.conv1d` and `F.conv2d` with a tuple
    geometry."""
    from linnet.torch import load_function

    source, _, cases = _shapes_case(tmp_path)
    numerics = "fast" if compile else "exact"
    function = load_function(source, name, std_root=STDLIB, numerics=numerics, compile=compile)
    inputs, expected = cases[name]
    got = function(*(torch.tensor(value) for value in inputs))
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-5, rtol=1e-5)
    if compile:
        assert ("F.conv1d(" if name == "signal" else "F.conv2d(") in function.generated_source()


@pytest.mark.parametrize("name", ["signal", "rectangle"])
def test_conv1d_and_rectangular_conv2d_in_jax_and_onnx(tmp_path: Path, name: str) -> None:
    source, _, cases = _shapes_case(tmp_path)
    inputs, expected = cases[name]
    pytest.importorskip("jax")
    from linnet.jax import load_function as load_jax_function

    function = load_jax_function(source, name, std_root=STDLIB)
    np.testing.assert_allclose(np.asarray(function(*inputs)), expected, atol=1e-5, rtol=1e-5)
    assert "conv_general_dilated" in function.generated_source()

    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_function

    generics: dict[str, int | str] = (
        {"B": 2, "L": 11} if name == "signal" else {"B": 2, "H": 6, "W": 9}
    )
    exported = export_function(source, name, generics=generics, std_root=STDLIB)
    assert any(node.op_type == "Conv" for node in exported.model.graph.node)
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds = {port.name: value for port, value in zip(exported.inputs, inputs, strict=True)}
    (got,) = session.run(None, feeds)
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_conv1d_and_rectangular_conv2d_under_xla(tmp_path: Path) -> None:
    """The blocks, through StableHLO's `convolution` with one and two
    spatial axes."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from linnet.jax import load as load_jax

    source, weights, cases = _shapes_case(tmp_path)
    path = tmp_path / "model.safetensors"
    save_file(weights, str(path))
    function = load_jax(source, generics={}, weights=path, std_root=STDLIB, root="Model")
    signal, image = cases["signal"][0][0], cases["rectangle"][0][0]
    one, two = function(jnp.asarray(signal), jnp.asarray(image))  # pyright: ignore[reportUnknownMemberType]
    np.testing.assert_allclose(np.asarray(one), cases["signal"][1], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(two), cases["rectangle"][1], atol=1e-5, rtol=1e-5)
