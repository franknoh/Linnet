"""Max pooling as each target's own pooling -- `F.max_pool2d` under torch,
`reduce_window` under XLA, ONNX's `MaxPool` -- computing what the gather body
does. Before they had one, an export read every window through an index
table: 1.8 GB of it for a ResNet stem at batch 32."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.compiler import find_compiler

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.pool

use std.nn.pool::{max_pool2d}

pub block Model<C: Dim, H: Dim, K: Dim, Stride: Dim, Pad: Dim, T: Float = f32>
where
    Stride > 0,
    K > 0
{
    param scale: Tensor[C; T]

    pub entry forward<B: Dim>(
        x: Tensor[B, C, H, H; T],
    ) -> Tensor[B, C, (H + 2 * Pad - K) / Stride + 1, (H + 2 * Pad - K) / Stride + 1; T] {
        let scaled[b, c, h, w] = x[b, c, h, w] * scale[c]
        return max_pool2d<B, C, H, H, K, Stride, Pad, T>(scaled)
    }
}
"""

# (H, K, Stride, Pad): a ResNet stem's pooling, one without padding, and one
# whose windows overlap nothing.
GEOMETRIES = [(16, 3, 2, 1), (9, 3, 1, 0), (8, 2, 2, 0)]


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
    source = tmp_path / "pool.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    scale = torch.randn(3)
    weights = tmp_path / "model.safetensors"
    save_file({"scale": scale}, str(weights))
    # Negative values, so padding that won a window would show.
    x = torch.randn(2, 3, height, height) - 2.0
    expected = functional.max_pool2d(x * scale[:, None, None], kernel, stride=stride, padding=pad)
    generics: dict[str, int | str] = {
        "C": 3,
        "H": height,
        "K": kernel,
        "Stride": stride,
        "Pad": pad,
    }
    return source, weights, x, expected, generics


@pytest.mark.parametrize(("height", "kernel", "stride", "pad"), GEOMETRIES)
def test_torch_pooling_is_f_max_pool2d(
    tmp_path: Path, height: int, kernel: int, stride: int, pad: int
) -> None:
    from linnet.torch import load

    source, weights, x, expected, generics = _case(tmp_path, height, kernel, stride, pad)
    for compile in (False, True):
        model = load(source, generics=generics, weights=weights, std_root=STDLIB, compile=compile)
        torch.testing.assert_close(model.run_entry("forward", [x]), expected)
    command = [find_compiler(), "torch", "--root", "Model", "--entry", "forward"]
    command += [a for k, v in {**generics, "B": 2}.items() for a in ("--bind", f"{k}={v}")]
    command += ["--std", str(STDLIB), str(source)]
    assert "F.max_pool2d(" in subprocess.run(command, capture_output=True, text=True).stdout


@pytest.mark.parametrize(("height", "kernel", "stride", "pad"), GEOMETRIES)
@pytest.mark.parametrize("generated", [False, True])
def test_xla_pooling_is_f_max_pool2d(
    tmp_path: Path, height: int, kernel: int, stride: int, pad: int, generated: bool
) -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from linnet.jax import load_model

    source, weights, x, expected, generics = _case(tmp_path, height, kernel, stride, pad)
    model = load_model(
        source, generics=generics, weights=weights, std_root=STDLIB, generated=generated
    )
    got = np.asarray(model.run_entry("forward", [jnp.asarray(x.numpy())]))
    np.testing.assert_allclose(got, expected.numpy(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(("height", "kernel", "stride", "pad"), GEOMETRIES)
def test_onnx_max_pool_is_f_max_pool2d(
    tmp_path: Path, height: int, kernel: int, stride: int, pad: int
) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_model

    source, weights, x, expected, generics = _case(tmp_path, height, kernel, stride, pad)
    exported = export_model(source, generics={**generics, "B": 2}, weights=weights, std_root=STDLIB)
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (got,) = session.run(None, {session.get_inputs()[0].name: x.numpy()})
    np.testing.assert_allclose(got, expected.numpy(), atol=1e-6, rtol=1e-6)
    assert any(node.op_type == "MaxPool" for node in exported.model.graph.node)
    assert not any(node.op_type == "GatherND" for node in exported.model.graph.node)


UPSAMPLE = """\
module tests.upsample

use std.nn.resize::{upsample_nearest2d}

pub block Model<C: Dim, H: Dim, T: Float = f32> {
    param scale: Tensor[C; T]

    pub entry forward<B: Dim>(x: Tensor[B, C, H, H; T]) -> Tensor[B, C, 2 * H, 2 * H; T] {
        let scaled[b, c, h, w] = x[b, c, h, w] * scale[c]
        return upsample_nearest2d<B, C, H, H, 2, T>(scaled)
    }
}
"""


def test_onnx_upsample_is_resize(tmp_path: Path) -> None:
    """Nearest upsampling is ONNX's `Resize`, not four stacked copies: at a
    VAE's last stage those were past the 2^31 elements ONNX Runtime's CUDA
    `Concat` indexes."""
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_model

    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    source = tmp_path / "upsample.linnet"
    source.write_text(UPSAMPLE, encoding="utf-8")
    scale = torch.randn(3)
    save_file({"scale": scale}, str(tmp_path / "model.safetensors"))
    x = torch.randn(2, 3, 4, 4)
    expected = functional.interpolate(x * scale[:, None, None], scale_factor=2, mode="nearest")
    exported = export_model(
        source,
        generics={"C": 3, "H": 4, "B": 2},
        weights=tmp_path / "model.safetensors",
        std_root=STDLIB,
    )
    assert any(node.op_type == "Resize" for node in exported.model.graph.node)
    assert not any(node.op_type == "Concat" for node in exported.model.graph.node)
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (got,) = session.run(None, {session.get_inputs()[0].name: x.numpy()})
    np.testing.assert_array_equal(got, expected.numpy())
