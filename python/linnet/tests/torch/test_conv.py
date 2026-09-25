"""Convolution, pooling, and batch normalization: the canonical bodies and
the kernels the compiler may select must both compute what PyTorch does."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from linnet.torch import CompiledLinnetModule, load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.conv

use std.nn.conv::{conv2d}
use std.nn.norm::{batch_norm}
use std.nn.norm::{group_norm}
use std.nn.pool::{global_average_pool2d, max_pool2d}
use std.nn.resize::{upsample_nearest2d}

pub block Ops<Cin: Dim, Cout: Dim, H: Dim, W: Dim, K: Dim, Stride: Dim, Pad: Dim, T: Float>
where
    Stride > 0,
    K > 0
{
    param weight: Tensor[Cout, Cin, K, K; T]
    param bias: Tensor[Cout; T]
    param mean: Tensor[Cin; T]
    param variance: Tensor[Cin; T]
    param scale: Tensor[Cin; T]
    param shift: Tensor[Cin; T]

    pub entry convolve<B: Dim>(
        x: Tensor[B, Cin, H, W; T],
    ) -> Tensor[B, Cout, (H + 2 * Pad - K) / Stride + 1, (W + 2 * Pad - K) / Stride + 1; T] {
        return conv2d<B, Cin, Cout, H, W, K, Stride, Pad, T>(x, weight, some(bias))
    }

    pub entry pool<B: Dim>(
        x: Tensor[B, Cin, H, W; T],
    ) -> Tensor[B, Cin, (H + 2 * Pad - K) / Stride + 1, (W + 2 * Pad - K) / Stride + 1; T] {
        return max_pool2d<B, Cin, H, W, K, Stride, Pad, T>(x)
    }

    pub entry normalize<B: Dim>(x: Tensor[B, Cin, H, W; T]) -> Tensor[B, Cin; T] {
        return global_average_pool2d(batch_norm(x, mean, variance, scale, shift))
    }

    pub entry grouped<B: Dim, Groups: Dim>(x: Tensor[B, Cin, H, W; T]) -> Tensor[B, Cin, H, W; T]
    where Groups > 0, Cin % Groups == 0 {
        return group_norm<B, Cin, H, W, Groups, T>(x, scale, shift, 1e-6)
    }

    pub entry upsample<B: Dim>(x: Tensor[B, Cin, H, W; T]) -> Tensor[B, Cin, H * 2, W * 2; T] {
        return upsample_nearest2d<B, Cin, H, W, 2, T>(x)
    }
}
"""

GEOMETRIES = [
    (7, 2, 3),  # a ResNet stem
    (3, 1, 1),  # a residual block
    (3, 2, 1),  # a downsampling block
    (1, 2, 0),  # a projection shortcut
]


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(("kernel", "stride", "pad"), GEOMETRIES)
@pytest.mark.parametrize("numerics", ["exact", "equivalent"])
def test_conv2d_matches_pytorch(
    tmp_path: Path, kernel: int, stride: int, pad: int, numerics: str
) -> None:
    source = tmp_path / "conv.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generics: dict[str, int | str] = {
        "Cin": 3,
        "Cout": 4,
        "H": 16,
        "W": 16,
        "K": kernel,
        "Stride": stride,
        "Pad": pad,
        "T": "f32",
    }
    model = load(source, generics=generics, std_root=STDLIB, numerics=numerics)
    weight = torch.randn(4, 3, kernel, kernel)
    bias = torch.randn(4)
    with torch.no_grad():
        model.get_parameter("root.weight").copy_(weight)
        model.get_parameter("root.bias").copy_(bias)
    x = torch.randn(2, 3, 16, 16)
    expected = functional.conv2d(x, weight, bias, stride=stride, padding=pad)
    torch.testing.assert_close(model.run_entry("convolve", [x]), expected, atol=1e-5, rtol=1e-5)

    pooled = model.run_entry("pool", [x])
    torch.testing.assert_close(
        pooled, functional.max_pool2d(x, kernel, stride=stride, padding=pad), atol=0, rtol=0
    )


def test_batch_norm_and_pooling_match_pytorch(tmp_path: Path) -> None:
    source = tmp_path / "conv.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generics: dict[str, int | str] = {
        "Cin": 5,
        "Cout": 2,
        "H": 8,
        "W": 8,
        "K": 3,
        "Stride": 1,
        "Pad": 1,
        "T": "f32",
    }
    model = load(source, generics=generics, std_root=STDLIB)
    mean, scale, shift = torch.randn(5), torch.randn(5), torch.randn(5)
    variance = torch.rand(5) + 0.5
    with torch.no_grad():
        model.get_parameter("root.mean").copy_(mean)
        model.get_parameter("root.variance").copy_(variance)
        model.get_parameter("root.scale").copy_(scale)
        model.get_parameter("root.shift").copy_(shift)
    x = torch.randn(2, 5, 8, 8)
    expected = functional.batch_norm(x, mean, variance, scale, shift, False, 0.0, 1e-5).mean(
        dim=(2, 3)
    )
    torch.testing.assert_close(model.run_entry("normalize", [x]), expected, atol=1e-5, rtol=1e-5)


def test_the_kernels_are_selected(tmp_path: Path) -> None:
    source = tmp_path / "conv.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generics: dict[str, int | str] = {
        "Cin": 3,
        "Cout": 4,
        "H": 16,
        "W": 16,
        "K": 3,
        "Stride": 2,
        "Pad": 1,
        "T": "f32",
    }
    model = load(source, generics=generics, std_root=STDLIB, compile=True)
    assert isinstance(model, CompiledLinnetModule)
    model.run_entry("convolve", [torch.randn(1, 3, 16, 16)])
    source_text = model.generated_source("convolve")
    # The geometry is recovered from the shapes, not written by the source.
    assert "F.conv2d(in_x, p0, p1, stride=2, padding=1)" in source_text


@pytest.mark.parametrize("groups", [1, 5])
def test_group_norm_and_upsampling_match_pytorch(tmp_path: Path, groups: int) -> None:
    source = tmp_path / "conv.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generics: dict[str, int | str] = {
        "Cin": 10,
        "Cout": 2,
        "H": 6,
        "W": 8,
        "K": 3,
        "Stride": 1,
        "Pad": 1,
        "T": "f32",
    }
    model = load(source, generics=generics, std_root=STDLIB)
    scale, shift = torch.randn(10), torch.randn(10)
    with torch.no_grad():
        model.get_parameter("root.scale").copy_(scale)
        model.get_parameter("root.shift").copy_(shift)
    x = torch.randn(2, 10, 6, 8)
    torch.testing.assert_close(
        model.run_entry("grouped", [x], generics={"Groups": groups}),
        functional.group_norm(x, groups, scale, shift, eps=1e-6),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        model.run_entry("upsample", [x]),
        functional.interpolate(x, scale_factor=2, mode="nearest"),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("compile", [False, True])
def test_a_geometry_the_shapes_cannot_tell_apart(tmp_path: Path, compile: bool) -> None:
    """A 3x3 window taking 4 positions to 2 fits stride 2 with padding 1 and
    stride 1 without padding alike. The kernel must take the call's own
    `Stride` and `Pad`; recovering them from the shapes picked the wrong pair
    at the bottom of every small UNet."""
    source = tmp_path / "conv.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generics: dict[str, int | str] = {
        "Cin": 3,
        "Cout": 4,
        "H": 4,
        "W": 4,
        "K": 3,
        "Stride": 2,
        "Pad": 1,
        "T": "f32",
    }
    model = load(source, generics=generics, std_root=STDLIB, numerics="equivalent", compile=compile)
    weight, bias = torch.randn(4, 3, 3, 3), torch.randn(4)
    with torch.no_grad():
        model.get_parameter("root.weight").copy_(weight)
        model.get_parameter("root.bias").copy_(bias)
    x = torch.randn(2, 3, 4, 4)
    expected = functional.conv2d(x, weight, bias, stride=2, padding=1)
    torch.testing.assert_close(model.run_entry("convolve", [x]), expected, atol=1e-5, rtol=1e-5)
