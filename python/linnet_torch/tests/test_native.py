"""Differential tests: the PyTorch implementations the compiler may select
under `numerics="equivalent"` must agree with the canonical decompositions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from linnet_torch import load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.native

use std.linalg::{batched_matmul, matmul}
use std.nn.activations::{gelu, relu, sigmoid, silu}
use std.nn.attention::{attention, causal_mask}
use std.nn.linear::{linear}
use std.nn.norm::{rms_norm}
use std.nn.softmax::{softmax}

pub block Ops<H: Dim, T: Float> {
    param weight: Tensor[H, H; T]
    param bias: Tensor[H; T]
    param norm: Tensor[H; T]

    pub entry activations<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        return gelu(silu(sigmoid(relu(x))))
    }

    pub entry projections<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        let y = linear(x, weight, some(bias))
        let z = matmul(y, weight) + batched_matmul(x, weight)
        return softmax(rms_norm(z, norm))
    }

    pub entry attend<B: Dim, N: Dim, S: Dim>(
        q: Tensor[B, N, S, H; T],
        k: Tensor[B, N, S, H; T],
        v: Tensor[B, N, S, H; T],
    ) -> Tensor[B, N, S, H; T] {
        return attention(q, k, v, 0.25, some(causal_mask<S, S>()))
    }
}
"""


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_native_implementations_agree_with_canonical(tmp_path: Path, dtype: str) -> None:
    source = tmp_path / "native.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generics: dict[str, int | str] = {"H": 8, "T": dtype}
    canonical = load(source, generics=generics, std_root=STDLIB, numerics="exact")
    native = load(source, generics=generics, std_root=STDLIB, numerics="equivalent")
    with torch.no_grad():
        for name, parameter in canonical.named_parameters():
            parameter.copy_(torch.randn(parameter.shape).to(parameter.dtype) * 0.5)
            native.get_parameter(name).copy_(parameter)

    torch_dtype = torch.float32 if dtype == "f32" else torch.bfloat16
    tolerance = 1e-5 if dtype == "f32" else 3e-2
    x = torch.randn(3, 8).to(torch_dtype)
    for entry in ("activations", "projections"):
        expected = getattr(canonical, entry)(x)
        actual = getattr(native, entry)(x)
        torch.testing.assert_close(actual.float(), expected.float(), atol=tolerance, rtol=tolerance)

    q, k, v = (torch.randn(2, 2, 5, 8).to(torch_dtype) for _ in range(3))
    expected = canonical.run_entry("attend", [q, k, v])
    actual = native.run_entry("attend", [q, k, v])
    torch.testing.assert_close(actual.float(), expected.float(), atol=tolerance, rtol=tolerance)
