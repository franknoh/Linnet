"""Differential tests: the PyTorch implementations the compiler may select
under `numerics="equivalent"` must agree with the canonical decompositions,
and the `numerics="fast"` tier must agree up to the input dtype's rounding."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from linnet.torch import CompiledLinnetModule, load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.native

use std.linalg::{batched_matmul, matmul}
use std.nn.activations::{gelu, gelu_erf, relu, sigmoid, silu}
use std.nn.attention::{attention, causal_mask}
use std.nn.linear::{linear}
use std.nn.norm::{layer_norm, rms_norm}
use std.nn.softmax::{softmax}

pub block Ops<H: Dim, T: Float> {
    param weight: Tensor[H, H; T]
    param bias: Tensor[H; T]
    param norm: Tensor[H; T]

    pub entry activations<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        return gelu(silu(sigmoid(relu(x))))
    }

    pub entry exact_gelu<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        return gelu_erf(x)
    }

    pub entry projections<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        let y = linear(x, weight, some(bias))
        let z = matmul(y, weight) + batched_matmul(x, weight)
        return softmax(layer_norm(rms_norm(z, norm), norm, some(bias)))
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pytorch_reductions_accumulate_in_f32(dtype: torch.dtype) -> None:
    """Why `torch.softmax` and `torch.rms_norm` need no casts: their kernels
    accumulate in f32 for low-precision inputs, so they return exactly what
    the canonical body's `.float()` ... `.to(dtype)` returns."""
    x = (torch.randn(4, 256, 512) * 3).to(dtype)
    assert torch.equal(torch.softmax(x, dim=-1), torch.softmax(x.float(), dim=-1).to(dtype))
    assert torch.equal(
        torch.rms_norm(x, [512], eps=1e-5),
        torch.rms_norm(x.float(), [512], eps=1e-5).to(dtype),
    )


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_fast_tier_agrees_up_to_input_rounding(tmp_path: Path, dtype: str) -> None:
    source = tmp_path / "native.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generics: dict[str, int | str] = {"H": 8, "T": dtype}
    canonical = load(source, generics=generics, std_root=STDLIB, numerics="exact")
    fast = load(source, generics=generics, std_root=STDLIB, numerics="fast")
    generated = load(source, generics=generics, std_root=STDLIB, numerics="fast", compile=True)
    with torch.no_grad():
        for name, parameter in canonical.named_parameters():
            parameter.copy_(torch.randn(parameter.shape).to(parameter.dtype) * 0.5)
            fast.get_parameter(name).copy_(parameter)
            generated.get_parameter(name).copy_(parameter)

    torch_dtype = torch.float32 if dtype == "f32" else torch.bfloat16
    # In f32 the fast kernels are the equivalent ones; in bf16 they round
    # inside softmax, the norms, and attention, so the tolerance is looser.
    tolerance = 1e-5 if dtype == "f32" else 8e-2
    x = torch.randn(3, 8).to(torch_dtype)
    expected = canonical.run_entry("projections", [x])
    for model in (fast, generated):
        actual = model.run_entry("projections", [x])
        torch.testing.assert_close(actual.float(), expected.float(), atol=tolerance, rtol=tolerance)

    q, k, v = (torch.randn(2, 2, 5, 8).to(torch_dtype) for _ in range(3))
    expected = canonical.run_entry("attend", [q, k, v])
    actual = fast.run_entry("attend", [q, k, v])
    torch.testing.assert_close(actual.float(), expected.float(), atol=tolerance, rtol=tolerance)
    assert isinstance(generated, CompiledLinnetModule)
    generated.run_entry("attend", [q, k, v])
    source_text = generated.generated_source("attend")
    assert "F.scaled_dot_product_attention(" in source_text and ".float()" not in source_text


@pytest.mark.parametrize("numerics", ["exact", "equivalent"])
def test_gelu_erf_is_the_activation_checkpoints_mean(tmp_path: Path, numerics: str) -> None:
    """`gelu_erf` must agree with `F.gelu`, whichever side computes it.

    At `exact` the canonical series runs; at `equivalent` the fused kernel is
    selected. The tanh approximation is three orders of magnitude further
    away, which is the whole reason this op exists.
    """
    source = tmp_path / "native.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    model = load(source, generics={"H": 64, "T": "f32"}, std_root=STDLIB, numerics=numerics)
    x = torch.linspace(-8.0, 8.0, 4 * 64).reshape(4, 64)
    expected = torch.nn.functional.gelu(x, approximate="none")
    got = model.run_entry("exact_gelu", [x])
    torch.testing.assert_close(got, expected, atol=2e-6, rtol=0)
    tanh_form = torch.nn.functional.gelu(x, approximate="tanh")
    assert (tanh_form - expected).abs().max() > 100 * (got - expected).abs().max()


def test_the_erf_gelu_kernel_is_selected(tmp_path: Path) -> None:
    source = tmp_path / "native.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    model = load(source, generics={"H": 64, "T": "f32"}, std_root=STDLIB, compile=True)
    assert isinstance(model, CompiledLinnetModule)
    model.run_entry("exact_gelu", [torch.randn(2, 64)])
    assert 'F.gelu(in_x, approximate="none")' in model.generated_source("exact_gelu")


SPAN_SOURCE = """\
module tests.span

use std.nn.cache::{write_at, write_span}

pub block Cache<S: Dim, N: Dim, T: Float>
where
    S > 0,
    N > 0
{
    pub entry span(cache: Tensor[1, 2, S, 4; T], value: Tensor[1, 2, N, 4; T], at: i32)
        -> Tensor[1, 2, S, 4; T] {
        return write_span(cache, value, at)
    }

    pub entry one(cache: Tensor[1, 2, S, 4; T], value: Tensor[1, 2, 1, 4; T], at: i32)
        -> Tensor[1, 2, S, 4; T] {
        return write_at(cache, value, at)
    }
}
"""


@pytest.mark.parametrize("numerics", ["exact", "equivalent"])
@pytest.mark.parametrize("at", [0, 3, 6])
def test_a_span_lands_where_a_slice_assignment_puts_it(
    tmp_path: Path, numerics: str, at: int
) -> None:
    """A prefill writes the whole prompt into the cache at once; the canonical
    select and the selected slice write must both leave every other position
    alone. `at = 6` with four positions ends exactly at the cache's end."""
    source = tmp_path / "span.linnet"
    source.write_text(SPAN_SOURCE, encoding="utf-8")
    model = load(source, generics={"S": 10, "N": 4, "T": "f32"}, std_root=STDLIB, numerics=numerics)
    cache = torch.randn(1, 2, 10, 4)
    value = torch.randn(1, 2, 4, 4)
    expected = cache.clone()
    expected[:, :, at : at + 4] = value
    got = model.run_entry("span", [cache, value, torch.tensor(at, dtype=torch.int32)])
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    # One position is the special case it always was.
    single = model.run_entry("one", [cache, value[:, :, :1], torch.tensor(at, dtype=torch.int32)])
    expected_one = cache.clone()
    expected_one[:, :, at] = value[:, :, 0]
    torch.testing.assert_close(single, expected_one, atol=0, rtol=0)


def test_the_span_write_is_a_slice_write(tmp_path: Path) -> None:
    source = tmp_path / "span.linnet"
    source.write_text(SPAN_SOURCE, encoding="utf-8")
    model = load(source, generics={"S": 10, "N": 4, "T": "f32"}, std_root=STDLIB, compile=True)
    assert isinstance(model, CompiledLinnetModule)
    model.run_entry(
        "span",
        [torch.zeros(1, 2, 10, 4), torch.ones(1, 2, 4, 4), torch.tensor(2, dtype=torch.int32)],
    )
    text = model.generated_source("span")
    assert "index_copy(2, " in text and "torch.arange(4, device=" in text
