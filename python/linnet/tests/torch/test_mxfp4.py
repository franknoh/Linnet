"""MXFP4 expert weights (`std.quant`): the format decoded, and the experts'
products read from it -- the canonical bodies, PyTorch's unpacked arithmetic
on the CPU, and Linnet's Triton kernel on CUDA -- agree."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.torch import CompiledLinnetModule, load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.mxfp4

use std.quant::{
    dequantize_mxfp4,
    mxfp4_combine_experts,
    mxfp4_experts,
    mxfp4_experts_shared,
    mxfp4_linear_experts_shared,
}

pub block Model<E: Dim, Out: Dim, G: Dim, K: Dim, T: Float = f32> {
    param blocks: Tensor[E, Out, G, 16; u8]
    param scales: Tensor[E, Out, G; u8]

    pub entry weights() -> Tensor[E, Out, G * 32; T] {
        return dequantize_mxfp4<E, Out, G, T>(blocks, scales)
    }

    pub entry each<R: Dim>(
        x: Tensor[R, K, G * 32; T],
        experts: Tensor[R, K; i32],
    ) -> Tensor[R, K, Out; T] {
        let chosen[r, k] = cast<i64>(experts[r, k])
        return mxfp4_experts<R, K, E, Out, G, T>(x, blocks, scales, chosen)
    }

    pub entry shared<R: Dim>(
        x: Tensor[R, 1, G * 32; T],
        experts: Tensor[R, K; i32],
    ) -> Tensor[R, K, Out; T] {
        let chosen[r, k] = cast<i64>(experts[r, k])
        return mxfp4_experts_shared<R, K, E, Out, G, T>(x, blocks, scales, chosen)
    }

    pub entry routed<R: Dim>(
        x: Tensor[R, G * 32; T],
        experts: Tensor[R, K; i32],
    ) -> Tensor[R, K, Out; T] {
        let chosen[r, k] = cast<i64>(experts[r, k])
        return mxfp4_linear_experts_shared<R, K, E, Out, G, T>(x, blocks, scales, chosen)
    }

    pub entry combined<R: Dim>(
        x: Tensor[R, K, G * 32; T],
        experts: Tensor[R, K; i32],
        weights: Tensor[R, K; T],
    ) -> Tensor[R, Out; T] {
        let chosen[r, k] = cast<i64>(experts[r, k])
        return mxfp4_combine_experts<R, K, E, Out, G, T>(x, blocks, scales, chosen, weights)
    }
}
"""

GENERICS: dict[str, int | str] = {"E": 4, "Out": 24, "G": 3, "K": 2}
FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.fixture
def model_files(tmp_path: Path) -> tuple[Path, Path, torch.Tensor, torch.Tensor]:
    source = tmp_path / "mxfp4.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generator = torch.Generator().manual_seed(0)
    blocks = torch.randint(0, 256, (4, 24, 3, 16), generator=generator, dtype=torch.uint8)
    scales = torch.randint(120, 132, (4, 24, 3), generator=generator, dtype=torch.uint8)
    weights = tmp_path / "model.safetensors"
    save_file({"blocks": blocks, "scales": scales}, str(weights))
    return source, weights, blocks, scales


def _dequantized(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(FP4)
    low, high = table[(blocks & 15).long()], table[(blocks >> 4).long()]
    values = torch.stack([low, high], dim=-1).reshape(*blocks.shape[:-1], 32)
    return (values * torch.exp2(scales.float() - 127)[..., None]).reshape(4, 24, 96)


def _inputs(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    each = torch.randn(3, 2, 96, generator=generator)
    shared = torch.randn(3, 1, 96, generator=generator)
    experts = torch.tensor([[0, 3], [2, 1], [3, 0]], dtype=torch.int32)
    return each.to(device), shared.to(device), experts.to(device)


def test_every_code_decodes_to_its_value(
    model_files: tuple[Path, Path, torch.Tensor, torch.Tensor],
) -> None:
    source, weights, blocks, scales = model_files
    model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, numerics="exact")
    got = model.run_entry("weights", [])
    torch.testing.assert_close(got, _dequantized(blocks, scales), atol=0, rtol=0)


@pytest.mark.parametrize("numerics", ["exact", "fast"])
def test_the_experts_products(
    model_files: tuple[Path, Path, torch.Tensor, torch.Tensor], numerics: str
) -> None:
    source, weights, blocks, scales = model_files
    weight = _dequantized(blocks, scales)
    each, shared, experts = _inputs("cpu")
    model = load(
        source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True, numerics=numerics
    )
    taken = weight[experts.long()]
    expected_each = torch.einsum("rki,rkoi->rko", each, taken)
    expected_shared = torch.einsum("ri,rkoi->rko", shared[:, 0], taken)
    torch.testing.assert_close(model.run_entry("each", [each, experts]), expected_each)
    torch.testing.assert_close(model.run_entry("shared", [shared, experts]), expected_shared)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel runs on CUDA")
def test_the_kernel_reads_the_bytes_as_they_are(
    model_files: tuple[Path, Path, torch.Tensor, torch.Tensor],
) -> None:
    pytest.importorskip("triton")
    source, weights, blocks, scales = model_files
    weight = _dequantized(blocks, scales)
    each, shared, experts = _inputs("cuda")
    model = load(
        source, generics=GENERICS, std_root=STDLIB, weights=weights, device="cuda", compile=True
    )
    taken = weight[experts.long().cpu()]
    expected_each = torch.einsum("rki,rkoi->rko", each.cpu(), taken)
    expected_shared = torch.einsum("ri,rkoi->rko", shared[:, 0].cpu(), taken)
    torch.testing.assert_close(
        model.run_entry("each", [each, experts]).cpu(), expected_each, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        model.run_entry("shared", [shared, experts]).cpu(), expected_shared, atol=1e-4, rtol=1e-4
    )


def _routed_inputs(rows: int, device: str, dtype: torch.dtype) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(rows, 96, generator=generator)
    each = torch.randn(rows, 2, 96, generator=generator)
    experts = torch.stack([torch.randperm(4, generator=generator)[:2] for _ in range(rows)])
    weights = torch.rand(rows, 2, generator=generator)
    return [
        x.to(device, dtype),
        each.to(device, dtype),
        experts.int().to(device),
        weights.to(device, dtype),
    ]


@pytest.mark.parametrize(("numerics", "compile"), [("exact", False), ("fast", True)])
def test_the_routed_forms_multiply_each_row_by_its_experts(
    model_files: tuple[Path, Path, torch.Tensor, torch.Tensor], numerics: str, compile: bool
) -> None:
    """`mxfp4_linear_experts_shared` and `mxfp4_combine_experts` against the
    dequantized experts: the bodies' dense form, and the generated code's
    fallback without CUDA."""
    source, weights_file, blocks, scales = model_files
    weight = _dequantized(blocks, scales)
    model = load(
        source,
        generics=GENERICS,
        std_root=STDLIB,
        weights=weights_file,
        compile=compile,
        numerics=numerics,
    )
    x, each, experts, weights = _routed_inputs(5, "cpu", torch.float32)
    taken = weight[experts.long()]
    expected = torch.einsum("ri,rkoi->rko", x, taken)
    combined = torch.einsum("rki,rkoi->rko", each, taken)
    # The dense form sums in another order than the reference: f32 apart.
    torch.testing.assert_close(
        model.run_entry("routed", [x, experts]), expected, atol=1e-4, rtol=1e-5
    )
    torch.testing.assert_close(
        model.run_entry("combined", [each, experts, weights]),
        (combined * weights[..., None]).sum(1),
        atol=1e-4,
        rtol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel runs on CUDA")
@pytest.mark.parametrize("rows", [3, 96])
def test_the_routed_experts_on_cuda(
    model_files: tuple[Path, Path, torch.Tensor, torch.Tensor], rows: int
) -> None:
    """`linnet.torch.moe` in bf16 on a Hopper GPU: each expert multiplied by
    the rows that chose it, for a few rows and for many."""
    pytest.importorskip("triton")
    source, weights_file, blocks, scales = model_files
    weight = _dequantized(blocks, scales)
    model = load(
        source,
        generics={**GENERICS, "T": "bf16"},
        std_root=STDLIB,
        weights=weights_file,
        device="cuda",
        compile=True,
    )
    x, each, experts, weights = _routed_inputs(rows, "cuda", torch.bfloat16)
    taken = weight[experts.long().cpu()]
    expected = torch.einsum("ri,rkoi->rko", x.float().cpu(), taken)
    combined = torch.einsum("rki,rkoi->rko", each.float().cpu(), taken)
    combined = (combined * weights.float().cpu()[..., None]).sum(1)
    # bf16 products, each rounded before it is weighed: within a few bf16
    # steps of the largest value.
    got = model.run_entry("routed", [x, experts]).float().cpu()
    torch.testing.assert_close(got, expected, atol=2e-2 * float(expected.abs().max()), rtol=0)
    got = model.run_entry("combined", [each, experts, weights]).float().cpu()
    torch.testing.assert_close(got, combined, atol=2e-2 * float(combined.abs().max()), rtol=0)
    assert isinstance(model, CompiledLinnetModule)
    assert "_mxfp4_grouped(" in model.generated_source("combined")
