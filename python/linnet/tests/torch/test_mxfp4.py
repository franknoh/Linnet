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

from linnet.torch import load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.mxfp4

use std.quant::{dequantize_mxfp4, mxfp4_experts, mxfp4_experts_shared}

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
