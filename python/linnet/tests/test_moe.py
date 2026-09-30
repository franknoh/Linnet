"""The products of a mixture of experts (`std.nn.moe::linear_experts`): each
row through the experts it chose. The body gathers the chosen weights; on
CUDA in bf16 PyTorch runs one grouped matrix product over the rows sorted by
expert, reading the weights where they lie, inside a CUDA graph too."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.torch import CompiledLinnetModule, load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.moe

use std.nn.moe::{linear_experts}

pub block Model<E: Dim, In: Dim, Out: Dim, K: Dim, T: Float = f32> {
    param weight: Tensor[E, Out, In; T]

    pub entry forward<R: Dim>(
        x: Tensor[R, K, In; T],
        experts: Tensor[R, K; i32],
    ) -> Tensor[R, K, Out; T] {
        let chosen[r, k] = cast<i64>(experts[r, k])
        return linear_experts(x, weight, chosen)
    }
}
"""

GENERICS: dict[str, int | str] = {"E": 8, "In": 64, "Out": 48, "K": 2}


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path, torch.Tensor]:
    source = tmp_path / "moe.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    weight = torch.randn(8, 48, 64, generator=torch.Generator().manual_seed(0)) * 0.1
    path = tmp_path / "model.safetensors"
    save_file({"weight": weight}, str(path))
    return source, path, weight


def _inputs(rows: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, 2, 64, generator=generator)
    # Repeats, experts nobody chose, and rows out of expert order.
    experts = torch.randint(0, 8, (rows, 2), generator=generator, dtype=torch.int32)
    return x.to(device), experts.to(device)


def _expected(x: torch.Tensor, weight: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
    chosen = cast("list[list[int]]", experts.tolist())  # pyright: ignore[reportUnknownMemberType]
    return torch.stack(
        [
            torch.stack([weight[e].double() @ x[r, k].double() for k, e in enumerate(row)])
            for r, row in enumerate(chosen)
        ]
    )


def test_every_path_computes_the_chosen_experts(files: tuple[Path, Path, torch.Tensor]) -> None:
    source, path, weight = files
    x, experts = _inputs(5)
    expected = _expected(x, weight, experts).float()
    for numerics, compiled in (("exact", False), ("fast", True)):
        model = load(
            source,
            generics=GENERICS,
            std_root=STDLIB,
            weights=path,
            numerics=numerics,
            compile=compiled,
        )
        got = model.run_entry("forward", [x, experts])
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_jax_runs_the_body(files: tuple[Path, Path, torch.Tensor]) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load_model

    source, path, weight = files
    x, experts = _inputs(5)
    model = load_model(source, generics=GENERICS, weights=path, std_root=STDLIB)
    got = np.asarray(model.run_entry("forward", [x.numpy(), experts.numpy()]))
    np.testing.assert_allclose(got, _expected(x, weight, experts).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="the grouped matrix product needs a Hopper GPU",
)
def test_cuda_groups_the_rows_by_expert(files: tuple[Path, Path, torch.Tensor]) -> None:
    source, path, weight = files
    generics = {**GENERICS, "T": "bf16"}
    model = load(
        source,
        generics=generics,
        std_root=STDLIB,
        weights=path,
        device="cuda",
        numerics="fast",
        compile=True,
        cast_dtype=True,
    )
    reference = weight.to(torch.bfloat16).double()
    for rows in (1, 3, 40):
        x, experts = _inputs(rows, "cuda")
        x = x.to(torch.bfloat16)
        got = model.run_entry("forward", [x, experts]).double().cpu()
        expected = _expected(x.double().cpu(), reference, experts.cpu())
        torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)
    assert isinstance(model, CompiledLinnetModule)
    assert "torch._grouped_mm(" in model.generated_source("forward")
    # Captured as a CUDA graph, the sorting and the offsets stay on the device.
    x, experts = _inputs(3, "cuda")
    x = x.to(torch.bfloat16)
    runs = [model.run_entry("forward", [x, experts], compile="reduce-overhead") for _ in range(3)]
    torch.testing.assert_close(runs[2], runs[0])
    other = torch.flip(experts, [0])
    moved = model.run_entry("forward", [x, other], compile="reduce-overhead")
    torch.testing.assert_close(moved, model.run_entry("forward", [x, other]))
