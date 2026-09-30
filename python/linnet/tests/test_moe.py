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


ROUTED = """\
module tests.routed

use std.nn.moe::{combine_experts, linear_experts_shared}

pub block Model<E: Dim, In: Dim, Mid: Dim, K: Dim, T: Float = f32> {
    param up: Tensor[E, Mid, In; T]
    param down: Tensor[E, In, Mid; T]

    pub entry forward<R: Dim>(
        x: Tensor[R, In; T],
        experts: Tensor[R, K; i32],
        weights: Tensor[R, K; T],
    ) -> Tensor[R, In; T] {
        let chosen[r, k] = cast<i64>(experts[r, k])
        let hidden = linear_experts_shared(x, up, chosen)
        return combine_experts(hidden, down, chosen, weights)
    }
}
"""

ROUTED_GENERICS: dict[str, int | str] = {"E": 8, "In": 64, "Mid": 48, "K": 2}


@pytest.fixture
def routed(tmp_path: Path) -> tuple[Path, Path, torch.Tensor, torch.Tensor]:
    source = tmp_path / "routed.linnet"
    source.write_text(ROUTED, encoding="utf-8")
    generator = torch.Generator().manual_seed(1)
    up = torch.randn(8, 48, 64, generator=generator) * 0.1
    down = torch.randn(8, 64, 48, generator=generator) * 0.1
    path = tmp_path / "routed.safetensors"
    save_file({"up": up, "down": down}, str(path))
    return source, path, up, down


def _routed_inputs(rows: int, device: str = "cpu") -> tuple[torch.Tensor, ...]:
    """Each row's two distinct experts and their weights, as a router's top-2."""
    generator = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, 64, generator=generator)
    experts = torch.stack([torch.randperm(8, generator=generator)[:2] for _ in range(rows)]).int()
    weights = torch.rand(rows, 2, generator=generator)
    return x.to(device), experts.to(device), weights.to(device)


def _routed_expected(
    x: torch.Tensor,
    experts: torch.Tensor,
    weights: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    chosen = cast("list[list[int]]", experts.tolist())  # pyright: ignore[reportUnknownMemberType]
    out: list[torch.Tensor] = []
    for r, row in enumerate(chosen):
        total = torch.zeros(64, dtype=torch.float64)
        for k, e in enumerate(row):
            hidden = up[e].double() @ x[r].double()
            total += float(weights[r, k]) * (down[e].double() @ hidden)
        out.append(total)
    return torch.stack(out)


def test_routed_ops_compute_the_chosen_experts(
    routed: tuple[Path, Path, torch.Tensor, torch.Tensor],
) -> None:
    """The dense bodies (every expert, the chosen kept or placed) and the
    generated source give each row its own experts' products, weighed."""
    source, path, up, down = routed
    x, experts, weights = _routed_inputs(6)
    expected = _routed_expected(x, experts, weights, up, down).float()
    for numerics, compiled in (("exact", False), ("fast", True)):
        model = load(
            source,
            generics=ROUTED_GENERICS,
            std_root=STDLIB,
            weights=path,
            numerics=numerics,
            compile=compiled,
        )
        got = model.run_entry("forward", [x, experts, weights])
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def test_jax_runs_the_routed_bodies(routed: tuple[Path, Path, torch.Tensor, torch.Tensor]) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load_model

    source, path, up, down = routed
    x, experts, weights = _routed_inputs(6)
    model = load_model(source, generics=ROUTED_GENERICS, weights=path, std_root=STDLIB)
    got = np.asarray(model.run_entry("forward", [x.numpy(), experts.numpy(), weights.numpy()]))
    expected = _routed_expected(x, experts, weights, up, down).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="the grouped matrix product needs a Hopper GPU",
)
def test_cuda_routes_the_rows(routed: tuple[Path, Path, torch.Tensor, torch.Tensor]) -> None:
    source, path, up, down = routed
    model = load(
        source,
        generics={**ROUTED_GENERICS, "T": "bf16"},
        std_root=STDLIB,
        weights=path,
        device="cuda",
        numerics="fast",
        compile=True,
        cast_dtype=True,
    )
    assert isinstance(model, CompiledLinnetModule)
    for rows in (1, 7, 300):
        x, experts, weights = _routed_inputs(rows, "cuda")
        x, weights = x.to(torch.bfloat16), weights.to(torch.bfloat16)
        got = model.run_entry("forward", [x, experts, weights]).double().cpu()
        expected = _routed_expected(
            x.cpu(), experts.cpu(), weights.cpu(), up.to(torch.bfloat16), down.to(torch.bfloat16)
        )
        torch.testing.assert_close(got, expected, atol=3e-2, rtol=3e-2)
    source_text = model.generated_source("forward")
    assert "_linear_experts_shared(" in source_text and "_combine_experts(" in source_text
