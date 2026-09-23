"""Training through PyTorch autograd: a Linnet model's parameters receive
gradients and an optimizer fits it, interpreted or as generated source."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from linnet.torch import load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module regression

use std.nn.activations::{silu}
use std.nn.linear::{Linear}

pub block Mlp<In: Dim, Hidden: Dim, Out: Dim> {
    sub up: Linear<In, Hidden, f32>
    sub down: Linear<Hidden, Out, f32>

    pub entry forward<B: Dim>(x: Tensor[B, In; f32]) -> Tensor[B, Out; f32] {
        return down.forward(silu(up.forward(x)))
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
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("compile", [False, True])
def test_sgd_fits_a_linnet_mlp(tmp_path: Path, compile: bool) -> None:
    source = tmp_path / "mlp.linnet"
    source.write_text(SOURCE)
    model = load(
        source,
        generics={"In": 3, "Hidden": 16, "Out": 1},
        std_root=STDLIB,
        numerics="equivalent",
        compile=compile,
        trainable=True,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.3)
    assert all(p.requires_grad for p in model.parameters())

    target_weight = torch.tensor([[1.0, -2.0, 0.5]])
    x = torch.randn(256, 3)
    y = x @ target_weight.T + 0.1
    optimizer: torch.optim.Optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    losses: list[float] = []
    for _ in range(200):
        optimizer.zero_grad()
        output: torch.Tensor = model(x)
        loss: torch.Tensor = torch.nn.functional.mse_loss(output, y)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.05, (losses[0], losses[-1])
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
