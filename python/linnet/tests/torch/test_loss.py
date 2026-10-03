"""`std.nn.loss`: log-probabilities, cross-entropy and entropy over tokens,
and the output-head forms that PyTorch computes a block of rows at a time.
Values and gradients agree with PyTorch's own, whatever the blocks."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional

import linnet.torch.loss as chunked
from linnet.torch import load_function

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.losses

use std.nn.loss::{
    cross_entropy,
    entropy,
    linear_cross_entropy,
    linear_token_log_probs,
    log_softmax,
    token_log_probs,
}

pub entry head_loss<N: Dim, H: Dim, V: Dim, T: Float>(
    hidden: Tensor[N, H; T],
    weight: Tensor[V, H; T],
    targets: Tensor[N; i64],
    weights: Tensor[N; f32],
) -> f32 {
    return linear_cross_entropy(hidden, weight, targets, weights)
}

pub entry head_log_probs<N: Dim, H: Dim, V: Dim, T: Float>(
    hidden: Tensor[N, H; T],
    weight: Tensor[V, H; T],
    targets: Tensor[N; i64],
) -> Tensor[N; f32] {
    return linear_token_log_probs(hidden, weight, targets)
}

pub entry loss<N: Dim, V: Dim, T: Float>(
    logits: Tensor[N, V; T],
    targets: Tensor[N; i64],
    weights: Tensor[N; f32],
) -> f32 {
    return cross_entropy(logits, targets, weights)
}

pub entry picked<N: Dim, V: Dim, T: Float>(
    logits: Tensor[N, V; T],
    targets: Tensor[N; i64],
) -> Tensor[N; f32] {
    return token_log_probs(logits, targets)
}

pub entry spread<N: Dim, V: Dim, T: Float>(logits: Tensor[N, V; T]) -> Tensor[N; f32] {
    return entropy(logits)
}

pub entry normalized<N: Dim, V: Dim, T: Float>(logits: Tensor[N, V; T]) -> Tensor[N, V; f32] {
    return log_softmax(logits)
}
"""

N, H, V = 7, 8, 11


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "losses.linnet"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(N, H, generator=generator)
    weight = torch.randn(V, H, generator=generator) * 0.5
    targets = torch.randint(0, V, (N,), generator=generator)
    weights = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]) / 5
    return hidden, weight, targets, weights


@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize("rows", [2, 1000])
def test_the_head_loss_matches_pytorch(
    source: Path, compile: bool, rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A block of `rows` rows at a time: several blocks, or one."""
    monkeypatch.setattr(chunked, "BLOCK_BYTES", 4 * V * rows)
    hidden, weight, targets, weights = _inputs()
    head_loss = load_function(source, "head_loss", std_root=STDLIB, compile=compile)

    ours = [hidden.clone().requires_grad_(), weight.clone().requires_grad_()]
    got = head_loss(ours[0], ours[1], targets, weights)
    got.backward()

    theirs = [hidden.clone().requires_grad_(), weight.clone().requires_grad_()]
    nll = functional.cross_entropy(theirs[0] @ theirs[1].T, targets, reduction="none")
    want = (weights * nll).sum()
    want.backward()

    torch.testing.assert_close(got, want)
    for mine, reference in zip(ours, theirs, strict=True):
        assert mine.grad is not None and reference.grad is not None
        torch.testing.assert_close(mine.grad, reference.grad)


@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize("rows", [3, 1000])
def test_the_head_log_probs_match_pytorch(
    source: Path, compile: bool, rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chunked, "BLOCK_BYTES", 4 * V * rows)
    hidden, weight, targets, _ = _inputs()
    head_log_probs = load_function(source, "head_log_probs", std_root=STDLIB, compile=compile)
    scale = torch.linspace(-1.0, 2.0, N)

    ours = [hidden.clone().requires_grad_(), weight.clone().requires_grad_()]
    got = head_log_probs(ours[0], ours[1], targets)
    (got * scale).sum().backward()

    theirs = [hidden.clone().requires_grad_(), weight.clone().requires_grad_()]
    logp = functional.log_softmax(theirs[0] @ theirs[1].T, dim=-1)
    want = logp.gather(1, targets[:, None]).squeeze(1)
    (want * scale).sum().backward()

    torch.testing.assert_close(got, want)
    for mine, reference in zip(ours, theirs, strict=True):
        assert mine.grad is not None and reference.grad is not None
        torch.testing.assert_close(mine.grad, reference.grad)


def test_without_gradients_the_head_loss_only_sums(source: Path) -> None:
    hidden, weight, targets, weights = _inputs()
    head_loss = load_function(source, "head_loss", std_root=STDLIB, compile=True)
    with torch.no_grad():
        got = head_loss(hidden, weight, targets, weights)
    want = (weights * functional.cross_entropy(hidden @ weight.T, targets, reduction="none")).sum()
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("compile", [False, True])
def test_the_logit_forms_match_pytorch(source: Path, compile: bool) -> None:
    hidden, weight, targets, weights = _inputs()
    logits = (hidden @ weight.T).to(torch.bfloat16)
    reference = functional.log_softmax(logits.float(), dim=-1)

    def entry(name: str) -> object:
        return load_function(source, name, std_root=STDLIB, compile=compile)

    torch.testing.assert_close(entry("normalized")(logits), reference)  # type: ignore[operator]
    picked = reference.gather(1, targets[:, None]).squeeze(1)
    torch.testing.assert_close(entry("picked")(logits, targets), picked)  # type: ignore[operator]
    torch.testing.assert_close(entry("loss")(logits, targets, weights), -(weights * picked).sum())  # type: ignore[operator]
    spread = -(reference.exp() * reference).sum(-1)
    torch.testing.assert_close(entry("spread")(logits), spread)  # type: ignore[operator]
