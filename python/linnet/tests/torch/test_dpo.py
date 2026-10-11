"""DPO (`linnet.train.dpo`): the loss, pairs kept in one batch, a model that
learns which answer is preferred, and a reference model giving the same
steps as reference log-probabilities computed first."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from linnet.torch import LinnetModule
from linnet.train import Example, pack
from linnet.train.dpo import dpo, dpo_loss
from tests.helpers import random_llama, tiny_pairs, trainable_llama


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    return random_llama(tmp_path / "model.safetensors")


def test_the_loss() -> None:
    chosen, rejected = torch.tensor([-1.0, -3.0]), torch.tensor([-2.0, -1.0])
    zero = torch.zeros(2)
    losses, good, bad = dpo_loss(chosen, rejected, zero, zero, beta=0.5)
    margins = [0.5 * (-1 + 2), 0.5 * (-3 + 1)]
    torch.testing.assert_close(losses, torch.tensor([math.log1p(math.exp(-m)) for m in margins]))
    torch.testing.assert_close(good - bad, torch.tensor(margins))
    smoothed, _, _ = dpo_loss(chosen, rejected, zero, zero, beta=0.5, label_smoothing=0.25)
    expected = [0.75 * math.log1p(math.exp(-m)) + 0.25 * math.log1p(math.exp(m)) for m in margins]
    torch.testing.assert_close(smoothed, torch.tensor(expected))


def test_pack_keeps_pairs_together() -> None:
    examples = [Example([1] * 3), Example([2] * 3), Example([3] * 3), Example([4] * 3)]
    batches = list(pack(examples, tokens=8, together=2))
    # Two pairs of six tokens: the second starts a batch of its own.
    assert [b.items for b in batches] == [[0, 1], [2, 3]]


def _train(model: LinnetModule, reference: LinnetModule | None) -> list[float]:
    trained = [p for p in model.parameters() if p.requires_grad]
    history = dpo(
        model,
        tiny_pairs(24),
        optimizer=torch.optim.AdamW(trained, lr=1e-2),
        reference=reference,
        steps=6,
        pairs_per_step=4,
        beta=0.5,
        tokens=24,
    )
    assert [step.step for step in history] == [1, 2, 3, 4, 5, 6]
    return [step.margin for step in history]


def test_dpo_learns_the_preference(weights: Path) -> None:
    model = trainable_llama(weights)
    margins = _train(model, None)
    assert margins[0] == pytest.approx(0, abs=1e-6)  # the reference is the model itself
    assert margins[-1] > max(0.3, margins[1])


def test_a_reference_model_gives_the_same_steps(weights: Path) -> None:
    computed_first = trainable_llama(weights)
    _train(computed_first, None)
    alongside = trainable_llama(weights)
    # Loaded alike, it runs the same generated code; never stepped, it stays.
    _train(alongside, trainable_llama(weights))
    theirs = dict(computed_first.named_parameters())
    for name, parameter in alongside.named_parameters():
        torch.testing.assert_close(parameter.detach(), theirs[name].detach())
