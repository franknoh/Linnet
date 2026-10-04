"""DPO (`linnet.train.dpo`): the loss, pairs kept in one batch, a model that
learns which answer is preferred, and a reference model giving the same
steps as reference log-probabilities computed first."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.torch import LinnetModule, load
from linnet.train import Example, pack
from linnet.train.dpo import Pair, dpo, dpo_loss

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"
LLAMA = REPO / "examples/01-llama/src/lib.linnet"
GENERICS: dict[str, int | str] = {
    "Vocab": 11,
    "H": 8,
    "Heads": 4,
    "KvHeads": 2,
    "Inner": 16,
    "Layers": 2,
    "Batch": 1,
    "MaxSeq": 8,
    "T": "f32",
}


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    skeleton = load(LLAMA, generics=GENERICS, std_root=STDLIB)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        name.removeprefix("root."): torch.randn(parameter.shape, generator=generator) * 0.3
        for name, parameter in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return path


def _load(weights: Path) -> LinnetModule:
    return load(
        LLAMA, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True, trainable=True
    )


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


def _pairs(count: int) -> list[Pair]:
    """Prompts of two tokens; the preferred answer repeats 3, the other 7."""
    generator = torch.Generator().manual_seed(2)
    return [
        Pair(torch.randint(0, 11, (2,), generator=generator).tolist(), [3, 3, 3, 3], [7, 7, 7, 7])
        for _ in range(count)
    ]


def _train(model: LinnetModule, reference: LinnetModule | None) -> list[float]:
    trained = [p for p in model.parameters() if p.requires_grad]
    history = dpo(
        model,
        _pairs(24),
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
    model = _load(weights)
    margins = _train(model, None)
    assert margins[0] == pytest.approx(0, abs=1e-6)  # the reference is the model itself
    assert margins[-1] > max(0.3, margins[1])


def test_a_reference_model_gives_the_same_steps(weights: Path) -> None:
    computed_first = _load(weights)
    _train(computed_first, None)
    alongside = _load(weights)
    # Loaded alike, it runs the same generated code; never stepped, it stays.
    _train(alongside, _load(weights))
    theirs = dict(computed_first.named_parameters())
    for name, parameter in alongside.named_parameters():
        torch.testing.assert_close(parameter.detach(), theirs[name].detach())
