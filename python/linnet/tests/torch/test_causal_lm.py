"""`CausalLM`: a transformers-style call (`input_ids`, `attention_mask`,
`labels`, `position_ids`) gives the loss transformers would, from the
model's `loss_packed` entry, for padded rows and for padding-free ones."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]
from torch.nn import functional

from linnet.torch import CausalLM, LinnetModule, load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"
LLAMA = REPO / "examples/05-llama/src/lib.linnet"
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
def model(tmp_path: Path) -> LinnetModule:
    skeleton = load(LLAMA, generics=GENERICS, std_root=STDLIB)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        name.removeprefix("root."): torch.randn(parameter.shape, generator=generator) * 0.3
        for name, parameter in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return load(
        LLAMA, generics=GENERICS, std_root=STDLIB, weights=tmp_path, compile=True, trainable=True
    )


def _reference(model: LinnetModule, sequences: list[torch.Tensor], labels: list[torch.Tensor]):
    """transformers' loss: each sequence's logits shifted against its
    labels, the mean over every labelled position of every sequence."""
    losses, count = [], 0
    for sequence, label in zip(sequences, labels, strict=True):
        logits = model.run_entry("forward", [sequence[None].to(torch.int32)])[0]
        shifted = label[1:]
        losses.append(
            functional.cross_entropy(logits[:-1], shifted, ignore_index=-100, reduction="sum")
        )
        count += int((shifted != -100).sum())
    return torch.stack(losses).sum() / count


@pytest.mark.parametrize("bucket", [1, 256])
def test_padded_rows_give_the_transformers_loss(model: LinnetModule, bucket: int) -> None:
    first = torch.tensor([3, 1, 4, 1, 5, 9])
    second = torch.tensor([2, 6, 5, 3])
    input_ids = torch.tensor([[3, 1, 4, 1, 5, 9], [2, 6, 5, 3, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    # The first row's first two tokens are a prompt; padding is never learned.
    labels = torch.tensor([[-100, -100, 4, 1, 5, 9], [2, 6, 5, 3, -100, -100]])
    loss = CausalLM(model, bucket=bucket)(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )["loss"]
    want = _reference(model, [first, second], [labels[0], labels[1, :4]])
    torch.testing.assert_close(loss, want, atol=1e-5, rtol=1e-5)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_padding_free_rows_keep_their_sequences_apart(model: LinnetModule) -> None:
    """Two sequences end to end in one row, told apart by `position_ids`."""
    input_ids = torch.tensor([[3, 1, 4, 1, 5, 2, 6, 5, 3]])
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3]])
    labels = input_ids.clone()
    loss = CausalLM(model)(input_ids=input_ids, labels=labels, position_ids=position_ids)["loss"]
    want = _reference(model, [input_ids[0, :5], input_ids[0, 5:]], [labels[0, :5], labels[0, 5:]])
    torch.testing.assert_close(loss, want, atol=1e-5, rtol=1e-5)
