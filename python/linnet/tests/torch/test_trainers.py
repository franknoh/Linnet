"""`CausalLM` inside the trainers it answers for: `transformers.Trainer`
with gradient accumulation, and TRL's `SFTTrainer` with each of its losses
and its token accuracy and entropy, over the Llama example on the CPU."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.torch import CausalLM, LinnetModule, load

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

transformers = pytest.importorskip("transformers")
trl = pytest.importorskip("trl")
datasets = pytest.importorskip("datasets")

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


def _rows() -> list[dict[str, list[int]]]:
    """Prompts of two tokens and completions of two to five, ids 1 to 9."""
    generator = torch.Generator().manual_seed(1)
    rows: list[dict[str, list[int]]] = []
    for length in [4, 7, 5, 6, 3, 7, 4, 5]:
        ids = torch.randint(1, 10, (length,), generator=generator).tolist()
        rows.append({"input_ids": ids, "completion_mask": [0, 0] + [1] * (length - 2)})
    return rows


def _args(tmp_path: Path) -> dict[str, object]:
    return {
        "output_dir": str(tmp_path / "out"),
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_steps": 2,
        "learning_rate": 1e-2,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "use_cpu": True,
        "dataloader_pin_memory": False,
    }


def _weights(model: LinnetModule) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def test_transformers_trainer(model: LinnetModule, tmp_path: Path) -> None:
    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in batch)
        ids = torch.zeros(len(batch), width, dtype=torch.long)
        labels = torch.full((len(batch), width), -100)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for i, row in enumerate(batch):
            n = len(row["input_ids"])
            ids[i, :n] = torch.tensor(row["input_ids"])
            learned = torch.tensor(row["completion_mask"]).bool()
            labels[i, :n] = torch.where(learned, ids[i, :n], -100)
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    before = _weights(model)
    trainer = transformers.Trainer(
        model=CausalLM(model, bucket=8),
        args=transformers.TrainingArguments(**_args(tmp_path), remove_unused_columns=False),
        train_dataset=_rows(),
        data_collator=collate,
    )
    trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    assert len(losses) == 2 and all(0 < loss < 10 for loss in losses)
    after = _weights(model)
    assert any(not torch.equal(before[name], after[name]) for name in before)


def _tokenizer() -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore[import-untyped]

    vocab = {f"t{i}": i for i in range(11)}
    words = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    words.pre_tokenizer = pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=words, pad_token="t0", eos_token="t10", unk_token="t0"
    )


@pytest.mark.parametrize("loss_type", ["chunked_nll", "nll"])
def test_trl_sft_trainer(model: LinnetModule, tmp_path: Path, loss_type: str) -> None:
    """TRL's default loss reads `base_model`'s states and the output head;
    `nll` reads the logits."""
    before = _weights(model)
    trainer = trl.SFTTrainer(
        model=CausalLM(model, bucket=8, logits=loss_type == "nll", name="llama-example"),
        args=trl.SFTConfig(**_args(tmp_path), bf16=False, max_length=8, loss_type=loss_type),
        train_dataset=datasets.Dataset.from_list(_rows()),
        processing_class=_tokenizer(),
    )
    with pytest.warns(UserWarning, match="checkpointing"):
        trainer.train()
    assert "trl" in trainer.model.model_tags
    logged = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert len(logged) == 2
    assert all(0 <= entry["mean_token_accuracy"] <= 1 and entry["entropy"] > 0 for entry in logged)
    after = _weights(model)
    assert any(not torch.equal(before[name], after[name]) for name in before)
