"""Checkpoints (`save_checkpoint`, `load_checkpoint`, `train(checkpoint=...)`):
a run stopped after two steps and resumed ends where an unbroken run ends,
optimizer state and schedule included, for a whole model and for one split
across processes; a LoRA run keeps only its adapters."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file  # pyright: ignore[reportUnknownVariableType]

from linnet.torch import LinnetModule, fully_shard
from linnet.train import Batch, Example, cosine_schedule, pack, train
from tests.helpers import random_llama, trainable_llama


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    directory = tmp_path / "weights"
    directory.mkdir()
    random_llama(directory / "model.safetensors")
    return directory


def _batches(seed: int = 1) -> list[Batch]:
    generator = torch.Generator().manual_seed(seed)
    examples = [
        Example.prompted(
            torch.randint(0, 11, (2,), generator=generator).tolist(),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            torch.randint(0, 11, (n - 2,), generator=generator).tolist(),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        )
        for n in [5, 6, 4, 7, 5, 6, 7, 4]
    ]
    return [next(pack(examples[i : i + 2], tokens=12)) for i in range(0, 8, 2)]


def _run(model: LinnetModule, steps: int, checkpoint: Path | None, seed: int = 1) -> list[int]:
    trained = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trained, lr=3e-2)
    history = train(
        model,
        iter(_batches(seed)),
        optimizer=optimizer,
        steps=steps,
        schedule=cosine_schedule(optimizer, warmup=1, total=4),
        checkpoint=checkpoint,
        checkpoint_every=1 if checkpoint is not None else None,
    )
    return [step.step for step in history.steps]


def _weights(model: LinnetModule) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_a_resumed_run_ends_where_an_unbroken_one_does(weights: Path, tmp_path: Path) -> None:
    unbroken = trainable_llama(weights)
    assert _run(unbroken, 4, None) == [1, 2, 3, 4]

    assert _run(trainable_llama(weights), 2, tmp_path / "run") == [1, 2]
    resumed = trainable_llama(weights)
    assert _run(resumed, 4, tmp_path / "run") == [3, 4]
    for name, value in _weights(unbroken).items():
        torch.testing.assert_close(_weights(resumed)[name], value)
    # Every step wrote one; the two latest are kept.
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == [
        "step-00000003",
        "step-00000004",
    ]


def test_a_lora_run_keeps_only_its_adapters(weights: Path, tmp_path: Path) -> None:
    from torch.distributed.checkpoint import (
        FileSystemReader,  # pyright: ignore[reportPrivateImportUsage]
    )

    model = trainable_llama(weights)
    model.add_lora("layers.*.attention.*_proj.weight", rank=2, alpha=4)
    _run(model, 1, tmp_path / "run")
    metadata = FileSystemReader(str(tmp_path / "run" / "step-00000001")).read_metadata()
    saved = [key for key in metadata.state_dict_metadata if key.startswith("model.")]
    assert saved and all(key.endswith(("lora_a", "lora_b")) for key in saved)


def _sharded_rank(rank: int, world: int, port: int, weights: str, out: str) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:

        def sharded() -> LinnetModule:
            model = trainable_llama(Path(weights))
            fully_shard(model)
            return model

        unbroken = sharded()
        _run(unbroken, 4, None, seed=1 + rank)
        unbroken.save_weights(Path(out) / "unbroken.safetensors", names="linnet")

        _run(sharded(), 2, Path(out) / "run", seed=1 + rank)
        resumed = sharded()
        assert _run(resumed, 4, Path(out) / "run", seed=1 + rank) == [3, 4]
        resumed.save_weights(Path(out) / "resumed.safetensors", names="linnet")
    finally:
        dist.destroy_process_group()


def test_a_sharded_run_resumes(weights: Path, tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.multiprocessing.spawn(  # pyright: ignore[reportPrivateImportUsage, reportUnknownMemberType]
        _sharded_rank, args=(2, port, str(weights), str(tmp_path)), nprocs=2
    )
    unbroken = load_file(str(tmp_path / "unbroken.safetensors"))
    resumed = load_file(str(tmp_path / "resumed.safetensors"))
    assert unbroken.keys() == resumed.keys()
    for name, value in unbroken.items():
        torch.testing.assert_close(resumed[name], value)
