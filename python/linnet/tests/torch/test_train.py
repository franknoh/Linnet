"""`linnet.train`: examples packed into fixed batches with their targets and
masks, gradient accumulation following the mean over every batch of a
step, and a training loop that lowers the loss and saves what it trained."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import itertools
import os
import socket
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.torch import LinnetModule, load
from linnet.train import Example, cosine_schedule, pack, train

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


def test_pack_lays_out_targets_and_masks() -> None:
    batches = list(pack([Example.prompted([1, 2], [3, 4]), Example([5, 6, 7])], tokens=8))
    assert len(batches) == 1
    batch = batches[0]
    assert batch.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 0]
    assert batch.positions.tolist() == [0, 1, 2, 3, 0, 1, 2, 0]
    assert batch.segments.tolist() == [0, 0, 0, 0, 1, 1, 1, 2]
    assert batch.targets.tolist() == [2, 3, 4, 0, 6, 7, 0, 0]
    # The prompt is read, not learned; a sequence's last position predicts
    # nothing; padding counts for nothing.
    assert batch.mask.tolist() == [0, 1, 1, 0, 1, 1, 0, 0]
    assert batch.count == 4 and batch.sequences == 2


def test_pack_starts_a_batch_rather_than_split_a_sequence() -> None:
    examples = [Example([1] * 5), Example([2] * 5), Example([3] * 12)]
    batches = list(pack(examples, tokens=8))
    assert [b.sequences for b in batches] == [1, 1, 1]
    assert all(b.tokens.numel() == 8 for b in batches)
    assert batches[2].tokens.tolist() == [3] * 8  # cut to fit


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    skeleton = load(LLAMA, generics=GENERICS, std_root=STDLIB)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        name.removeprefix("root."): torch.randn(parameter.shape, generator=generator) * 0.3
        for name, parameter in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    directory = tmp_path / "weights"
    directory.mkdir()
    save_file(tensors, str(directory / "model.safetensors"))
    return directory


def _load(weights: Path) -> LinnetModule:
    return load(
        LLAMA,
        generics=GENERICS,
        std_root=STDLIB,
        weights=weights,
        compile=True,
        trainable=True,
    )


@pytest.fixture
def model(weights: Path) -> LinnetModule:
    return _load(weights)


def _examples() -> list[Example]:
    generator = torch.Generator().manual_seed(1)
    lengths = [5, 6, 4, 7]
    return [
        Example.prompted(
            torch.randint(0, 11, (2,), generator=generator).tolist(),
            torch.randint(0, 11, (n - 2,), generator=generator).tolist(),
        )
        for n in lengths
    ]


def _gradients(model: LinnetModule) -> dict[str, torch.Tensor]:
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    return grads


def test_accumulated_batches_follow_the_mean_over_all_of_them(model: LinnetModule) -> None:
    examples = _examples()
    first, second = next(pack(examples[:2], tokens=12)), next(pack(examples[2:], tokens=12))
    together = next(pack(examples, tokens=24))
    count = first.count + second.count
    assert together.count == count

    for batch in (first, second):
        model.run_entry("loss_packed", batch.inputs(count)).backward()
    accumulated = _gradients(model)
    model.run_entry("loss_packed", together.inputs(count)).backward()
    whole = _gradients(model)
    assert accumulated.keys() == whole.keys()
    for name, grad in accumulated.items():
        torch.testing.assert_close(grad, whole[name], atol=1e-5, rtol=1e-4)


def test_training_lowers_the_loss_and_saves(model: LinnetModule, tmp_path: Path) -> None:
    batch = next(pack(_examples(), tokens=24))
    trained = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trained, lr=3e-2)
    history = train(
        model,
        itertools.repeat(batch),
        optimizer=optimizer,
        steps=12,
        accumulate=2,
        schedule=cosine_schedule(optimizer, warmup=2, total=12),
        save_to=tmp_path / "run",
    )
    assert len(history.steps) == 12
    assert history.losses[-1] < history.losses[0] * 0.8
    assert all(step.grad_norm is not None and step.tokens == 48 for step in history.steps)
    assert (tmp_path / "run" / "model.safetensors").exists()


def test_adapters_alone_are_saved(model: LinnetModule, tmp_path: Path) -> None:
    model.add_lora("layers.*.attention.*_proj.weight", rank=2, alpha=4)
    trained = [p for p in model.parameters() if p.requires_grad]
    train(
        model,
        pack(_examples(), tokens=24),
        optimizer=torch.optim.AdamW(trained, lr=1e-2),
        save_to=tmp_path / "run",
    )
    assert (tmp_path / "run" / "adapters.safetensors").exists()
    assert not (tmp_path / "run" / "model.safetensors").exists()


def _data_parallel_rank(rank: int, world: int, port: int, weights: str, out: str) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        model = _load(Path(weights))
        mine = next(pack(_examples()[2 * rank : 2 * rank + 2], tokens=12))
        trained = [p for p in model.parameters() if p.requires_grad]
        train(model, itertools.repeat(mine), optimizer=torch.optim.SGD(trained, lr=0.1), steps=3)
        parameters = {n: p.detach().clone() for n, p in model.named_parameters()}
        torch.save(parameters, Path(out) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_data_parallel_steps_match_one_process(weights: Path, tmp_path: Path) -> None:
    """Two processes, each on its own batch, take the steps one process
    takes accumulating both batches."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.multiprocessing.spawn(
        _data_parallel_rank, args=(2, port, str(weights), str(tmp_path)), nprocs=2
    )
    ranks = [torch.load(tmp_path / f"rank{rank}.pt") for rank in range(2)]

    model = _load(weights)
    examples = _examples()
    both = [next(pack(examples[:2], tokens=12)), next(pack(examples[2:], tokens=12))]
    trained = [p for p in model.parameters() if p.requires_grad]
    train(
        model,
        itertools.cycle(both),
        optimizer=torch.optim.SGD(trained, lr=0.1),
        steps=3,
        accumulate=2,
    )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(ranks[0][name], ranks[1][name])
        torch.testing.assert_close(ranks[0][name], parameter.detach(), atol=1e-5, rtol=1e-4)
