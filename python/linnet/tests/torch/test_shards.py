"""A model that says how it splits (a `Shards` generic,
`std.nn.parallel::all_reduce`, and `all_gather`) loaded with
`tensor_parallel=`: each process binds its own part of every weight the
checkpoint holds `Shards` times over, and the result is the whole model's,
on every process."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file  # type: ignore[import-untyped]
from torch.multiprocessing.spawn import spawn

from linnet.torch import load
from linnet.weights import write_bindings

if TYPE_CHECKING:
    from linnet.train import Batch

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.shards

use std.nn.activations::{silu}
use std.nn.linear::{Linear}
use std.nn.parallel::{all_gather, all_reduce}

pub block Model<H: Dim, Inner: Dim, Vocab: Dim, T: Float = f32, Shards: Dim = 1>
where
    Shards > 0,
    Inner % Shards == 0,
    Vocab % Shards == 0
{
    sub up: Linear<H, Inner / Shards, T>
    sub down: Linear<Inner / Shards, H, T>
    sub head: Linear<H, Vocab / Shards, T>

    pub entry forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        return x + all_reduce(down.forward(silu(up.forward(x))))
    }

    // Each shard's slice of the vocabulary, gathered.
    pub entry logits<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, Vocab; T] {
        return all_gather<B, Vocab / Shards, Shards, T>(head.forward(x))
    }
}
"""

GENERICS: dict[str, int | str] = {"H": 8, "Inner": 12, "Vocab": 10}


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


def _shard(rank: int, port: int, source: str, weights: str, out: str) -> None:
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (2,))
        model = load(
            source,
            generics=GENERICS,
            std_root=STDLIB,
            weights=weights,
            compile=True,
            tensor_parallel=mesh,
        )
        x = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 10
        y = model.run_entry("forward", [x])
        logits = model.run_entry("logits", [x])
        shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
        torch.save({"y": y, "logits": logits, "shapes": shapes}, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_each_process_holds_its_part_and_gets_the_whole_result(tmp_path: Path) -> None:
    source = tmp_path / "shards.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "up.weight": torch.randn(12, 8, generator=generator),
        "up.bias": torch.randn(12, generator=generator),
        "down.weight": torch.randn(8, 12, generator=generator),
        "head.weight": torch.randn(10, 8, generator=generator),
    }
    weights = tmp_path / "model.safetensors"
    save_file(tensors, str(weights))
    whole = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 10
    expected = whole.run_entry("forward", [x])
    expected_logits = whole.run_entry("logits", [x])
    # One shard's gather is its slice, the whole: the interpreter agrees.
    interpreted = load(source, generics=GENERICS, std_root=STDLIB, weights=weights)
    torch.testing.assert_close(interpreted.run_entry("logits", [x]), expected_logits)
    torch.testing.assert_close(expected_logits, x @ tensors["head.weight"].T)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = str(tmp_path / "result")
    spawn(_shard, args=(port, str(source), str(weights), out), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(f"{out}.{rank}")
        torch.testing.assert_close(result["y"], expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result["logits"], expected_logits, atol=1e-5, rtol=1e-5)
        # Half of `up` by its outputs (with its bias), half of `down` by its
        # inputs, half of `head` by its outputs.
        shapes = result["shapes"]
        assert shapes["root.up.weight"] == (6, 8) and shapes["root.up.bias"] == (6,)
        assert shapes["root.down.weight"] == (8, 6)
        assert shapes["root.head.weight"] == (5, 8)


TRAINING = """\
module tests.shards_training

use std.nn.activations::{silu}
use std.nn.embedding::{Embedding}
use std.nn.linear::{Linear}
use std.nn.loss::{split_cross_entropy}
use std.nn.norm::{RmsNorm}
use std.nn.parallel::{all_reduce, shared}

pub block Layer<H: Dim, Inner: Dim, T: Float, Shards: Dim>
where
    Shards > 0,
    Inner % Shards == 0
{
    sub norm: RmsNorm<H, T>
    sub up: Linear<H, Inner / Shards, T>
    sub down: Linear<Inner / Shards, H, T>

    pub fn forward<P: Dim>(x: Tensor[P, H; T]) -> Tensor[P, H; T] {
        return x + all_reduce(down.forward(silu(up.forward(shared(norm.forward(x))))))
    }
}

pub block Model<Vocab: Dim, H: Dim, Inner: Dim, Layers: Dim, T: Float = f32, Shards: Dim = 1>
where
    Shards > 0,
    Inner % Shards == 0,
    Vocab % Shards == 0
{
    sub embedding: Embedding<Vocab, H, T>
    sub layers: [Layer<H, Inner, T, Shards>; Layers]
    sub norm: RmsNorm<H, T>
    sub head: Linear<H, Vocab / Shards, T>

    pub entry loss_packed<P: Dim>(
        tokens: Tensor[P; i32],
        positions: Tensor[P; i32],
        segments: Tensor[P; i32],
        targets: Tensor[P; i64],
        weights: Tensor[P; f32],
    ) -> f32
    where P > 0 {
        var x = embedding.forward(tokens)
        static for layer in layers {
            x = layer.forward(x)
        }
        let hidden = shared(norm.forward(x))
        return split_cross_entropy<P, H, Vocab / Shards, Shards, T>(
            hidden,
            head.weight,
            targets,
            weights,
        )
    }
}
"""
TRAINING_GENERICS: dict[str, int | str] = {"Vocab": 10, "H": 8, "Inner": 12, "Layers": 2}


def training_files(directory: Path) -> tuple[Path, Path]:
    """The training model's source and a checkpoint of the whole model."""
    source = directory / "training.linnet"
    source.write_text(TRAINING, encoding="utf-8")
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "embedding.weight": torch.randn(10, 8, generator=generator),
        "norm.weight": 1 + 0.1 * torch.randn(8, generator=generator),
        "head.weight": torch.randn(10, 8, generator=generator) * 0.5,
    }
    for i in range(2):
        tensors[f"layers.{i}.norm.weight"] = 1 + 0.1 * torch.randn(8, generator=generator)
        tensors[f"layers.{i}.up.weight"] = torch.randn(12, 8, generator=generator) * 0.4
        tensors[f"layers.{i}.down.weight"] = torch.randn(8, 12, generator=generator) * 0.4
    weights = directory / "training.safetensors"
    save_file(tensors, str(weights))
    return source, weights


def training_batches() -> list[Batch]:
    """Two packed batches of 16 positions."""
    from linnet.train import Example, pack

    generator = torch.Generator().manual_seed(1)
    examples = [
        Example.prompted(
            torch.randint(0, 10, (2,), generator=generator).tolist(),
            torch.randint(0, 10, (n,), generator=generator).tolist(),
        )
        for n in (5, 6, 3, 7, 4, 5)
    ]
    return list(pack(examples, tokens=16))[:2]


def _train_split(
    rank: int, port: int, source: str, weights: str, bindings: str | None, out: str
) -> None:
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from linnet.train import train

        mesh = init_device_mesh("cpu", (2,))
        model = load(
            source,
            generics=TRAINING_GENERICS,
            std_root=STDLIB,
            weights=weights,
            bindings=bindings,
            compile=True,
            trainable=True,
            tensor_parallel=mesh,
        )
        batch = training_batches()[0]
        loss = model.run_entry("loss_packed", batch.inputs(batch.count, "cpu"))
        loss.backward()
        grads = {
            name.removeprefix("root."): p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        model.zero_grad()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
        history = train(model, training_batches(), optimizer=optimizer, clip=0.5)
        trained = {name.removeprefix("root."): p.detach() for name, p in model.named_parameters()}
        torch.save(
            {
                "loss": loss.detach(),
                "grads": grads,
                "parts": dict(model.shard_parts),
                "losses": history.losses,
                "norms": [step.grad_norm for step in history.steps],
                "trained": trained,
            },
            f"{out}.{rank}",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tied", [False, True])
def test_split_training_matches_one_process(tmp_path: Path, tied: bool) -> None:
    """Two processes, each with half of every split weight, train as the
    whole model does: the same loss, each its part of every split weight's
    gradient and the whole of every other, and `train`'s steps -- the
    clipped norm summed across the parts -- the same. A head tied to the
    embedding is its part of the whole embedding, which trains as one."""
    from linnet.train import train

    source, weights = training_files(tmp_path)
    bindings = None
    if tied:
        bindings = write_bindings(tmp_path / "tied.json", {"head.weight": "embedding.weight"})
    whole = load(
        source,
        generics=TRAINING_GENERICS,
        std_root=STDLIB,
        weights=weights,
        bindings=bindings,
        compile=True,
        trainable=True,
    )
    batch = training_batches()[0]
    loss = whole.run_entry("loss_packed", batch.inputs(batch.count, "cpu"))
    loss.backward()
    expected = {
        name.removeprefix("root."): p.grad.clone()
        for name, p in whole.named_parameters()
        if p.grad is not None
    }
    whole.zero_grad()
    optimizer = torch.optim.SGD([p for p in whole.parameters() if p.requires_grad], lr=0.5)
    history = train(whole, training_batches(), optimizer=optimizer, clip=0.5)
    trained = {name.removeprefix("root."): p.detach() for name, p in whole.named_parameters()}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = str(tmp_path / "split")
    tie = str(bindings) if bindings is not None else None
    spawn(_train_split, args=(port, str(source), str(weights), tie, out), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(f"{out}.{rank}")
        torch.testing.assert_close(result["loss"], loss.detach(), atol=1e-5, rtol=1e-5)
        parts = result["parts"]
        assert set(parts) >= {"layers.0.up.weight", "layers.1.down.weight"}
        assert ("head.weight" in parts) != tied
        assert set(result["grads"]) == set(expected)
        for name, want in expected.items():
            got = result["grads"][name]
            if name in parts:
                axis, extent = parts[name]
                want = want.narrow(axis, rank * extent, extent)
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
        assert result["losses"] == pytest.approx(history.losses, rel=1e-5)
        assert result["norms"] == pytest.approx([s.grad_norm for s in history.steps], rel=1e-5)
        for name in expected:  # what trains; an absent bias stays as it is
            want, got = trained[name], result["trained"][name]
            if name in parts:
                axis, extent = parts[name]
                want = want.narrow(axis, rank * extent, extent)
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
