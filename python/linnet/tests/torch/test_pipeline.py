"""Pipeline parallelism: an entry's generated source split into stages by its
data flow computes what the whole entry computes, and the stages run as a
pipeline over processes train as one process does."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.compiler import bind_arguments, run_compiler
from linnet.torch import LinnetModule, load
from linnet.torch.stages import split
from linnet.train import Example, pack

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
UNITS = ["embedding", "layers.0", "layers.1", "norm", "lm_head"]


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


def _model(weights: Path) -> LinnetModule:
    return load(
        LLAMA, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True, trainable=True
    )


def _packs() -> list[list[torch.Tensor]]:
    """Two packs of 12 positions: the micro-batches of one step."""
    generator = torch.Generator().manual_seed(1)
    examples = [
        Example.prompted(
            torch.randint(0, 11, (2,), generator=generator).tolist(),
            torch.randint(0, 11, (n - 2,), generator=generator).tolist(),
        )
        for n in [5, 6, 4, 7]
    ]
    packs = [next(pack(examples[:2], tokens=12)), next(pack(examples[2:], tokens=12))]
    count = sum(p.count for p in packs)
    return [p.inputs(count) for p in packs]


def _stage(path: str, starts: list[str]) -> int:
    unit = next(u for u in UNITS if path == u or path.startswith(u + "."))
    return sum(1 for start in starts if UNITS.index(start) <= UNITS.index(unit))


@pytest.mark.parametrize("starts", [["layers.1"], ["layers.0", "norm"]])
def test_split_stages_compute_what_main_computes(weights: Path, starts: list[str]) -> None:
    source = run_compiler(
        "torch",
        "--std",
        str(STDLIB),
        "--entry",
        "loss_packed",
        *bind_arguments({**GENERICS, "P": 12}),
        str(LLAMA),
    )
    pieces = split(source, lambda path: _stage(path, starts), len(starts) + 1)
    whole: dict[str, Any] = {}
    exec(compile(source, "<main>", "exec"), whole)
    staged: dict[str, Any] = {}
    exec(compile(pieces.source, "<stages>", "exec"), staged)

    model = _model(weights)
    inputs = _packs()[0]
    device = torch.device("cpu")
    constants = list(whole["constants"](device)) if pieces.constants else []
    by_name = dict(zip(pieces.constants, constants, strict=True))

    def parameters() -> list[torch.Tensor]:
        return [
            model.get_parameter("root." + path).detach().clone().requires_grad_(True)
            for path in pieces.parameters
        ]

    reference = parameters()
    (expected,) = whole["main"](*inputs, *reference, *constants)
    expected.backward()

    mine = parameters()
    received: Any = ()
    named = dict(zip(pieces.inputs, inputs, strict=True))
    for stage in pieces.stages:
        received = staged[stage.name](
            *received,
            *[mine[i] for i in stage.parameters],
            *[by_name[name] for name in stage.constants],
            _device=device,
            **{name: named[name] for name in stage.inputs},
        )
    (loss,) = received
    loss.backward()

    # Only hidden states cross: masks and positions are computed per stage.
    for stage in pieces.stages[1:]:
        assert len(stage.receives) == 1
    torch.testing.assert_close(loss, expected)
    for got, want in zip(mine, reference, strict=True):
        assert got.grad is not None and want.grad is not None
        torch.testing.assert_close(got.grad, want.grad, atol=1e-6, rtol=1e-5)


def _rank(rank: int, world: int, port: int, weights: str, out: str, schedule: str) -> None:
    import torch.distributed as dist

    from linnet.torch import pipeline

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        pipe = pipeline(
            LLAMA,
            generics=GENERICS,
            std_root=STDLIB,
            weights=weights,
            entry="loss_packed",
            microbatches=2,
            stages=["layers.1"],
            schedule="1f1b" if schedule == "1f1b" else "gpipe",
            device="cpu",
        )
        # Each stage holds its own blocks' weights and nothing of the rest.
        held = {name for name, _ in pipe.named_parameters()}
        assert all(_stage(name, ["layers.1"]) == rank for name in held) and held
        inputs = [torch.cat(parts) for parts in zip(*_packs(), strict=True)]
        loss = pipe.step(*inputs)
        result: dict[str, torch.Tensor] = {
            name: parameter.grad.detach().clone()
            for name, parameter in pipe.named_parameters()
            if parameter.grad is not None
        }
        # Every stage takes part in a forward pass; the last returns it.
        values = pipe.run(*inputs)
        if loss is not None:
            result["loss"] = loss
            result["run"] = values
        torch.save(result, Path(out) / f"stage{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("schedule", ["1f1b", "gpipe"])
def test_a_pipeline_over_two_processes_matches_one(
    weights: Path, tmp_path: Path, schedule: str
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.multiprocessing.spawn(
        _rank, args=(2, port, str(weights), str(tmp_path), schedule), nprocs=2
    )
    staged: dict[str, torch.Tensor] = {}
    for rank in range(2):
        staged.update(torch.load(tmp_path / f"stage{rank}.pt"))

    model = _model(weights)
    losses = [model.run_entry("loss_packed", inputs) for inputs in _packs()]
    total = torch.stack(losses).sum()
    total.backward()
    torch.testing.assert_close(staged.pop("loss"), total.detach())
    torch.testing.assert_close(staged.pop("run"), torch.stack(losses).detach())
    expected = {
        name.removeprefix("root."): parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert set(staged) == set(expected)
    for name, grad in staged.items():
        torch.testing.assert_close(grad, expected[name], atol=1e-6, rtol=1e-5)


SPLIT = """\
module tests.split_pipeline

use std.nn.activations::{silu}
use std.nn.linear::{Linear}
use std.nn.parallel::{all_reduce}

pub block Layer<H: Dim, Inner: Dim, T: Float, Shards: Dim>
where
    Shards > 0,
    Inner % Shards == 0
{
    sub up: Linear<H, Inner / Shards, T>
    sub down: Linear<Inner / Shards, H, T>

    pub fn forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        return x + all_reduce(down.forward(silu(up.forward(x))))
    }
}

pub block Model<H: Dim, Inner: Dim, Layers: Dim, T: Float = f32, Shards: Dim = 1>
where
    Shards > 0,
    Inner % Shards == 0
{
    sub layers: [Layer<H, Inner, T, Shards>; Layers]

    pub entry forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        var h = x
        static for layer in layers {
            h = layer.forward(h)
        }
        return h
    }
}
"""
SPLIT_GENERICS: dict[str, int | str] = {"H": 8, "Inner": 12, "Layers": 2}


def _split_rank(rank: int, port: int, source: str, weights: str, out: str) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from linnet.torch import pipeline

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=4)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("pp", "tp"))
        pipe = pipeline(
            source,
            generics=SPLIT_GENERICS,
            std_root=STDLIB,
            weights=weights,
            entry="forward",
            microbatches=2,
            stages=["layers.1"],
            device="cpu",
            group=mesh["pp"].get_group(),
            tensor_parallel=mesh["tp"],
            trainable=False,
        )
        shapes = {name: tuple(p.shape) for name, p in pipe.named_parameters()}
        x = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 10
        result = pipe.run(x)
        torch.save({"shapes": shapes, "result": result}, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_a_pipeline_of_split_stages_matches_one_process(tmp_path: Path) -> None:
    source = tmp_path / "split.linnet"
    source.write_text(SPLIT, encoding="utf-8")
    generator = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(2):
        tensors[f"layers.{layer}.up.weight"] = torch.randn(12, 8, generator=generator) * 0.3
        tensors[f"layers.{layer}.up.bias"] = torch.randn(12, generator=generator) * 0.3
        tensors[f"layers.{layer}.down.weight"] = torch.randn(8, 12, generator=generator) * 0.3
    weights = tmp_path / "model.safetensors"
    save_file(tensors, str(weights))
    whole = load(source, generics=SPLIT_GENERICS, std_root=STDLIB, weights=weights, compile=True)
    x = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 10
    expected = whole.run_entry("forward", [x])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = str(tmp_path / "rank")
    torch.multiprocessing.spawn(_split_rank, args=(port, str(source), str(weights), out), nprocs=4)
    for rank in range(4):
        saved = torch.load(f"{out}.{rank}")
        stage = rank // 2
        # Each process holds its stage's layer, a half of each split weight.
        assert all(name.startswith(f"layers.{stage}.") for name in saved["shapes"])
        assert saved["shapes"][f"layers.{stage}.up.weight"] == (6, 8)
        assert saved["shapes"][f"layers.{stage}.down.weight"] == (8, 6)
        if stage == 1:
            torch.testing.assert_close(saved["result"], expected)
        else:
            assert saved["result"] is None


def _other_packs() -> list[list[torch.Tensor]]:
    """Two more packs, for a second pipeline's batch."""
    generator = torch.Generator().manual_seed(2)
    examples = [
        Example.prompted(
            torch.randint(0, 11, (2,), generator=generator).tolist(),
            torch.randint(0, 11, (n - 2,), generator=generator).tolist(),
        )
        for n in [6, 5, 7, 4]
    ]
    packs = [next(pack(examples[:2], tokens=12)), next(pack(examples[2:], tokens=12))]
    count = sum(p.count for p in packs)
    return [p.inputs(count) for p in packs]


def _sharded_rank(rank: int, port: int, weights: str, out: str) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from linnet.torch import pipeline

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=4)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("pp", "dp"))
        pipe = pipeline(
            LLAMA,
            generics=GENERICS,
            std_root=STDLIB,
            weights=weights,
            entry="loss_packed",
            microbatches=2,
            stages=["layers.1"],
            device="cpu",
            group=mesh["pp"].get_group(),
            data_parallel=mesh["dp"],
        )
        # Each data-parallel process trains its pipeline on its own batch.
        packs = _packs() if mesh["dp"].get_local_rank() == 0 else _other_packs()
        inputs = [torch.cat(parts) for parts in zip(*packs, strict=True)]
        loss = pipe.step(*inputs)
        result: dict[str, Any] = {}
        for name, parameter in pipe.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue  # an optional parameter the model leaves out
            assert isinstance(parameter, DTensor) and isinstance(grad, DTensor)
            result[name] = grad.full_tensor()
        if loss is not None:
            result["loss"] = loss
        torch.save(result, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_a_pipeline_of_sharded_stages_trains_as_one_process(weights: Path, tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = str(tmp_path / "rank")
    torch.multiprocessing.spawn(_sharded_rank, args=(port, str(weights), out), nprocs=4)
    model = _model(weights)
    first = torch.stack([model.run_entry("loss_packed", i) for i in _packs()]).sum()
    second = torch.stack([model.run_entry("loss_packed", i) for i in _other_packs()]).sum()
    (first + second).backward()
    expected = {
        name.removeprefix("root."): parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    seen: set[str] = set()
    for rank in range(4):
        saved = torch.load(f"{out}.{rank}")
        loss = saved.pop("loss", None)
        if loss is not None:
            # The last stage of each pipeline: its own batch's loss.
            torch.testing.assert_close(loss, first.detach() if rank % 2 == 0 else second.detach())
        for name, grad in saved.items():
            # The gradients of both batches, summed across the pipelines.
            torch.testing.assert_close(grad, expected[name], atol=1e-6, rtol=1e-5)
            seen.add(name)
    assert seen == set(expected)
