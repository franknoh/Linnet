"""Fully sharded data parallelism (`fully_shard`): each process keeps part
of every parameter, the generated code gathers a parameter where it is used
and drops it after, backward keeps the parts rather than the wholes, and
training takes the steps one process takes over every process's batches."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import gc
import itertools
import os
import re
import socket
import subprocess
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import load_file  # pyright: ignore[reportUnknownVariableType]

from linnet.compiler import find_compiler
from linnet.torch import fully_shard
from linnet.train import pack, train
from tests.helpers import (
    LLAMA,
    STDLIB,
    TINY_LLAMA,
    random_llama,
    tiny_examples,
    trainable_llama,
)

if TYPE_CHECKING:
    from torch.distributed.tensor import DTensor

UNITS = ["embedding", "layers.0", "layers.1", "norm", "lm_head"]


def _generate(*flags: str, target: str = "torch") -> subprocess.CompletedProcess[str]:
    command = [find_compiler(), target, "--std", str(STDLIB), "--entry", "loss_packed"]
    for name, value in {**TINY_LLAMA, "P": 12}.items():
        command += ["--bind", f"{name}={value}"]
    return subprocess.run(
        [*command, *flags, str(LLAMA)], capture_output=True, text=True, check=False
    )


def test_generated_code_gathers_each_parameter_where_it_is_used() -> None:
    flags = [flag for unit in UNITS for flag in ("--fully-shard", unit)]
    completed = _generate(*flags)
    assert completed.returncode == 0, completed.stderr
    body = completed.stdout.split("def main(", 1)[1]
    parameters = re.findall(r"\bp\d+\b", body.split(")", 1)[0])
    assert body.count("= _gather(p") == len(parameters) > 0
    # The embedding table is gathered just before the lookup and dropped
    # right after it.
    lines = [line.strip() for line in body.splitlines()]
    first = next(i for i, line in enumerate(lines) if "_gather(p0," in line)
    name = lines[first].split(" = ")[0]
    assert "F.embedding(" in lines[first + 1] and name in lines[first + 1]
    assert any(line.startswith("del") and name in line for line in lines[first + 2 : first + 4])

    # `prepare` would gather each weight whole once and keep it.
    for target in ("torch", "jax"):
        refused = _generate("--fully-shard", "layers.0", "--prepare", target=target)
        assert refused.returncode != 0 and "no --prepare" in refused.stderr
    refused = _generate("--fully-shard", "layers.0", "--offload", "layers.1")
    assert refused.returncode != 0 and "offload" in refused.stderr


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    directory = tmp_path / "weights"
    directory.mkdir()
    random_llama(directory / "model.safetensors")
    return directory


def _rank(rank: int, world: int, port: int, weights: str, out: str) -> None:
    import torch.distributed as dist

    from linnet.torch import fsdp

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        model = trainable_llama(Path(weights))
        assert fully_shard(model) == UNITS
        part = model.get_parameter("root.embedding.weight")
        assert tuple(part.to_local().shape) == ((6, 8) if rank == 0 else (5, 8))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownArgumentType]
        # The part owns its memory; a view would keep the whole table alive.
        for parameter in model.parameters():
            local = parameter.to_local() if hasattr(parameter, "to_local") else None  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
            if local is not None:
                assert local.untyped_storage().nbytes() == local.numel() * local.element_size()

        # Backward keeps the parts: no whole weight outlives the forward.
        mine = next(pack(tiny_examples()[2 * rank : 2 * rank + 2], tokens=12))
        wholes: list[weakref.ref[torch.Tensor]] = []
        gathered = fsdp._whole  # pyright: ignore[reportPrivateUsage]

        def recording(part: DTensor, dtype: torch.dtype) -> torch.Tensor:
            whole = gathered(part, dtype)
            if torch.is_grad_enabled():
                wholes.append(weakref.ref(whole))
            return whole

        fsdp._whole = recording  # pyright: ignore[reportPrivateUsage]
        try:
            loss = model.run_entry("loss_packed", mine.inputs(mine.count))
            gc.collect()
            assert wholes and all(ref() is None for ref in wholes)
            loss.backward()
        finally:
            fsdp._whole = gathered  # pyright: ignore[reportPrivateUsage]
        model.zero_grad(set_to_none=True)

        trained = [p for p in model.parameters() if p.requires_grad]
        train(model, itertools.repeat(mine), optimizer=torch.optim.SGD(trained, lr=0.1), steps=3)
        model.save_weights(Path(out) / "sharded.safetensors", names="linnet")
    finally:
        dist.destroy_process_group()


def test_sharded_training_matches_one_process(weights: Path, tmp_path: Path) -> None:
    """Two processes, each on its own batch, end where one process ends that
    accumulates both batches, gradient clipping included."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.multiprocessing.spawn(_rank, args=(2, port, str(weights), str(tmp_path)), nprocs=2)  # pyright: ignore[reportPrivateImportUsage]
    sharded = load_file(str(tmp_path / "sharded.safetensors"))

    model = trainable_llama(weights)
    examples = tiny_examples()
    both = [next(pack(examples[:2], tokens=12)), next(pack(examples[2:], tokens=12))]
    trained = [p for p in model.parameters() if p.requires_grad]
    train(
        model,
        itertools.cycle(both),
        optimizer=torch.optim.SGD(trained, lr=0.1),
        steps=3,
        accumulate=2,
    )
    expected = {n.removeprefix("root."): p.detach() for n, p in model.named_parameters()}
    assert set(sharded) == {n for n in expected if not n.endswith(".bias")}
    for name, value in sharded.items():
        torch.testing.assert_close(value, expected[name], atol=1e-5, rtol=1e-4)


ADAPTED = "layers.*.attention.*_proj.weight"


def _lora_rank(rank: int, world: int, port: int, weights: str, out: str) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        model = trainable_llama(Path(weights))
        model.add_lora(ADAPTED, rank=2, alpha=4)
        fully_shard(model)
        # The adapted weights are split, in their own dtype; the adapters
        # stay whole.
        weight = model.get_parameter("root.layers.0.attention.q_proj.weight")
        adapter = model.get_parameter("root.layers.0.attention.q_proj.lora_a")
        assert hasattr(weight, "to_local") and not weight.requires_grad
        assert not hasattr(adapter, "to_local") and adapter.requires_grad
        with pytest.raises(Exception, match="before"):
            model.add_lora(ADAPTED)
        mine = next(pack(tiny_examples()[2 * rank : 2 * rank + 2], tokens=12))
        trained = [p for p in model.parameters() if p.requires_grad]
        train(model, itertools.repeat(mine), optimizer=torch.optim.SGD(trained, lr=0.1), steps=3)
        model.save_weights(
            Path(out) / "adapters.safetensors", names="linnet", include=["*.lora_a", "*.lora_b"]
        )
    finally:
        dist.destroy_process_group()


def test_adapters_on_a_sharded_model_match_one_process(weights: Path, tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.multiprocessing.spawn(_lora_rank, args=(2, port, str(weights), str(tmp_path)), nprocs=2)  # pyright: ignore[reportPrivateImportUsage]
    sharded = load_file(str(tmp_path / "adapters.safetensors"))

    model = trainable_llama(weights)
    model.add_lora(ADAPTED, rank=2, alpha=4)
    examples = tiny_examples()
    both = [next(pack(examples[:2], tokens=12)), next(pack(examples[2:], tokens=12))]
    trained = [p for p in model.parameters() if p.requires_grad]
    train(
        model,
        itertools.cycle(both),
        optimizer=torch.optim.SGD(trained, lr=0.1),
        steps=3,
        accumulate=2,
    )
    expected = {n.removeprefix("root."): p.detach() for n, p in model.named_parameters()}
    assert len(sharded) == 16
    for name, value in sharded.items():
        torch.testing.assert_close(value, expected[name], atol=1e-5, rtol=1e-4)
