"""Fully sharded data parallelism (`fully_shard`): each process keeps part
of every parameter, the generated code gathers a parameter where it is used
and drops it after, backward keeps the parts rather than the wholes, and
training takes the steps one process takes over every process's batches."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false, reportAttributeAccessIssue=false, reportPrivateUsage=false

from __future__ import annotations

import gc
import itertools
import os
import re
import socket
import subprocess
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file  # type: ignore[import-untyped]

from linnet.compiler import find_compiler
from linnet.torch import LinnetModule, fully_shard, load
from linnet.train import Example, pack, train

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
UNITS = ["embedding", "layers.0", "layers.1", "norm", "lm_head"]


def _generate(*flags: str) -> subprocess.CompletedProcess[str]:
    command = [find_compiler(), "torch", "--std", str(STDLIB), "--entry", "loss_packed"]
    for name, value in {**GENERICS, "P": 12}.items():
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

    refused = _generate("--fully-shard", "layers.0", "--prepare")
    assert refused.returncode != 0 and "no --prepare" in refused.stderr
    refused = _generate("--fully-shard", "layers.0", "--offload", "layers.1")
    assert refused.returncode != 0 and "offload" in refused.stderr


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
        LLAMA, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True, trainable=True
    )


def _examples() -> list[Example]:
    generator = torch.Generator().manual_seed(1)
    return [
        Example.prompted(
            torch.randint(0, 11, (2,), generator=generator).tolist(),
            torch.randint(0, 11, (n - 2,), generator=generator).tolist(),
        )
        for n in [5, 6, 4, 7]
    ]


def _rank(rank: int, world: int, port: int, weights: str, out: str) -> None:
    import torch.distributed as dist

    from linnet.torch import fsdp

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        model = _load(Path(weights))
        assert fully_shard(model) == UNITS
        part = model.get_parameter("root.embedding.weight")
        assert tuple(part.to_local().shape) == ((6, 8) if rank == 0 else (5, 8))

        # Backward keeps the parts: no whole weight outlives the forward.
        mine = next(pack(_examples()[2 * rank : 2 * rank + 2], tokens=12))
        wholes: list[Any] = []
        gathered = fsdp._whole

        def recording(part: Any, dtype: torch.dtype) -> torch.Tensor:
            whole = gathered(part, dtype)
            if torch.is_grad_enabled():
                wholes.append(weakref.ref(whole))
            return whole

        fsdp._whole = recording
        try:
            loss = model.run_entry("loss_packed", mine.inputs(mine.count))
            gc.collect()
            assert wholes and all(ref() is None for ref in wholes)
            loss.backward()
        finally:
            fsdp._whole = gathered
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
    torch.multiprocessing.spawn(_rank, args=(2, port, str(weights), str(tmp_path)), nprocs=2)
    sharded = load_file(str(tmp_path / "sharded.safetensors"))

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
    expected = {n.removeprefix("root."): p.detach() for n, p in model.named_parameters()}
    assert set(sharded) == {n for n in expected if not n.endswith(".bias")}
    for name, value in sharded.items():
        torch.testing.assert_close(value, expected[name], atol=1e-5, rtol=1e-4)
