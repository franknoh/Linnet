"""A model that says how it splits (a `Shards` generic and
`std.nn.parallel::all_reduce`) loaded with `tensor_parallel=`: each process
binds its own part of every weight the checkpoint holds `Shards` times over,
and the result is the whole model's, on every process."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file  # type: ignore[import-untyped]
from torch.multiprocessing.spawn import spawn

from linnet.torch import load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.shards

use std.nn.activations::{silu}
use std.nn.linear::{Linear}
use std.nn.parallel::{all_reduce}

pub block Model<H: Dim, Inner: Dim, T: Float = f32, Shards: Dim = 1>
where
    Shards > 0,
    Inner % Shards == 0
{
    sub up: Linear<H, Inner / Shards, T>
    sub down: Linear<Inner / Shards, H, T>

    pub entry forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        return x + all_reduce(down.forward(silu(up.forward(x))))
    }
}
"""

GENERICS: dict[str, int | str] = {"H": 8, "Inner": 12}


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
        shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
        torch.save({"y": y, "shapes": shapes}, f"{out}.{rank}")
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
    }
    weights = tmp_path / "model.safetensors"
    save_file(tensors, str(weights))
    whole = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 10
    expected = whole.run_entry("forward", [x])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = str(tmp_path / "result")
    spawn(_shard, args=(port, str(source), str(weights), out), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(f"{out}.{rank}")
        torch.testing.assert_close(result["y"], expected, atol=1e-5, rtol=1e-5)
        # Half of `up` by its outputs (with its bias), half of `down` by its inputs.
        shapes = result["shapes"]
        assert shapes["root.up.weight"] == (6, 8) and shapes["root.up.bias"] == (6,)
        assert shapes["root.down.weight"] == (8, 6)
