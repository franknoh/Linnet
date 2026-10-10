"""DPO and GRPO on a model split across processes (`tensor_parallel=`): two
`gloo` processes, each with half of every split weight, train as the whole
model does on one -- the same steps and the same weights after."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import itertools
import os
import random
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file  # type: ignore[import-untyped]
from torch.multiprocessing.spawn import spawn

from linnet.serve import Completion, Request
from linnet.torch import LinnetModule, load
from linnet.train.dpo import Pair, dpo
from linnet.train.grpo import Prompt, grpo

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.split_rl

use std.nn.activations::{silu}
use std.nn.embedding::{Embedding}
use std.nn.linear::{Linear}
use std.nn.loss::{split_token_log_probs}
use std.nn.norm::{RmsNorm}
use std.nn.parallel::{all_gather, all_reduce, shared}

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

    fn hidden<P: Dim>(tokens: Tensor[P; i32]) -> Tensor[P, H; T] {
        var x = embedding.forward(tokens)
        static for layer in layers {
            x = layer.forward(x)
        }
        return shared(norm.forward(x))
    }

    // Every position's logits over the whole vocabulary, gathered.
    pub entry forward<B: Dim, S: Dim>(tokens: Tensor[B, S; i32]) -> Tensor[B, S, Vocab; T] {
        let flat = hidden(reshape(tokens, [B * S]))
        let logits = all_gather<B * S, Vocab / Shards, Shards, T>(head.forward(flat))
        return reshape(logits, [B, S, Vocab])
    }

    pub entry log_probs_packed<P: Dim>(
        tokens: Tensor[P; i32],
        positions: Tensor[P; i32],
        segments: Tensor[P; i32],
        targets: Tensor[P; i64],
    ) -> Tensor[P; f32] {
        return split_token_log_probs<P, H, Vocab / Shards, Shards, T>(
            hidden(tokens),
            head.weight,
            targets,
        )
    }
}
"""
GENERICS: dict[str, int | str] = {"Vocab": 10, "H": 8, "Inner": 12, "Layers": 2}


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


def _files(directory: Path) -> tuple[Path, Path]:
    source = directory / "split_rl.linnet"
    source.write_text(SOURCE, encoding="utf-8")
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
    weights = directory / "split_rl.safetensors"
    save_file(tensors, str(weights))
    return source, weights


def _model(
    source: str | Path,
    weights: str | Path,
    *,
    trainable: bool = False,
    tensor_parallel: DeviceMesh | None = None,
) -> LinnetModule:
    return load(
        source,
        generics=GENERICS,
        std_root=STDLIB,
        weights=weights,
        compile=True,
        trainable=trainable,
        tensor_parallel=tensor_parallel,
    )


def _pairs() -> list[Pair]:
    generator = torch.Generator().manual_seed(2)
    return [
        Pair(torch.randint(0, 10, (2,), generator=generator).tolist(), [3, 3, 3], [7, 7, 7])
        for _ in range(8)
    ]


def _prompts() -> itertools.cycle[Prompt]:
    rng = random.Random(0)
    return itertools.cycle([Prompt([rng.randrange(10), rng.randrange(10)]) for _ in range(4)])


def _threes(_: Prompt, completion: list[int]) -> float:
    return sum(token == 3 for token in completion) / len(completion)


class _Sampler:
    """Draws completions from the model's `forward`, as `Engine` would."""

    def __init__(self, model: LinnetModule) -> None:
        self.model = model

    def load_weights(self, source: LinnetModule) -> None:
        self.model.copy_weights(source)

    def run(self, requests: Sequence[Request]) -> tuple[list[Completion], None]:
        rows = [list(r.prompt) for r in requests]
        draws = [torch.Generator().manual_seed(r.seed or 0) for r in requests]
        for _ in range(requests[0].max_new_tokens):
            with torch.no_grad():
                logits = self.model.run_entry("forward", [torch.tensor(rows, dtype=torch.int32)])
            probs = torch.softmax(logits[:, -1] / requests[0].temperature, -1)
            for row, p, draw in zip(rows, probs, draws, strict=True):
                row.append(int(torch.multinomial(p, 1, generator=draw)))
        return [
            Completion(r, row[len(r.prompt) :]) for r, row in zip(requests, rows, strict=True)
        ], None


def _run(
    method: str,
    policy: LinnetModule,
    source: str,
    weights: str,
    mesh: DeviceMesh | None = None,
) -> dict[str, object]:
    """One method's steps and the weights after them."""
    trained = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trained, lr=0.3)
    if method == "dpo":
        history = dpo(policy, _pairs(), optimizer=optimizer, steps=2, pairs_per_step=4, tokens=24)
        steps = [v for step in history for v in (step.loss, step.margin, step.grad_norm)]
    else:
        sampler = _Sampler(_model(source, weights, tensor_parallel=mesh))
        grpo_history = grpo(
            policy,
            sampler,
            _prompts(),
            _threes,
            optimizer=optimizer,
            steps=2,
            group=4,
            prompts_per_step=2,
            max_new_tokens=4,
            tokens=32,
            seed=0,
        )
        steps = [v for step in grpo_history for v in (step.reward, step.loss, step.grad_norm)]
    # The weights the checkpoint holds; an absent bias is left out.
    bound = set(getattr(policy, "weight_names", {}))
    return {
        "steps": steps,
        "trained": {
            name.removeprefix("root."): p.detach()
            for name, p in policy.named_parameters()
            if name.removeprefix("root.") in bound
        },
        "parts": dict(getattr(policy, "shard_parts", {})),
    }


def _split_rank(rank: int, port: int, method: str, source: str, weights: str, out: str) -> None:
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (2,))
        policy = _model(source, weights, trainable=True, tensor_parallel=mesh)
        torch.save(_run(method, policy, source, weights, mesh), f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("method", ["dpo", "grpo"])
def test_split_training_matches_one_process(tmp_path: Path, method: str) -> None:
    source, weights = _files(tmp_path)
    whole = _run(method, _model(source, weights, trainable=True), str(source), str(weights))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = str(tmp_path / "split")
    spawn(_split_rank, args=(port, method, str(source), str(weights), out), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(f"{out}.{rank}")
        assert result["steps"] == pytest.approx(whole["steps"], rel=1e-4, abs=1e-6)
        parts = result["parts"]
        assert "head.weight" in parts
        for name, want in cast("dict[str, torch.Tensor]", whole["trained"]).items():
            got = result["trained"][name]
            if name in parts:
                axis, extent = parts[name]
                want = want.narrow(axis, rank * extent, extent)
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
