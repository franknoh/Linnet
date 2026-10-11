"""DPO and GRPO in JAX: DPO raises the preferred answer's margin and takes
the steps PyTorch's DPO takes; GRPO learns what its reward asks, and with
the engine reporting the policy's own log-probabilities sees no mismatch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
import torch

from linnet.jax import SourceFunction, load_source
from linnet.jax.dpo import dpo
from linnet.jax.grpo import grpo
from linnet.serve import Completion, Request
from tests.helpers import (
    LLAMA,
    STDLIB,
    TINY_LLAMA,
    random_llama,
    threes,
    tiny_pairs,
    tiny_prompts,
)


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    directory = tmp_path / "weights"
    directory.mkdir()
    random_llama(directory / "model.safetensors")
    return directory


def _entry(weights: Path, entry: str) -> SourceFunction:
    return load_source(LLAMA, generics=TINY_LLAMA, weights=weights, entry=entry, std_root=STDLIB)


def test_dpo_raises_the_margin(weights: Path) -> None:
    _, history = dpo(
        _entry(weights, "log_probs_packed"),
        tiny_pairs(24),
        optimizer=optax.adamw(1e-2),
        steps=6,
        pairs_per_step=4,
        beta=0.5,
        tokens=24,
    )
    margins = [step.margin for step in history]
    assert len(margins) == 6 and margins[0] == pytest.approx(0, abs=1e-5)
    assert margins[-1] > max(0.3, margins[1])


def test_dpo_takes_the_steps_pytorch_takes(weights: Path) -> None:
    from linnet.torch import load as load_torch
    from linnet.train.dpo import dpo as dpo_torch

    params, _ = dpo(
        _entry(weights, "log_probs_packed"),
        tiny_pairs(8),
        optimizer=optax.sgd(0.1),
        steps=2,
        pairs_per_step=4,
        beta=0.5,
        tokens=24,
        clip=None,
    )
    theirs = load_torch(
        LLAMA, generics=TINY_LLAMA, std_root=STDLIB, weights=weights, compile=True, trainable=True
    )
    trained = [p for p in theirs.parameters() if p.requires_grad]
    dpo_torch(
        theirs,
        tiny_pairs(8),
        optimizer=torch.optim.SGD(trained, lr=0.1),
        steps=2,
        pairs_per_step=4,
        beta=0.5,
        tokens=24,
        clip_grad=None,
    )
    for name, parameter in theirs.named_parameters():
        path = name.removeprefix("root.")
        if path in params:
            np.testing.assert_allclose(
                np.asarray(params[path]), parameter.detach().numpy(), rtol=1e-4, atol=1e-5
            )


class _Sampler:
    """Draws completions from the Llama example's `forward` in JAX, as an
    `Engine` would: every request's prompt the same length, each drawn with
    its own seed, its log-probabilities before temperature reported."""

    def __init__(self, model: SourceFunction) -> None:
        self.model = model
        self.parameters: dict[str, jax.Array] | None = None

    def load_weights(self, parameters: dict[str, jax.Array]) -> None:
        self.parameters = dict(parameters)

    def run(self, requests: Sequence[Request]) -> tuple[list[Completion], None]:
        rows = [list(r.prompt) for r in requests]
        draws = [np.random.default_rng(r.seed or 0) for r in requests]
        chosen: list[list[float]] = [[] for _ in requests]
        for _ in range(requests[0].max_new_tokens):
            tokens = np.asarray(rows, dtype=np.int32)
            logits = (
                self.model(tokens)
                if self.parameters is None
                else self.model.apply(self.parameters, tokens)
            )
            last = np.asarray(logits[:, -1], dtype=np.float64)
            top = last.max(-1, keepdims=True)
            logs = last - top - np.log(np.exp(last - top).sum(-1, keepdims=True))
            for row, log, draw, mine in zip(rows, logs, draws, chosen, strict=True):
                probs = np.exp(log / requests[0].temperature)
                token = int(draw.choice(len(probs), p=probs / probs.sum()))
                row.append(token)
                mine.append(float(log[token]))
        done = [
            Completion(
                request,
                row[len(request.prompt) :],
                logprobs=mine if request.logprobs is not None else [],
            )
            for request, row, mine in zip(requests, rows, chosen, strict=True)
        ]
        return done, None


def test_grpo_learns_what_the_reward_asks(weights: Path) -> None:
    _, history = grpo(
        _entry(weights, "log_probs_packed"),
        _Sampler(_entry(weights, "forward")),
        tiny_prompts(),
        threes,
        optimizer=optax.adamw(3e-2),
        steps=10,
        group=6,
        prompts_per_step=4,
        max_new_tokens=6,
        tokens=64,
        correction_cap=2.0,
    )
    rewards = [step.reward for step in history]
    assert len(rewards) == 10
    assert sum(rewards[-3:]) / 3 > sum(rewards[:3]) / 3 + 0.1, rewards
    # The sampler is the policy itself: nothing to correct.
    assert all(step.mismatch is not None and step.mismatch < 1e-4 for step in history)
    assert all(step.clipped == 0 for step in history)


def test_grpo_takes_the_engine_weights_each_step(weights: Path) -> None:
    sampler = _Sampler(_entry(weights, "forward"))
    params, _ = grpo(
        _entry(weights, "log_probs_packed"),
        sampler,
        tiny_prompts(),
        threes,
        optimizer=optax.sgd(0.1),
        steps=3,
        group=4,
        prompts_per_step=2,
        max_new_tokens=6,
        tokens=64,
    )
    assert sampler.parameters is not None
    # The last step's sampling saw the weights the second step left; they
    # differ from the start and from the end.
    start = _entry(weights, "log_probs_packed").parameters_for(
        np.zeros(64, np.int32),
        np.zeros(64, np.int32),
        np.zeros(64, np.int32),
        np.zeros(64, np.int64),
    )
    path = "layers.0.mlp.gate.weight"
    assert not np.allclose(np.asarray(sampler.parameters[path]), np.asarray(start[path]))
    assert not np.allclose(np.asarray(sampler.parameters[path]), np.asarray(params[path]))
    assert jnp.asarray(params[path]).dtype == jnp.float32  # pyright: ignore[reportUnknownMemberType]


MESH_SCRIPT = """
import sys
from pathlib import Path

import jax
import numpy as np
import optax
from jax.sharding import Mesh

sys.path.insert(0, sys.argv[2])
from test_rl import _entry, tiny_pairs, tiny_prompts, _Sampler, threes  # noqa: E402

from linnet.jax.dpo import dpo  # noqa: E402
from linnet.jax.grpo import grpo  # noqa: E402

weights = Path(sys.argv[1])
assert len(jax.devices()) == 2
mesh = Mesh(np.array(jax.devices()), ("data",))


def compare(split, one):
    for path, value in one.items():
        np.testing.assert_allclose(np.asarray(split[path]), np.asarray(value), rtol=1e-4, atol=1e-6)


# DPO: two batches a step, one per device.
options = dict(optimizer=optax.sgd(0.1), steps=2, pairs_per_step=4, beta=0.5, tokens=24, clip=None)
split, _ = dpo(_entry(weights, "log_probs_packed"), tiny_pairs(8), mesh=mesh, **options)
one, _ = dpo(_entry(weights, "log_probs_packed"), tiny_pairs(8), **options)
compare(split, one)

# GRPO: one batch a step, the other device's a batch that learns nothing;
# the engine samples from the weights as they are split.
options = dict(optimizer=optax.sgd(0.1), steps=2, group=4, prompts_per_step=2, max_new_tokens=6,
               tokens=64, clip_grad=None)
split, _ = grpo(_entry(weights, "log_probs_packed"), _Sampler(_entry(weights, "forward")),
                tiny_prompts(), threes, mesh=mesh, remat=True, **options)
one, _ = grpo(_entry(weights, "log_probs_packed"), _Sampler(_entry(weights, "forward")),
              tiny_prompts(), threes, **options)
compare(split, one)
print("same")
"""


def test_dpo_and_grpo_over_a_mesh_match_one_device(weights: Path, tmp_path: Path) -> None:
    """Two host devices, each with its own batches and part of every weight,
    take the steps one device takes."""
    import os
    import subprocess
    import sys

    script = tmp_path / "mesh.py"
    script.write_text(MESH_SCRIPT, encoding="utf-8")
    environment = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    completed = subprocess.run(
        [sys.executable, str(script), str(weights), str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0 and "same" in completed.stdout, completed.stderr[-3000:]
