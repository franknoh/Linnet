"""GRPO (`linnet.train.grpo`): group advantages, the clipped surrogate's
gradient, weights copied from a policy with adapters into the model that
samples, and a policy that learns what its reward asks for."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import itertools
import os
import random
import socket
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.serve import Completion, Request
from linnet.torch import LinnetModule, fully_shard, load
from linnet.train.grpo import Prompt, group_advantages, grpo, grpo_loss

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
TOKENS = torch.tensor([[3, 1, 4, 1, 5, 9, 2]], dtype=torch.int32)


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    skeleton = load(LLAMA, generics=GENERICS, std_root=STDLIB)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        name.removeprefix("root."): torch.randn(parameter.shape, generator=generator) * 0.3
        for name, parameter in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return path


def _model(weights: Path, **options: object) -> LinnetModule:
    return load(LLAMA, generics=GENERICS, std_root=STDLIB, weights=weights, **options)  # type: ignore[arg-type]


def test_group_advantages() -> None:
    assert group_advantages([1, 3, 5, 5], 2, scale=False) == [-1, 1, 0, 0]
    scaled = group_advantages([0, 2, 7, 7], 2)
    assert scaled[0] == pytest.approx(-1 / (2**0.5 + 1e-4)) and scaled[2:] == [0, 0]


def test_the_surrogate_is_the_policy_gradient_until_clipped() -> None:
    log_probs = torch.tensor([-1.0, -2.0, -0.5, -0.5], requires_grad=True)
    advantages = torch.tensor([1.0, -2.0, 1.0, -1.0])
    weights = torch.tensor([0.25, 0.25, 0.25, 0.0])
    loss, share, _ = grpo_loss(log_probs, log_probs.detach(), advantages, weights)
    loss.backward()
    assert log_probs.grad is not None
    torch.testing.assert_close(log_probs.grad, -advantages * weights)
    assert float(share) == 0

    # A ratio past 1 + 0.2 for a good token stops its gradient; for a bad
    # token the larger penalty stands.
    log_probs.grad = None
    old = log_probs.detach() - torch.tensor([0.5, 0.5, 0.0, 0.0])
    loss, share, _ = grpo_loss(log_probs, old, advantages, weights)
    loss.backward()
    grad = log_probs.grad
    assert grad is not None and float(grad[0]) == 0 and float(grad[1]) != 0
    assert float(share) == pytest.approx(0.25)


def test_copied_weights_carry_the_adapters(weights: Path) -> None:
    policy = _model(weights, compile=True, trainable=True)
    policy.add_lora("layers.*.attention.*_proj.weight", rank=2, alpha=4)
    trained = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trained, lr=5e-2)
    for _ in range(3):
        policy.run_entry("forward", [TOKENS]).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    served = _model(weights, compile=True)
    with torch.no_grad():
        before = served.run_entry("forward", [TOKENS])
        served.copy_weights(policy)
        after = served.run_entry("forward", [TOKENS])
        expected = policy.run_entry("forward", [TOKENS])
    assert not torch.allclose(before, expected)
    torch.testing.assert_close(after, expected, atol=1e-5, rtol=1e-5)


class _Sampler:
    """Draws completions from the Llama example's `forward`, as `Engine`
    would from its cache: every request's prompt the same length."""

    def __init__(self, model: LinnetModule) -> None:
        self.model = model

    def load_weights(self, source: LinnetModule) -> None:
        self.model.copy_weights(source)

    def run(self, requests: Sequence[Request]) -> tuple[list[Completion], None]:
        rows = [list(r.prompt) for r in requests]
        draws = [torch.Generator().manual_seed(r.seed or 0) for r in requests]
        # Each token's log-probability before temperature, as `Engine` reports.
        chosen: list[list[float]] = [[] for _ in requests]
        for _ in range(requests[0].max_new_tokens):
            with torch.no_grad():
                logits = self.model.run_entry("forward", [torch.tensor(rows, dtype=torch.int32)])
            probs = torch.softmax(logits[:, -1] / requests[0].temperature, -1)
            logs = torch.log_softmax(logits[:, -1], -1)
            for row, p, log, draw, mine in zip(rows, probs, logs, draws, chosen, strict=True):
                token = int(torch.multinomial(p, 1, generator=draw))
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


def _prompts() -> itertools.cycle[Prompt]:
    rng = random.Random(0)
    return itertools.cycle([Prompt([rng.randrange(11), rng.randrange(11)]) for _ in range(8)])


def _threes(_: Prompt, completion: list[int]) -> float:
    return sum(token == 3 for token in completion) / len(completion)


def test_grpo_learns_what_the_reward_asks(weights: Path) -> None:
    torch.manual_seed(0)
    policy = _model(weights, compile=True, trainable=True)
    trained = [p for p in policy.parameters() if p.requires_grad]
    history = grpo(
        policy,
        _Sampler(_model(weights)),
        _prompts(),
        _threes,
        optimizer=torch.optim.AdamW(trained, lr=3e-2),
        steps=10,
        group=6,
        prompts_per_step=4,
        max_new_tokens=6,
        tokens=64,
    )
    rewards = [step.reward for step in history]
    assert len(rewards) == 10
    assert sum(rewards[-3:]) / 3 > sum(rewards[:3]) / 3 + 0.1, rewards
    # One optimizer step per sample: the ratio is 1, so nothing is clipped.
    assert all(step.clipped == 0 and step.kl is None for step in history)


def test_grpo_reuses_samples_against_a_reference(weights: Path) -> None:
    policy = _model(weights, compile=True, trainable=True)
    trained = [p for p in policy.parameters() if p.requires_grad]
    history = grpo(
        policy,
        _Sampler(_model(weights)),
        _prompts(),
        _threes,
        optimizer=torch.optim.AdamW(trained, lr=3e-2),
        steps=2,
        group=4,
        prompts_per_step=2,
        max_new_tokens=6,
        tokens=64,
        iterations=2,
        beta=0.1,
        reference=_model(weights, compile=True),
    )
    assert len(history) == 2
    assert all(step.kl is not None and step.kl >= 0 for step in history)
    # The second step starts after the first moved the policy away.
    assert history[1].kl is not None and history[1].kl > 0
    with pytest.raises(ValueError, match="reference"):
        # No engine: the options are refused before one is needed.
        grpo(policy, None, [], _threes, optimizer=torch.optim.SGD(trained), steps=1, beta=0.1)  # pyright: ignore[reportArgumentType]


def _grpo_rank(rank: int, world: int, port: int, weights: str) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        policy = _model(Path(weights), compile=True, trainable=True)
        fully_shard(policy)
        sampler = _Sampler(_model(Path(weights)))
        start = torch.stack([p.detach().double().sum() for p in sampler.model.parameters()])
        trained = [p for p in policy.parameters() if p.requires_grad]
        # The first process packs four batches a step, the second two.
        history = grpo(
            policy,
            sampler,
            _prompts(),
            _threes,
            optimizer=torch.optim.AdamW(trained, lr=3e-2),
            steps=3,
            group=4,
            prompts_per_step=2 if rank == 0 else 1,
            max_new_tokens=6,
            tokens=16,
            seed=rank,
        )
        assert len(history) == 3

        def same_everywhere(value: torch.Tensor) -> torch.Tensor:
            everyone = [torch.zeros_like(value) for _ in range(world)]
            dist.all_gather(everyone, value)
            assert all(torch.equal(everyone[0], other) for other in everyone[1:])
            return value

        same_everywhere(torch.tensor([[s.reward, s.loss, s.length] for s in history]))
        # Each engine took the whole of every split weight, the same everywhere.
        sampler.load_weights(policy)
        assert sampler.model.get_parameter("root.embedding.weight").shape == (11, 8)
        end = same_everywhere(
            torch.stack([p.detach().double().sum() for p in sampler.model.parameters()])
        )
        assert not torch.equal(start, end)
    finally:
        dist.destroy_process_group()


def test_grpo_across_processes(weights: Path) -> None:
    """Two processes sample on their own and train one sharded policy, each
    packing a different number of batches."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.multiprocessing.spawn(_grpo_rank, args=(2, port, str(weights)), nprocs=2)


def test_uniform_groups_are_counted_and_dropped(weights: Path) -> None:
    """With one reward for everything, every group is uniform: reported, and
    with `drop_uniform` nothing is learned."""
    policy = _model(weights, compile=True, trainable=True)
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    trained = [p for p in policy.parameters() if p.requires_grad]
    history = grpo(
        policy,
        _Sampler(_model(weights)),
        _prompts(),
        lambda prompt, completion: 1.0,
        optimizer=torch.optim.SGD(trained, lr=0.1),
        steps=1,
        group=4,
        prompts_per_step=2,
        max_new_tokens=6,
        tokens=64,
        drop_uniform=True,
    )
    assert history[0].uniform == 1.0 and history[0].loss == 0
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name])


def test_a_resumed_grpo_run_ends_where_an_unbroken_one_does(weights: Path, tmp_path: Path) -> None:
    def run(steps: int, checkpoint: Path | None) -> tuple[LinnetModule, list[int]]:
        policy = _model(weights, compile=True, trainable=True)
        trained = [p for p in policy.parameters() if p.requires_grad]
        history = grpo(
            policy,
            _Sampler(_model(weights)),
            _prompts(),
            _threes,
            optimizer=torch.optim.AdamW(trained, lr=3e-2),
            steps=steps,
            group=4,
            prompts_per_step=2,
            max_new_tokens=6,
            tokens=64,
            checkpoint=checkpoint,
            checkpoint_every=1 if checkpoint is not None else None,
        )
        return policy, [step.step for step in history]

    unbroken, numbers = run(4, None)
    assert numbers == [1, 2, 3, 4]
    assert run(2, tmp_path / "run")[1] == [1, 2]
    resumed, numbers = run(4, tmp_path / "run")
    assert numbers == [3, 4]
    theirs = dict(unbroken.named_parameters())
    for name, parameter in resumed.named_parameters():
        torch.testing.assert_close(parameter.detach(), theirs[name].detach())


def test_the_correction_weighs_each_token() -> None:
    log_probs = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    advantages = torch.tensor([1.0, -2.0, 1.0])
    weights = torch.tensor([0.5, 0.25, 0.25])
    correction = torch.tensor([0.5, 2.0, 1.0])
    loss, _, _ = grpo_loss(
        log_probs, log_probs.detach(), advantages, weights, correction=correction
    )
    loss.backward()
    assert log_probs.grad is not None
    torch.testing.assert_close(log_probs.grad, -advantages * weights * correction)


def test_corrections_for_the_same_policy_change_nothing(weights: Path) -> None:
    """An engine computing the policy's own log-probabilities: no mismatch,
    and the steps of an uncorrected run."""

    def run(cap: float | None) -> tuple[LinnetModule, list[float | None]]:
        policy = _model(weights, compile=True, trainable=True)
        trained = [p for p in policy.parameters() if p.requires_grad]
        history = grpo(
            policy,
            _Sampler(_model(weights)),
            _prompts(),
            _threes,
            optimizer=torch.optim.AdamW(trained, lr=3e-2),
            steps=3,
            group=4,
            prompts_per_step=2,
            max_new_tokens=6,
            tokens=64,
            correction_cap=cap,
        )
        return policy, [step.mismatch for step in history]

    plain, none = run(None)
    corrected, mismatches = run(2.0)
    assert none == [None] * 3
    assert all(m is not None and m < 1e-4 for m in mismatches)
    theirs = dict(plain.named_parameters())
    for name, parameter in corrected.named_parameters():
        torch.testing.assert_close(parameter.detach(), theirs[name].detach(), atol=1e-4, rtol=1e-3)
    with pytest.raises(ValueError, match="temperature 1"):
        grpo(
            plain,
            _Sampler(_model(weights)),
            _prompts(),
            _threes,
            optimizer=torch.optim.SGD([p for p in plain.parameters() if p.requires_grad]),
            steps=1,
            temperature=0.7,
            correction_cap=2.0,
        )
