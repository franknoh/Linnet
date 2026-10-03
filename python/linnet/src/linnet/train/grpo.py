"""Group relative policy optimization (GRPO) with a serving engine sampling.

Each step samples `group` completions of each of `prompts_per_step` prompts
from an `Engine` holding the policy's latest weights and scores them with
`reward`. A completion's advantage is its reward less its group's mean, over
the group's standard deviation. The policy then takes the clipped
policy-gradient step on its completions' tokens:

```python
from linnet import nest
from linnet.serve import Engine
from linnet.train.grpo import grpo

policy = nest.load(card, backend="torch", device="cuda", compile="inductor", trainable=True)
engine = Engine(nest.load(card, backend="torch", device="cuda", compile=True))
optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-6)
grpo(policy, engine, prompts, reward, optimizer=optimizer, steps=200, group=8)
```

The policy needs a `log_probs_packed` entry: `(tokens [P] i32, positions [P]
i32, segments [P] i32, targets [P] i64) -> [P] f32`, each position's
log-probability of its target. Each Nest decoder card has one.
"""

from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..serve import Request
from . import Batch, Example, clip_gradients, pack, reduce_gradients


@dataclass(frozen=True)
class Prompt:
    """A prompt's tokens, and what `reward` needs to score its completions
    (the answer, say)."""

    tokens: Sequence[int]
    data: Any = None


Reward = Callable[[Prompt, list[int]], float]


@dataclass
class GrpoStep:
    """What one step did: `reward` and `reward_std` over its completions,
    `length` their mean token count, `clipped` the share of learned tokens
    whose ratio was clipped, `kl` the mean estimate against `reference`."""

    step: int
    reward: float
    reward_std: float
    length: float
    loss: float
    clipped: float
    kl: float | None
    sample_seconds: float
    train_seconds: float
    learning_rate: float
    grad_norm: float | None


def group_advantages(rewards: Sequence[float], group: int, *, scale: bool = True) -> list[float]:
    """Each reward less its group's mean (`group` consecutive rewards), over
    the group's standard deviation when `scale`."""
    advantages: list[float] = []
    for start in range(0, len(rewards), group):
        chunk = rewards[start : start + group]
        mean = sum(chunk) / len(chunk)
        spread = (
            math.sqrt(sum((r - mean) ** 2 for r in chunk) / (len(chunk) - 1))
            if len(chunk) > 1
            else 0.0
        )
        advantages += [(r - mean) / (spread + 1e-4) if scale else r - mean for r in chunk]
    return advantages


def grpo_loss(
    log_probs: torch.Tensor,
    old: torch.Tensor,
    advantages: torch.Tensor,
    weights: torch.Tensor,
    *,
    clip: tuple[float, float] = (0.2, 0.2),
    reference: torch.Tensor | None = None,
    beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The clipped surrogate over token positions, `[P]` each: `log_probs`
    under the policy, `old` under the policy that sampled, each position's
    `advantages`, and `weights` (0 where nothing is learned, summing to 1
    over a step). With `reference` log-probabilities and `beta`, adds
    `beta` times the k3 estimate of the KL divergence from it. Returns the
    loss, the weighted share of clipped positions, and the KL estimate."""
    ratio = torch.exp(log_probs - old)
    low, high = clip
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - low, 1 + high) * advantages)
    per_token = -surrogate
    kl = None
    if reference is not None and beta:
        delta = reference - log_probs
        estimate = torch.exp(delta) - delta - 1
        per_token = per_token + beta * estimate
        kl = (estimate.detach() * weights).sum()
    with torch.no_grad():
        clipped = ((ratio < 1 - low) & (advantages < 0)) | ((ratio > 1 + high) & (advantages > 0))
        share = (clipped.float() * weights).sum()
    return (per_token * weights).sum(), share, kl


def grpo(
    policy: Any,
    engine: Any,
    prompts: Iterable[Prompt | Sequence[int]],
    reward: Reward,
    *,
    optimizer: torch.optim.Optimizer,
    steps: int,
    group: int = 8,
    prompts_per_step: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos: Iterable[int] = (),
    tokens: int = 4096,
    iterations: int = 1,
    clip: tuple[float, float] = (0.2, 0.2),
    beta: float = 0.0,
    reference: Any = None,
    scale_rewards: bool = True,
    clip_grad: float | None = 1.0,
    schedule: Any = None,
    entry: str = "log_probs_packed",
    seed: int = 0,
    warmup: bool = True,
    on_step: Callable[[GrpoStep], None] | None = None,
) -> list[GrpoStep]:
    """Trains `policy` with GRPO for `steps` steps or until `prompts` run out.

    `engine` samples: a `linnet.serve.Engine` (anything with
    `load_weights(model)` and `run(requests)`) over a second copy of the
    model, which takes the policy's weights before each step, its adapters
    merged in. Completions are drawn at `temperature`, `top_k` and `top_p`,
    up to `max_new_tokens` or a token in `eos`, each with its own seed from
    `seed`. They are packed into batches of `tokens` positions.

    Each step's samples serve `iterations` optimizer steps, the first with
    the sampling policy's log-probabilities (recomputed by `policy`), so
    its ratio is 1. The loss is the mean over every learned token of the
    step (`grpo_loss`, the ratio clipped to `1 - clip[0]`, `1 + clip[1]`),
    with `beta` times the KL estimate against `reference` (a frozen model
    with the same entry) when `beta` is set. `clip_grad` caps the gradient
    norm; `schedule` steps after each optimizer step.

    Under `torch.distributed`, each process samples its own `prompts` with
    its own engine and the processes train together: the loss is the mean
    over every process's learned tokens, the gradients are summed (or
    reduced into the parts of a policy split by `fully_shard`), and the
    step's numbers cover every process. A process with fewer batches runs
    empty ones, so every process makes the same calls. Give each process its
    own `seed`; all stop when any runs out of prompts.

    `warmup` first compiles every pass the engine can run
    (`Engine.warmup`). Otherwise a pass size first met mid-run compiles
    then, and under `torch.distributed` every other process waits for it."""
    import torch.distributed as dist

    if beta and reference is None:
        raise ValueError("a KL penalty (`beta`) needs a `reference` model")
    device = next(policy.parameters()).device
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    trained = [p for group_ in optimizer.param_groups for p in group_["params"]]
    draws = random.Random(seed)
    stop = frozenset(eos)
    source = iter(prompts)
    history: list[GrpoStep] = []
    if warmup and callable(getattr(engine, "warmup", None)):
        engine.warmup()
    for step in range(1, steps + 1):
        chosen = [_prompt(p) for p in itertools.islice(source, prompts_per_step)]
        if distributed:
            ready = torch.tensor([float(bool(chosen))], device=device)
            dist.all_reduce(ready, op=dist.ReduceOp.MIN)
            if not ready.item():
                break
        elif not chosen:
            break

        begin = time.perf_counter()
        engine.load_weights(policy)
        requests = [
            Request(
                prompt=list(prompt.tokens),
                max_new_tokens=max_new_tokens,
                eos=stop,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=draws.getrandbits(32),
            )
            for prompt in chosen
            for _ in range(group)
        ]
        completions, _ = engine.run(requests)
        sampled = time.perf_counter() - begin

        begin = time.perf_counter()
        rewards = [
            float(reward(chosen[index // group], list(completion.tokens)))
            for index, completion in enumerate(completions)
        ]
        advantages = group_advantages(rewards, group, scale=scale_rewards)
        examples = [Example.prompted(c.request.prompt, c.tokens) for c in completions]
        batches = list(pack(examples, tokens))
        count = float(sum(batch.count for batch in batches))
        if distributed:
            # The count over every process, and as many batches as the
            # process with the most: a sharded policy gathers in each call.
            totals = torch.tensor([count], dtype=torch.float64, device=device)
            dist.all_reduce(totals)
            count = float(totals.item())
            most = torch.tensor([len(batches)], device=device)
            dist.all_reduce(most, op=dist.ReduceOp.MAX)
            batches += [_empty(tokens)] * (int(most.item()) - len(batches))
        count = max(1.0, count)
        prepared = [_Prepared(batch, advantages, count, device) for batch in batches]
        if iterations > 1 or reference is not None:
            with torch.no_grad():
                for item in prepared:
                    if iterations > 1:
                        item.old = policy.run_entry(entry, item.inputs).detach()
                    if reference is not None and beta:
                        item.reference = reference.run_entry(entry, item.inputs).detach()

        loss_total = clipped_total = 0.0
        kl_total: float | None = None
        norm: float | None = None
        for _ in range(iterations):
            loss_total = clipped_total = 0.0
            kl_total = None
            for item in prepared:
                log_probs = policy.run_entry(entry, item.inputs)
                old = item.old if item.old is not None else log_probs.detach()
                loss, share, kl = grpo_loss(
                    log_probs,
                    old,
                    item.advantages,
                    item.weights,
                    clip=clip,
                    reference=item.reference,
                    beta=beta,
                )
                loss.backward()  # pyright: ignore[reportUnknownMemberType]
                loss_total += float(loss.detach())
                clipped_total += float(share)
                if kl is not None:
                    kl_total = (kl_total or 0.0) + float(kl)
            if distributed:
                reduce_gradients(trained)
                sums = torch.tensor(
                    [loss_total, clipped_total, kl_total or 0.0], dtype=torch.float64, device=device
                )
                dist.all_reduce(sums)
                loss_total, clipped_total = float(sums[0]), float(sums[1])
                kl_total = float(sums[2]) if kl_total is not None else None
            norm = clip_gradients(trained, clip_grad) if clip_grad is not None else None
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if schedule is not None:
                schedule.step()

        # Over every process: the rewards' sum and sum of squares, the
        # completions, and their tokens.
        moments = [
            sum(rewards),
            sum(r * r for r in rewards),
            float(len(rewards)),
            float(sum(len(c.tokens) for c in completions)),
        ]
        if distributed:
            summed = torch.tensor(moments, dtype=torch.float64, device=device)
            dist.all_reduce(summed)
            moments = [float(value) for value in summed]
        total, squares, n, generated = moments
        mean = total / n
        record = GrpoStep(
            step=step,
            reward=mean,
            reward_std=math.sqrt(max(0.0, squares / n - mean * mean)),
            length=generated / n,
            loss=loss_total,
            clipped=clipped_total,
            kl=kl_total,
            sample_seconds=sampled,
            train_seconds=time.perf_counter() - begin,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            grad_norm=norm,
        )
        history.append(record)
        if on_step is not None:
            on_step(record)
    return history


def _prompt(value: Prompt | Sequence[int]) -> Prompt:
    return value if isinstance(value, Prompt) else Prompt(value)


def _empty(size: int) -> Batch:
    """A batch that learns nothing, run so every process makes the same calls."""
    zeros = torch.zeros(size, dtype=torch.int32)
    return Batch(zeros, zeros, zeros, zeros.long(), torch.zeros(size))


class _Prepared:
    """A batch's inputs on the device, its weights (its learned positions
    over the step's count), and each position's advantage."""

    def __init__(
        self, batch: Batch, advantages: list[float], count: float, device: torch.device
    ) -> None:
        self.inputs = [
            value.to(device)
            for value in (batch.tokens, batch.positions, batch.segments, batch.targets)
        ]
        self.weights = (batch.mask / count).to(device)
        # Padding is the segment after the last sequence: no advantage.
        per_sequence = torch.tensor([advantages[i] for i in batch.items] + [0.0])
        self.advantages = per_sequence[batch.segments.long()].to(device)
        self.old: torch.Tensor | None = None
        self.reference: torch.Tensor | None = None


__all__ = ["GrpoStep", "Prompt", "Reward", "group_advantages", "grpo", "grpo_loss"]
