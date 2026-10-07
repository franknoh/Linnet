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
from pathlib import Path
from typing import Any

import torch

from ..packing import Prompt, Reward, group_advantages
from ..serve import Request
from . import (
    Batch,
    Example,
    clip_gradients,
    load_checkpoint,
    pack,
    reduce_gradients,
    save_checkpoint,
)


@dataclass
class GrpoStep:
    """What one step did: `reward` and `reward_std` over its completions,
    `length` their mean token count, `clipped` the share of learned tokens
    whose ratio was clipped, `kl` the mean estimate against `reference`,
    `uniform` the share of groups whose rewards were all equal (no
    advantage, so nothing to learn), `mismatch` the mean absolute
    difference of the policy's and the engine's log-probabilities of the
    sampled tokens (with `correction_cap`)."""

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
    uniform: float = 0.0
    mismatch: float | None = None


def grpo_loss(
    log_probs: torch.Tensor,
    old: torch.Tensor,
    advantages: torch.Tensor,
    weights: torch.Tensor,
    *,
    clip: tuple[float, float] = (0.2, 0.2),
    reference: torch.Tensor | None = None,
    beta: float = 0.0,
    correction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The clipped surrogate over token positions, `[P]` each: `log_probs`
    under the policy, `old` under the policy that sampled, each position's
    `advantages`, and `weights` (0 where nothing is learned, summing to 1
    over a step). With `reference` log-probabilities and `beta`, adds
    `beta` times the k3 estimate of the KL divergence from it. `correction`
    weighs each position's surrogate (the importance of a token sampled by
    another implementation of the policy). Returns the loss, the weighted
    share of clipped positions, and the KL estimate."""
    ratio = torch.exp(log_probs - old)
    low, high = clip
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - low, 1 + high) * advantages)
    if correction is not None:
        surrogate = surrogate * correction
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
    drop_uniform: bool = False,
    correction_cap: float | None = None,
    checkpoint: str | Path | None = None,
    checkpoint_every: int | None = None,
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

    `drop_uniform` leaves out the groups whose rewards are all equal: they
    have no advantage, and the loss is then the mean over the tokens that
    have one. `correction_cap` corrects for the engine sampling from a
    slightly different policy than the one trained (another kernel, another
    precision): each token's surrogate is weighed by the policy's
    probability of it over the engine's, at most the cap (truncated
    importance sampling). It reads the engine's log-probabilities, so it
    takes `temperature` 1 and no `top_k` or `top_p`.

    With `checkpoint`, a directory, training resumes from the latest
    checkpoint there, skipping the prompts and seeds its steps took,
    and writes one every `checkpoint_every` steps and at the end
    (`linnet.train.save_checkpoint`)."""
    import torch.distributed as dist

    if beta and reference is None:
        raise ValueError("a KL penalty (`beta`) needs a `reference` model")
    if correction_cap is not None and (temperature != 1.0 or top_k or top_p < 1.0):
        raise ValueError(
            "the engine's log-probabilities are of the model's own distribution: correcting "
            "for them takes temperature 1 and no top_k or top_p"
        )
    device = next(policy.parameters()).device
    if getattr(policy, "shard_group", None) is not None:
        raise ValueError(
            "grpo trains a policy whole on each process or split by `fully_shard`, "
            "not by `tensor_parallel`"
        )
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    trained = [p for group_ in optimizer.param_groups for p in group_["params"]]
    draws = random.Random(seed)
    stop = frozenset(eos)
    source = iter(prompts)
    history: list[GrpoStep] = []
    start = 0
    if checkpoint is not None:
        start = load_checkpoint(checkpoint, policy, optimizer, schedule=schedule)
        for _ in range(start):
            for _ in itertools.islice(source, prompts_per_step):
                for _ in range(group):
                    draws.getrandbits(32)
    for step in range(start + 1, steps + 1):
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
                logprobs=0 if correction_cap is not None else None,
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
        groups = [rewards[i : i + group] for i in range(0, len(rewards), group)]
        uniform = [max(g) == min(g) for g in groups]
        kept = [i for i in range(len(completions)) if not (drop_uniform and uniform[i // group])]
        examples = [
            Example.prompted(completions[i].request.prompt, completions[i].tokens) for i in kept
        ]
        advantages = [advantages[i] for i in kept]
        sampled_by = (
            [(len(completions[i].request.prompt), completions[i].logprobs) for i in kept]
            if correction_cap is not None
            else None
        )
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
        prepared = [_Prepared(batch, advantages, count, device, sampled_by) for batch in batches]
        if iterations > 1 or reference is not None:
            with torch.no_grad():
                for item in prepared:
                    if iterations > 1:
                        item.old = policy.run_entry(entry, item.inputs).detach()
                    if reference is not None and beta:
                        item.reference = reference.run_entry(entry, item.inputs).detach()

        loss_total = clipped_total = mismatch = 0.0
        kl_total: float | None = None
        norm: float | None = None
        for _ in range(iterations):
            loss_total = clipped_total = mismatch = 0.0
            kl_total = None
            for item in prepared:
                log_probs = policy.run_entry(entry, item.inputs)
                old = item.old if item.old is not None else log_probs.detach()
                correction = None
                if item.engine is not None and correction_cap is not None:
                    correction = torch.exp(old - item.engine).clamp(max=correction_cap)
                    mismatch += float(((old - item.engine).abs() * item.weights).sum())
                loss, share, kl = grpo_loss(
                    log_probs,
                    old,
                    item.advantages,
                    item.weights,
                    clip=clip,
                    reference=item.reference,
                    beta=beta,
                    correction=correction,
                )
                loss.backward()  # pyright: ignore[reportUnknownMemberType]
                loss_total += float(loss.detach())
                clipped_total += float(share)
                if kl is not None:
                    kl_total = (kl_total or 0.0) + float(kl)
            if distributed:
                reduce_gradients(trained)
                sums = torch.tensor(
                    [loss_total, clipped_total, kl_total or 0.0, mismatch],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(sums)
                loss_total, clipped_total = float(sums[0]), float(sums[1])
                kl_total = float(sums[2]) if kl_total is not None else None
                mismatch = float(sums[3])
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
            float(sum(uniform)),
            float(len(groups)),
        ]
        if distributed:
            summed = torch.tensor(moments, dtype=torch.float64, device=device)
            dist.all_reduce(summed)
            moments = [float(value) for value in summed]
        total, squares, n, generated, flat, n_groups = moments
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
            uniform=flat / n_groups,
            mismatch=mismatch if correction_cap is not None else None,
        )
        history.append(record)
        if on_step is not None:
            on_step(record)
        if checkpoint is not None and checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(checkpoint, policy, optimizer, step=step, schedule=schedule)
    last = history[-1].step if history else 0
    if (
        checkpoint is not None
        and history
        and not (checkpoint_every and last % checkpoint_every == 0)
    ):
        save_checkpoint(checkpoint, policy, optimizer, step=last, schedule=schedule)
    return history


def _prompt(value: Prompt | Sequence[int]) -> Prompt:
    return value if isinstance(value, Prompt) else Prompt(value)


def _empty(size: int) -> Batch:
    """A batch that learns nothing, run so every process makes the same calls."""
    zeros = torch.zeros(size, dtype=torch.int32)
    return Batch(zeros, zeros, zeros, zeros.long(), torch.zeros(size))


class _Prepared:
    """A batch's inputs on the device, its weights (its learned positions
    over the step's count), each position's advantage, and with `sampled_by`
    (each example's prompt length and the engine's log-probabilities of its
    completion) the engine's log-probability of each position's target."""

    def __init__(
        self,
        batch: Batch,
        advantages: list[float],
        count: float,
        device: torch.device,
        sampled_by: list[tuple[int, list[float]]] | None = None,
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
        self.engine: torch.Tensor | None = None
        if sampled_by is not None:
            # Completion token k is the target of the position before it:
            # the prompt's length - 1 + k within its sequence.
            rows = [sampled_by[i] for i in batch.items]
            width = max([len(values) for _, values in rows] + [1])
            table = torch.zeros(len(rows) + 1, width)
            for s, (_, values) in enumerate(rows):
                table[s, : len(values)] = torch.tensor(values, dtype=torch.float32)
            lengths = torch.tensor([length for length, _ in rows] + [1])
            segments = batch.segments.long()
            k = (batch.positions.long() - (lengths[segments] - 1)).clamp(0, width - 1)
            values = torch.where(batch.mask > 0, table[segments, k], torch.zeros(()))
            self.engine = values.to(device)


__all__ = ["GrpoStep", "Prompt", "Reward", "group_advantages", "grpo", "grpo_loss"]
