"""Group relative policy optimization (GRPO) in JAX, as `linnet.train.grpo`
does it in PyTorch: an `Engine` over the JAX model samples `group`
completions of each prompt with the policy's latest weights, `reward`
scores them, and the policy takes the clipped policy-gradient step on each
completion's advantage over its group.

```python
import optax
from linnet.jax import load_model
from linnet.jax.grpo import grpo
from linnet.packing import Prompt
from linnet.serve import Engine

policy = nest.load(card, backend="jax_source", entry="log_probs_packed", ...)
engine = Engine(load_model(...))  # the same card's serving entries
params, history = grpo(policy, engine, prompts, reward, optimizer=optax.adamw(1e-6), steps=200)
```
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..packing import Example, Packed, Prompt, Reward, group_advantages, pack
from ..serve import Request
from .train import Learner, add, merge_lora


@dataclass
class GrpoStep:
    """What one step did, as `linnet.train.grpo.GrpoStep` reports it."""

    step: int
    reward: float
    reward_std: float
    length: float
    loss: float
    clipped: float
    kl: float | None
    sample_seconds: float
    train_seconds: float
    grad_norm: float | None
    uniform: float = 0.0
    mismatch: float | None = None


def grpo(
    policy: Any,
    engine: Any,
    prompts: Iterable[Prompt | Sequence[int]],
    reward: Reward,
    *,
    optimizer: Any,
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
    reference: dict[str, Any] | None = None,
    scale_rewards: bool = True,
    clip_grad: float | None = 1.0,
    seed: int = 0,
    drop_uniform: bool = False,
    correction_cap: float | None = None,
    trainable: bool | str | Sequence[str] | None = None,
    parameters: dict[str, Any] | None = None,
    on_step: Callable[[GrpoStep], None] | None = None,
) -> tuple[dict[str, Any], list[GrpoStep]]:
    """Trains `policy` (the `log_probs_packed` entry as generated JAX) with
    GRPO for `steps` steps or until `prompts` run out; returns its
    parameters and the steps.

    `engine` samples: a `linnet.serve.Engine` over the JAX model (anything
    with `load_weights(parameters)` and `run(requests)`), which takes the
    policy's weights before each step, adapters merged in. The options are
    `linnet.train.grpo.grpo`'s; `reference` is the frozen reference's
    parameters for the KL penalty `beta`. One process."""
    if beta and reference is None:
        raise ValueError("a KL penalty (`beta`) needs `reference` parameters")
    if correction_cap is not None and (temperature != 1.0 or top_k or top_p < 1.0):
        raise ValueError(
            "the engine's log-probabilities are of the model's own distribution: correcting "
            "for them takes temperature 1 and no top_k or top_p"
        )
    draws = random.Random(seed)
    stop = frozenset(eos)
    source = iter(prompts)
    learner: Learner | None = None
    if parameters is not None:
        learner = Learner(policy, dict(parameters), optimizer, trainable=trainable)
    low, high = clip

    def log_probs(values: Any, inputs: Any) -> Any:
        return policy.apply(values, *inputs)

    forward = jax.jit(log_probs)

    def loss(
        values: Any, inputs: Any, weights: Any, advantages: Any, old: Any, sampled: Any, theirs: Any
    ) -> Any:
        mine = log_probs(values, inputs)
        if iterations == 1:
            old = jax.lax.stop_gradient(mine)
        ratio = jnp.exp(mine - old)
        surrogate = jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - low, 1 + high) * advantages)
        mismatch = jnp.zeros(())
        if correction_cap is not None:
            surrogate = surrogate * jnp.minimum(jnp.exp(old - sampled), correction_cap)
            mismatch = jnp.sum(jnp.abs(old - sampled) * weights)
        per_token = -surrogate
        kl = jnp.zeros(())
        if beta:
            delta = theirs - mine
            estimate = jnp.exp(delta) - delta - 1
            per_token = per_token + beta * estimate
            kl = jnp.sum(jax.lax.stop_gradient(estimate) * weights)
        clipped = ((ratio < 1 - low) & (advantages < 0)) | ((ratio > 1 + high) & (advantages > 0))
        share = jnp.sum(clipped * weights)
        return jnp.sum(per_token * weights), jnp.stack([share, kl, mismatch])

    gradient: Any = None
    history: list[GrpoStep] = []
    for step in range(1, steps + 1):
        chosen = [_prompt(p) for p in itertools.islice(source, prompts_per_step)]
        if not chosen:
            break
        begin = time.perf_counter()
        if learner is not None:
            engine.load_weights(merge_lora(policy, learner.parameters()))
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
        sampled_seconds = time.perf_counter() - begin

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
        batches = list(pack(examples, tokens))
        count = max(1.0, float(sum(batch.count for batch in batches)))
        if learner is None:
            # The loaded weights, once the entry has compiled for the shape.
            first = batches[0] if batches else None
            if first is None:
                continue
            weights = policy.parameters_for(*first.arrays(1.0)[:4])
            learner = Learner(policy, dict(weights), optimizer, trainable=trainable)
        if gradient is None:
            gradient = learner.gradient(loss)
        prepared = [
            _prepared(
                batch, [advantages[i] for i in kept], count, kept, completions, correction_cap
            )
            for batch in batches
        ]
        olds = [None] * len(prepared)
        if iterations > 1:
            values = learner.values(learner.trained)
            olds = [forward(values, item[0]) for item in prepared]
        theirs = [
            forward(reference, item[0]) if beta and reference is not None else None
            for item in prepared
        ]
        loss_total = norm = 0.0
        moments = np.zeros(3)
        for _ in range(iterations):
            loss_total = 0.0
            moments = np.zeros(3)
            grads: Any = None
            for (inputs, weights_, advantage, sampled), old, other in zip(
                prepared, olds, theirs, strict=True
            ):
                zero = np.zeros_like(weights_)
                (value, found), more = gradient(
                    learner.trained,
                    learner.frozen,
                    inputs,
                    weights_,
                    advantage,
                    zero if old is None else old,
                    zero if sampled is None else sampled,
                    zero if other is None else other,
                )
                grads = more if grads is None else add(grads, more)
                loss_total += float(value)
                moments += np.asarray(found)
            if grads is not None:
                norm = learner.step(grads, clip_grad)

        mean = sum(rewards) / len(rewards)
        record = GrpoStep(
            step=step,
            reward=mean,
            reward_std=math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards)),
            length=sum(len(c.tokens) for c in completions) / len(completions),
            loss=loss_total,
            clipped=float(moments[0]),
            kl=float(moments[1]) if beta else None,
            sample_seconds=sampled_seconds,
            train_seconds=time.perf_counter() - begin,
            grad_norm=norm,
            uniform=sum(uniform) / len(uniform),
            mismatch=float(moments[2]) if correction_cap is not None else None,
        )
        history.append(record)
        if on_step is not None:
            on_step(record)
    return (learner.parameters() if learner is not None else dict(parameters or {})), history


def _prompt(value: Prompt | Sequence[int]) -> Prompt:
    return value if isinstance(value, Prompt) else Prompt(value)


def _prepared(
    batch: Packed,
    advantages: list[float],
    count: float,
    kept: list[int],
    completions: list[Any],
    cap: float | None,
) -> tuple[list[Any], Any, Any, Any]:
    """A batch's inputs, weights (its learned positions over the step's
    count), each position's advantage, and with `cap` the engine's
    log-probability of each position's target."""
    inputs = batch.arrays(1.0)[:4]
    weights = (batch.mask / count).astype(np.float32)
    per_sequence = np.asarray([advantages[i] for i in batch.items] + [0.0], dtype=np.float32)
    advantage = per_sequence[batch.segments]
    sampled = None
    if cap is not None:
        # Completion token k is the target of the position before it.
        rows = [
            (len(completions[kept[i]].request.prompt), completions[kept[i]].logprobs)
            for i in batch.items
        ]
        width = max([len(values) for _, values in rows] + [1])
        table = np.zeros((len(rows) + 1, width), np.float32)
        for s, (_, values) in enumerate(rows):
            table[s, : len(values)] = values
        lengths = np.asarray([length for length, _ in rows] + [1])
        k = np.clip(batch.positions - (lengths[batch.segments] - 1), 0, width - 1)
        sampled = np.where(batch.mask > 0, table[batch.segments, k], 0.0).astype(np.float32)
    return inputs, weights, advantage, sampled


__all__ = ["GrpoStep", "grpo"]
