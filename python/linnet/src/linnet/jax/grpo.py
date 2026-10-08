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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..packing import Prompt, Reward, empty, pack
from ..runs import GrpoStep, as_prompt, check_grpo, learned_positions, sample_requests, score
from .train import Learner, add, merge_lora, prepare_blocks


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
    mesh: Any = None,
    remat: bool = False,
    on_step: Callable[[GrpoStep], None] | None = None,
) -> tuple[dict[str, Any], list[GrpoStep]]:
    """Trains `policy` (the `log_probs_packed` entry as generated JAX) with
    GRPO for `steps` steps or until `prompts` run out; returns its
    parameters and the steps.

    `engine` samples: a `linnet.serve.Engine` over the JAX model (anything
    with `load_weights(parameters)` and `run(requests)`), which takes the
    policy's weights before each step, adapters merged in. The options are
    `linnet.train.grpo.grpo`'s; `reference` is the frozen reference's
    parameters for the KL penalty `beta`. One process.

    Over `mesh`, the policy trains fully sharded, each device on its own
    batches of the step's completions, as `linnet.jax.train.train` does;
    `remat` recomputes each layer in the backward pass. The engine takes
    the weights as they are split and places them as its model wants (a
    model split for serving over the same devices, say)."""
    check_grpo(beta, reference, correction_cap, temperature, top_k, top_p)
    draws = random.Random(seed)
    stop = frozenset(eos)
    source = iter(prompts)
    learner: Learner | None = None
    prepare_blocks(policy, mesh=mesh, remat=remat, parameters=parameters)
    if parameters is not None:
        learner = Learner(policy, dict(parameters), optimizer, trainable=trainable, mesh=mesh)
    low, high = clip

    def log_probs(values: Any, inputs: Any) -> Any:
        return policy.apply(values, *inputs)

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
        chosen = [as_prompt(p) for p in itertools.islice(source, prompts_per_step)]
        if not chosen:
            break
        begin = time.perf_counter()
        if learner is not None:
            # Copies: the engine keeps them past the next step's donation.
            engine.load_weights(merge_lora(policy, learner.parameters(copy=True)))
        requests = sample_requests(
            chosen,
            group,
            draws,
            max_new_tokens=max_new_tokens,
            eos=stop,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            logprobs=correction_cap is not None,
        )
        completions, _ = engine.run(requests)
        sampled_seconds = time.perf_counter() - begin

        begin = time.perf_counter()
        scored = score(
            chosen, completions, reward, group, scale=scale_rewards, drop_uniform=drop_uniform
        )
        rewards, uniform = scored.rewards, scored.uniform
        sampled_by = (
            [(len(completions[i].request.prompt), completions[i].logprobs) for i in scored.kept]
            if correction_cap is not None
            else None
        )
        batches = list(pack(scored.examples, tokens))
        count = max(1.0, float(sum(batch.count for batch in batches)))
        if learner is None:
            # The loaded weights, once the entry has compiled for the shape.
            first = batches[0] if batches else None
            if first is None:
                continue
            weights = policy.parameters_for(*first.arrays(1.0)[:4])
            learner = Learner(policy, dict(weights), optimizer, trainable=trainable, mesh=mesh)
        if gradient is None:
            gradient = learner.gradient(loss)
        # A device's share of each call: its batch's inputs, weights,
        # advantages and engine log-probabilities; batches that learn
        # nothing fill the last call.
        width = learner.width
        padded = batches + [empty(tokens) for _ in range(-len(batches) % width)]
        prepared = [
            (
                batch.arrays(1.0)[:4],
                *learned_positions(
                    batch.items,
                    batch.segments,
                    batch.positions,
                    batch.mask,
                    scored.advantages,
                    count,
                    sampled_by,
                ),
            )
            for batch in padded
        ]
        calls = [prepared[i : i + width] for i in range(0, len(prepared), width)]

        olds: list[list[Any] | None] = [None] * len(calls)
        if iterations > 1:
            own = learner.forward(log_probs)
            olds = [_per_device(learner, own, call) for call in calls]
        theirs: list[list[Any] | None] = [None] * len(calls)
        if beta and reference is not None:
            frozen = learner.forward(log_probs, reference)
            theirs = [_per_device(learner, frozen, call) for call in calls]
        loss_total = norm = 0.0
        moments = np.zeros(3)
        for _ in range(iterations):
            loss_total = 0.0
            moments = np.zeros(3)
            grads: Any = None
            for call, old, other in zip(calls, olds, theirs, strict=True):
                rows: list[tuple[Any, ...]] = []
                for k, (inputs, weights_, advantage, sampled) in enumerate(call):
                    zero = np.zeros_like(weights_)
                    rows.append(
                        (
                            inputs,
                            weights_,
                            advantage,
                            zero if old is None else old[k],
                            zero if sampled is None else sampled,
                            zero if other is None else other[k],
                        )
                    )
                (value, found), more = gradient(
                    learner.trained, learner.frozen, *learner.stack(rows)
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
            learning_rate=None,
            grad_norm=norm,
            uniform=sum(uniform) / len(uniform),
            mismatch=float(moments[2]) if correction_cap is not None else None,
        )
        history.append(record)
        if on_step is not None:
            on_step(record)
    return (learner.parameters() if learner is not None else dict(parameters or {})), history


def _per_device(learner: Learner, function: Callable[..., Any], call: list[Any]) -> list[Any]:
    """`function` (a `Learner.forward`) of each device's batch inputs in
    `call`, one row per device."""
    out = np.asarray(function(*learner.stack([(item[0],) for item in call])))
    return list(out.reshape(len(call), *out.shape[-1:]))


__all__ = ["GrpoStep", "grpo"]
