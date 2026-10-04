"""Direct preference optimization (DPO) in JAX, as `linnet.train.dpo` does
it in PyTorch: each pair's answers packed into one batch, the model's
`log_probs_packed` entry summed per answer, and
`-log sigmoid(beta * ((chosen - ref_chosen) - (rejected - ref_rejected)))`
averaged over the step's pairs.

```python
import optax
from linnet.jax.dpo import dpo
from linnet.packing import Pair

model = nest.load(card, backend="jax_source", entry="log_probs_packed", ...)
params, history = dpo(model, pairs, optimizer=optax.adamw(5e-7), steps=1000)
```

Without `reference` parameters, the reference log-probabilities are the
model's own before training, computed for every pair first.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..packing import Example, Packed, Pair, empty, pack
from .train import Learner, add, prepare_blocks


@dataclass
class DpoStep:
    """What one step did, over its pairs: `accuracy` is the share whose
    chosen answer the model now rewards more, `margin` the mean difference of
    the rewards, `chosen` and `rejected` their means."""

    step: int
    loss: float
    accuracy: float
    margin: float
    chosen: float
    rejected: float
    seconds: float
    grad_norm: float | None


def dpo_loss(
    chosen: Any,
    rejected: Any,
    reference_chosen: Any,
    reference_rejected: Any,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[Any, Any, Any]:
    """Per pair: the loss, and the chosen and rejected answers' rewards."""
    good = beta * (chosen - reference_chosen)
    bad = beta * (rejected - reference_rejected)
    logits = good - bad
    losses = (
        -jax.nn.log_sigmoid(logits) * (1 - label_smoothing)
        - jax.nn.log_sigmoid(-logits) * label_smoothing
    )
    return losses, jax.lax.stop_gradient(good), jax.lax.stop_gradient(bad)


def dpo(
    model: Any,
    pairs: Sequence[Pair],
    *,
    optimizer: Any,
    reference: dict[str, Any] | None = None,
    steps: int | None = None,
    pairs_per_step: int = 32,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    tokens: int = 4096,
    clip: float | None = 1.0,
    trainable: bool | str | Sequence[str] | None = None,
    parameters: dict[str, Any] | None = None,
    mesh: Any = None,
    remat: bool = False,
    on_step: Callable[[DpoStep], None] | None = None,
) -> tuple[dict[str, Any], list[DpoStep]]:
    """Trains `model` (the `log_probs_packed` entry as generated JAX) on
    preference `pairs`, `pairs_per_step` an optimizer step, for `steps` steps
    or until they run out; returns the parameters and the steps.

    `reference` is the frozen reference's parameters (path -> array); without
    them the model's own log-probabilities before training are computed for
    every pair first. Answers are packed into batches of `tokens` positions,
    each pair in one batch. `clip` caps the gradient norm; `trainable` picks
    what trains as `linnet.jax.train.train` does.

    Over `mesh`, each device takes its own batches and the parameters are
    fully sharded, as `linnet.jax.train.train` does; `remat` recomputes
    each layer in the backward pass. Set both before the model's first
    call."""
    rounds = [list(pairs[i : i + pairs_per_step]) for i in range(0, len(pairs), pairs_per_step)]
    if steps is not None:
        rounds = rounds[:steps]
    if not rounds:
        return dict(parameters or {}), []
    first = _batches(rounds[0], tokens)
    prepare_blocks(model, mesh=mesh, remat=remat, parameters=parameters)
    weights = (
        parameters if parameters is not None else model.parameters_for(*first[0].arrays(1.0)[:4])
    )
    learner = Learner(model, dict(weights), optimizer, trainable=trainable, mesh=mesh)

    def answers(values: Any, inputs: Any) -> Any:
        tokens_, positions, segments, targets, mask = inputs
        per_token = model.apply(values, tokens_, positions, segments, targets)
        sums = jax.ops.segment_sum(per_token * mask, segments, num_segments=tokens + 1)
        return sums

    sums_of = learner.forward(answers, reference)
    known = _reference(sums_of, learner, rounds, tokens, pairs_per_step)

    def loss(
        values: Any, inputs: Any, chosen: Any, rejected: Any, valid: Any, refs: Any, total: Any
    ) -> Any:
        sums = answers(values, inputs)
        losses, good, bad = dpo_loss(
            sums[chosen],
            sums[rejected],
            refs[:, 0],
            refs[:, 1],
            beta=beta,
            label_smoothing=label_smoothing,
        )
        moments = jnp.stack(
            [
                jnp.sum(valid * losses),
                jnp.sum(valid * (good > bad)),
                jnp.sum(valid * (good - bad)),
                jnp.sum(valid * good),
                jnp.sum(valid * bad),
            ]
        )
        return jnp.sum(valid * losses) / total, moments

    gradient = learner.gradient(loss)
    history: list[DpoStep] = []
    for number, chosen_pairs in enumerate(rounds):
        begin = time.perf_counter()
        batches = first if number == 0 else _batches(chosen_pairs, tokens)
        total = float(len(chosen_pairs))
        grads: Any = None
        moments = np.zeros(5)
        for chunk in _chunks(batches, learner.width, tokens):
            rows: list[tuple[Any, ...]] = []
            for batch in chunk:
                inputs = [*batch.arrays(1.0)[:4], batch.mask]
                chosen, rejected, valid, which = _paired(batch, pairs_per_step)
                refs = np.zeros((pairs_per_step, 2), np.float32)
                for slot, k in enumerate(which):
                    refs[slot] = known[number * pairs_per_step + k]
                rows.append((inputs, chosen, rejected, valid, refs, np.float32(total)))
            (_, found), more = gradient(learner.trained, learner.frozen, *learner.stack(rows))
            grads = more if grads is None else add(grads, more)
            moments += np.asarray(found)
        norm = learner.step(grads, clip)
        loss_sum, right, margin, good, bad = (float(v) / total for v in moments)
        record = DpoStep(
            step=number + 1,
            loss=loss_sum,
            accuracy=right,
            margin=margin,
            chosen=good,
            rejected=bad,
            seconds=time.perf_counter() - begin,
            grad_norm=norm,
        )
        history.append(record)
        if on_step is not None:
            on_step(record)
    return learner.parameters(), history


def _batches(chosen_pairs: list[Pair], tokens: int) -> list[Packed]:
    examples = [
        Example.prompted(pair.prompt, answer)
        for pair in chosen_pairs
        for answer in (pair.chosen, pair.rejected)
    ]
    return list(pack(examples, tokens, together=2))


def _paired(batch: Packed, width: int) -> tuple[Any, Any, Any, list[int]]:
    """The batch's chosen and rejected answers' sequences and their pairs,
    `width` long (unused ones marked invalid)."""
    first = [s for s, item in enumerate(batch.items) if item % 2 == 0]
    which = [batch.items[s] // 2 for s in first]
    chosen = np.zeros(width, np.int32)
    rejected = np.zeros(width, np.int32)
    valid = np.zeros(width, np.float32)
    chosen[: len(first)] = first
    rejected[: len(first)] = [s + 1 for s in first]
    valid[: len(first)] = 1.0
    return chosen, rejected, valid, which


def _chunks(batches: list[Packed], width: int, tokens: int) -> list[list[Packed]]:
    """`batches` `width` at a time, one per device, the last run out with
    batches that learn nothing."""
    out = [batches[i : i + width] for i in range(0, len(batches), width)]
    if out and len(out[-1]) < width:
        out[-1] = out[-1] + [empty(tokens) for _ in range(width - len(out[-1]))]
    return out


def _reference(
    sums: Callable[..., Any],
    learner: Learner,
    rounds: list[list[Pair]],
    tokens: int,
    width: int,
) -> list[tuple[float, float]]:
    """Every pair's answers' summed log-probabilities under the reference."""
    known: list[tuple[float, float]] = []
    for chosen_pairs in rounds:
        found: dict[int, tuple[float, float]] = {}
        for chunk in _chunks(_batches(chosen_pairs, tokens), learner.width, tokens):
            rows = [([*batch.arrays(1.0)[:4], batch.mask],) for batch in chunk]
            values = np.asarray(sums(*learner.stack(rows))).reshape(len(chunk), -1)
            for batch, row in zip(chunk, values, strict=True):
                chosen, rejected, _, which = _paired(batch, width)
                for slot, k in enumerate(which):
                    found[k] = (float(row[chosen[slot]]), float(row[rejected[slot]]))
        known += [found[k] for k in range(len(chosen_pairs))]
        known += [(0.0, 0.0)] * (width - len(chosen_pairs))
    return known


__all__ = ["DpoStep", "dpo", "dpo_loss"]
