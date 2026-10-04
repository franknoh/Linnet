"""Training a Linnet model in JAX: packed batches, an optax optimizer, and
one compiled step.

```python
import optax
from linnet import nest
from linnet.jax.train import train
from linnet.packing import Example, pack

model = nest.load("llama-3.1-8b-instruct", backend="jax_source", entry="loss_packed",
                  generics={"Batch": 1, "MaxSeq": 4096, "T": "bf16"}, cast_dtype=True)
examples = [Example.prompted(prompt, completion) for prompt, completion in data]
params, history = train(model, pack(examples, tokens=4096), optimizer=optax.adamw(1e-5),
                        steps=1000, accumulate=8)
```

`model` is the entry `loss_packed` as generated JAX (`load_source`, or
`nest.load(..., backend="jax_source")`): `(tokens [P] i32, positions [P]
i32, segments [P] i32, targets [P] i64, weights [P] f32) -> f32`, the
weighted sum of each position's cross-entropy for its target.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from __future__ import annotations

import fnmatch
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..packing import Packed


@dataclass
class Step:
    """What one optimizer step did."""

    step: int
    loss: float  # the mean over the step's counted positions
    tokens: int  # positions processed, padding included
    seconds: float
    grad_norm: float | None = None


@dataclass
class History:
    steps: list[Step] = field(default_factory=lambda: list[Step]())

    @property
    def losses(self) -> list[float]:
        return [step.loss for step in self.steps]


def train(
    model: Any,
    batches: Iterable[Packed],
    *,
    optimizer: Any,
    steps: int | None = None,
    accumulate: int = 1,
    clip: float | None = 1.0,
    trainable: bool | str | Sequence[str] = True,
    master: Any = jnp.float32,
    parameters: dict[str, Any] | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> tuple[dict[str, Any], History]:
    """Fits `model`'s parameters on `batches` (`linnet.packing.pack`) with
    `optimizer` (an optax transformation); returns them, path -> array, with
    the steps taken.

    Each step sums `accumulate` batches' gradients, every batch's loss
    weighed by the step's total count of learned positions, so the step
    follows the mean over all of them; the batches run in one compiled
    step, a `lax.scan` over them. `clip` rescales the gradients to that
    global norm at most. It stops after `steps` steps or when the batches
    run out.

    `trainable` picks the floating-point parameters that train: all, or
    those whose path matches a glob pattern (`"layers.*.mlp.*"`). They are
    kept in `master` (f32) and cast to the dtype the model computes in on
    every call, so gradients and optimizer state are in f32; the rest stay
    as loaded. Paths bound to one checkpoint tensor (a tied embedding and
    output head) are one parameter, their gradients summed. `parameters`
    starts from given arrays instead of the loaded weights."""
    iterator = iter(batches)
    group = [batch for _, batch in zip(range(accumulate), iterator, strict=False)]
    if not group:
        return dict(parameters or {}), History()
    weights = (
        dict(parameters) if parameters is not None else model.parameters_for(*group[0].arrays(1.0))
    )
    ties = _ties(weights)
    chosen = _chosen([path for path in weights if path not in ties], weights, trainable)
    dtypes = {path: weights[path].dtype for path in chosen}
    # Copies: the step donates them, and the model's own arrays must stay.
    trained = {
        path: jnp.array(weights[path], dtype=master or weights[path].dtype, copy=True)
        for path in chosen
    }
    frozen = {path: weights[path] for path in weights if path not in chosen and path not in ties}
    state = optimizer.init(trained)

    def loss_of(trained: Any, frozen: Any, inputs: Any) -> Any:
        values = {**frozen, **{p: v.astype(dtypes[p]) for p, v in trained.items()}}
        values.update({path: values[tie] for path, tie in ties.items()})
        return model.apply(values, *inputs)

    def update(trained: Any, state: Any, frozen: Any, stacked: Any) -> Any:
        def one(carry: Any, inputs: Any) -> Any:
            grads, total = carry
            loss, more = jax.value_and_grad(loss_of)(trained, frozen, inputs)
            return (jax.tree.map(jnp.add, grads, more), total + loss), None

        zeros = jax.tree.map(jnp.zeros_like, trained)
        (grads, loss), _ = jax.lax.scan(one, (zeros, jnp.zeros((), jnp.float32)), stacked)
        norm = _global_norm(grads)
        if clip is not None:
            scale = jnp.minimum(1.0, clip / (norm + 1e-6))
            grads = jax.tree.map(lambda g: g * scale.astype(g.dtype), grads)
        updates, state = optimizer.update(grads, state, trained)
        trained = jax.tree.map(lambda p, u: p + u.astype(p.dtype), trained, updates)
        return trained, state, loss, norm

    compiled = jax.jit(update, donate_argnums=(0, 1))
    history = History()
    while group and (steps is None or len(history.steps) < steps):
        count = max(1, sum(batch.count for batch in group))
        stacked = [
            np.stack(values) for values in zip(*(b.arrays(count) for b in group), strict=True)
        ]
        begin = time.perf_counter()
        trained, state, loss, norm = compiled(trained, state, frozen, stacked)
        record = Step(
            step=len(history.steps) + 1,
            loss=float(loss),
            tokens=sum(batch.tokens.size for batch in group),
            seconds=time.perf_counter() - begin,
            grad_norm=float(norm),
        )
        history.steps.append(record)
        if on_step is not None:
            on_step(record)
        group = [batch for _, batch in zip(range(accumulate), iterator, strict=False)]
        # A last group short of `accumulate` would compile the step again.
        if len(group) < accumulate:
            break
    final = {**frozen, **trained}
    final.update({path: final[tie] for path, tie in ties.items()})
    return final, history


def _ties(weights: dict[str, Any]) -> dict[str, str]:
    """Each path whose array another path already holds, to that path."""
    first: dict[int, str] = {}
    ties: dict[str, str] = {}
    for path, array in weights.items():
        owner = first.setdefault(id(array), path)
        if owner != path:
            ties[path] = owner
    return ties


def _chosen(
    paths: list[str], weights: dict[str, Any], trainable: bool | str | Sequence[str]
) -> list[str]:
    floating = [p for p in paths if jnp.issubdtype(weights[p].dtype, jnp.floating)]
    if trainable is True:
        return floating
    if trainable is False:
        return []
    patterns = [trainable] if isinstance(trainable, str) else list(trainable)
    return [p for p in floating if any(fnmatch.fnmatchcase(p, pattern) for pattern in patterns)]


def _global_norm(tree: Any) -> Any:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.zeros((), jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves))


__all__ = ["History", "Step", "train"]
