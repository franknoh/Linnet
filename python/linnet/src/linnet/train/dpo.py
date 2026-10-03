"""Direct preference optimization (DPO): learning which of two answers to a
prompt is preferred, against a frozen reference.

Each pair's answers are packed into one batch; the model's
`log_probs_packed` entry gives every answer token's log-probability, summed
per answer. The loss is `-log sigmoid(beta * ((chosen - ref_chosen) -
(rejected - ref_rejected)))`, the mean over the step's pairs:

```python
from linnet.train.dpo import Pair, dpo

pairs = [Pair(prompt, chosen, rejected) for prompt, chosen, rejected in data]
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-7)
dpo(model, pairs, optimizer=optimizer, steps=1000, pairs_per_step=32, beta=0.1)
```

Without a `reference` model, the reference log-probabilities are the
model's own before training, computed for every pair first: a LoRA run then
needs one model.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

from . import (
    Batch,
    Example,
    clip_gradients,
    load_checkpoint,
    pack,
    reduce_gradients,
    save_checkpoint,
)


@dataclass(frozen=True)
class Pair:
    """A prompt, the answer preferred, and the one not."""

    prompt: Sequence[int]
    chosen: Sequence[int]
    rejected: Sequence[int]


@dataclass
class DpoStep:
    """What one step did, over its pairs: `accuracy` is the share whose
    chosen answer the model now rewards more, `margin` the mean difference of
    the rewards, `chosen` and `rejected` their means (a reward is `beta`
    times an answer's log-probability less the reference's)."""

    step: int
    loss: float
    accuracy: float
    margin: float
    chosen: float
    rejected: float
    seconds: float
    learning_rate: float
    grad_norm: float | None


def dpo_loss(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per pair, `[N]` each: the loss, and the chosen and rejected answers'
    rewards. `label_smoothing` is the share of pairs taken to be labelled
    the wrong way round (conservative DPO)."""
    chosen_reward = beta * (chosen - reference_chosen)
    rejected_reward = beta * (rejected - reference_rejected)
    logits = chosen_reward - rejected_reward
    losses = (
        -functional.logsigmoid(logits) * (1 - label_smoothing)
        - functional.logsigmoid(-logits) * label_smoothing
    )
    return losses, chosen_reward.detach(), rejected_reward.detach()


def dpo(
    model: Any,
    pairs: Iterable[Pair],
    *,
    optimizer: torch.optim.Optimizer,
    reference: Any = None,
    steps: int | None = None,
    pairs_per_step: int = 32,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    tokens: int = 4096,
    clip_grad: float | None = 1.0,
    schedule: Any = None,
    entry: str = "log_probs_packed",
    checkpoint: str | Path | None = None,
    checkpoint_every: int | None = None,
    on_step: Callable[[DpoStep], None] | None = None,
) -> list[DpoStep]:
    """Trains `model` on preference `pairs`, `pairs_per_step` an optimizer
    step, for `steps` steps or until they run out.

    `reference` is a frozen model with the same entry; without one, `pairs`
    must be a sequence, and the model's own log-probabilities before
    training are computed for all of them first. Answers are packed into
    batches of `tokens` positions, each pair in one batch. `clip_grad` caps
    the gradient norm; `schedule` steps after each optimizer step;
    `checkpoint` and `checkpoint_every` resume and save as `train`'s do.

    Under `torch.distributed`, each process takes its own pairs: the loss
    is the mean over every process's pairs, gradients are summed (or reduced
    into the parts of a model split by `fully_shard`), and a process with
    fewer batches runs empty ones."""
    import torch.distributed as dist

    device = next(model.parameters()).device
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    trained = [p for group in optimizer.param_groups for p in group["params"]]
    precomputed: list[tuple[float, float]] | None = None
    if reference is None:
        if not isinstance(pairs, Sequence):
            raise ValueError(
                "without a reference model, pairs must be a sequence: their reference "
                "log-probabilities are computed first"
            )
        precomputed = _reference(model, pairs, pairs_per_step, tokens, entry, device, distributed)
    source = iter(pairs)
    start = 0
    if checkpoint is not None:
        start = load_checkpoint(checkpoint, model, optimizer, schedule=schedule)
        for _ in itertools.islice(source, start * pairs_per_step):
            pass
    history: list[DpoStep] = []
    step = start
    while steps is None or step < steps:
        offset = step * pairs_per_step
        chosen_pairs = list(itertools.islice(source, pairs_per_step))
        rounds = _rounds(chosen_pairs, tokens, device, distributed)
        if rounds is None:
            break
        batches, total = rounds
        step += 1
        begin = time.perf_counter()
        sums = torch.zeros(5, dtype=torch.float64, device=device)
        for batch in batches:
            sequences = _answers(model, entry, batch, device)
            first, second, which = _paired(batch, sequences)
            if precomputed is not None:
                known = [precomputed[offset + k] for k in which]
                ref_chosen = torch.tensor([c for c, _ in known], device=device)
                ref_rejected = torch.tensor([r for _, r in known], device=device)
            else:
                with torch.no_grad():
                    theirs = _answers(reference, entry, batch, device)
                ref_chosen, ref_rejected, _ = _paired(batch, theirs)
            losses, good, bad = dpo_loss(
                first,
                second,
                ref_chosen,
                ref_rejected,
                beta=beta,
                label_smoothing=label_smoothing,
            )
            (losses.sum() / total).backward()  # pyright: ignore[reportUnknownMemberType]
            sums += torch.stack(
                [
                    losses.detach().sum().double(),
                    (good > bad).double().sum(),
                    (good - bad).double().sum(),
                    good.double().sum(),
                    bad.double().sum(),
                ]
            )
        if distributed:
            reduce_gradients(trained)
            dist.all_reduce(sums)
        norm = clip_gradients(trained, clip_grad) if clip_grad is not None else None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if schedule is not None:
            schedule.step()
        loss, right, margin, good_total, bad_total = (float(v) / total for v in sums)
        record = DpoStep(
            step=step,
            loss=loss,
            accuracy=right,
            margin=margin,
            chosen=good_total,
            rejected=bad_total,
            seconds=time.perf_counter() - begin,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            grad_norm=norm,
        )
        history.append(record)
        if on_step is not None:
            on_step(record)
        if checkpoint is not None and checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(checkpoint, model, optimizer, step=step, schedule=schedule)
    if (
        checkpoint is not None
        and history
        and not (checkpoint_every and step % checkpoint_every == 0)
    ):
        save_checkpoint(checkpoint, model, optimizer, step=step, schedule=schedule)
    return history


def _reference(
    model: Any,
    pairs: Sequence[Pair],
    per_round: int,
    tokens: int,
    entry: str,
    device: torch.device,
    distributed: bool,
) -> list[tuple[float, float]]:
    """Every pair's answers' log-probabilities under `model` as it is."""
    known: list[tuple[float, float]] = []
    for offset in itertools.count(0, per_round):
        chosen_pairs = list(pairs[offset : offset + per_round])
        rounds = _rounds(chosen_pairs, tokens, device, distributed)
        if rounds is None:
            break
        found: dict[int, tuple[float, float]] = {}
        with torch.no_grad():
            for batch in rounds[0]:
                first, second, which = _paired(batch, _answers(model, entry, batch, device))
                for k, c, r in zip(which, first.tolist(), second.tolist(), strict=True):
                    found[k] = (c, r)
        known += [found[k] for k in range(len(chosen_pairs))]
    return known


def _rounds(
    chosen_pairs: list[Pair], tokens: int, device: torch.device, distributed: bool
) -> tuple[list[Batch], float] | None:
    """The pairs' batches, as many on every process as on the one with the
    most, and the count of pairs over every process; None when any process
    has run out."""
    import torch.distributed as dist

    examples = [
        Example.prompted(pair.prompt, answer)
        for pair in chosen_pairs
        for answer in (pair.chosen, pair.rejected)
    ]
    batches = list(pack(examples, tokens, together=2))
    total = float(len(chosen_pairs))
    if distributed:
        state = torch.tensor([float(bool(chosen_pairs)), total, len(batches)], device=device)
        least = state[:1].clone()
        dist.all_reduce(least, op=dist.ReduceOp.MIN)
        if not least.item():
            return None
        summed = state[1:2].clone()
        dist.all_reduce(summed)
        most = state[2:].clone()
        dist.all_reduce(most, op=dist.ReduceOp.MAX)
        total = float(summed.item())
        empty = torch.zeros(tokens, dtype=torch.int32)
        batches += [Batch(empty, empty, empty, empty.long(), torch.zeros(tokens))] * (
            int(most.item()) - len(batches)
        )
    elif not chosen_pairs:
        return None
    return batches, total


def _answers(model: Any, entry: str, batch: Batch, device: torch.device) -> torch.Tensor:
    """Each packed answer's summed log-probability, `[sequences]`."""
    inputs = [
        value.to(device) for value in (batch.tokens, batch.positions, batch.segments, batch.targets)
    ]
    per_token = model.run_entry(entry, inputs)
    segments = inputs[2].long()
    sums = per_token.new_zeros(batch.sequences + 1)
    return sums.scatter_add(0, segments, per_token * batch.mask.to(device))[: batch.sequences]


def _paired(batch: Batch, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """The batch's chosen and rejected answers, and their pairs' indices."""
    first = [s for s, item in enumerate(batch.items) if item % 2 == 0]
    which = [batch.items[s] // 2 for s in first]
    return sequences[first], sequences[[s + 1 for s in first]], which


__all__ = ["DpoStep", "Pair", "dpo", "dpo_loss"]
