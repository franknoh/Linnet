"""Supervised fine-tuning over packed sequences.

`pack` packs examples (a sequence of tokens and which of them to learn) into
batches of a fixed number of positions, and `train` fits a model's
`loss_packed` entry on them:

```python
from linnet import nest
from linnet.train import Example, pack, train

model = nest.load("llama-3.1-8b-instruct", backend="torch", device="cuda",
                  compile="inductor", trainable=True)
examples = [Example.prompted(prompt, completion) for prompt, completion in data]
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
train(model, pack(examples, tokens=4096), optimizer=optimizer, steps=1000, accumulate=8)
```

A model trains through any entry with `loss_packed`'s signature:
`(tokens [P] i32, positions [P] i32, segments [P] i32, targets [P] i64,
weights [P] f32) -> f32`, the weighted sum of each position's
cross-entropy for its target. Each Nest decoder card and
`examples/05-llama` have one.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Example:
    """One training sequence: `tokens`, and which of them the model learns
    to predict (`learned[i]` for `tokens[i]`; every token after the first
    when omitted)."""

    tokens: Sequence[int]
    learned: Sequence[bool] | None = None

    @staticmethod
    def prompted(prompt: Sequence[int], completion: Sequence[int]) -> Example:
        """A completion to learn after a prompt that is only read."""
        return Example([*prompt, *completion], [False] * len(prompt) + [True] * len(completion))


@dataclass
class Batch:
    """Sequences packed into `P` positions: token `p` is at `positions[p]` of
    sequence `segments[p]` and predicts `targets[p]` (the next token of its
    sequence) where `mask[p]` is 1. Padding is one more sequence, masked
    out."""

    tokens: torch.Tensor  # [P] i32
    positions: torch.Tensor  # [P] i32
    segments: torch.Tensor  # [P] i32
    targets: torch.Tensor  # [P] i64
    mask: torch.Tensor  # [P] f32
    sequences: int = 0

    @property
    def count(self) -> int:
        """The positions that count toward the loss."""
        return int(self.mask.sum().item())

    def inputs(self, count: float, device: torch.device | str | None = None) -> list[torch.Tensor]:
        """`loss_packed`'s inputs, the loss weighed to a mean over `count`
        positions (this batch's, or every micro-batch's of one step)."""
        values = [self.tokens, self.positions, self.segments, self.targets, self.mask / count]
        return [value.to(device) for value in values] if device is not None else values


def pack(
    examples: Iterable[Example],
    tokens: int,
    *,
    max_length: int | None = None,
) -> Iterator[Batch]:
    """Packs `examples` in order into batches of exactly `tokens` positions,
    a sequence never split across two. A sequence longer than `max_length`
    (`tokens` when omitted) is cut to it. Each batch's unused positions are
    padding, so every batch has one shape and the model compiles once."""
    limit = min(tokens, max_length) if max_length is not None else tokens
    pending: list[tuple[list[int], list[bool]]] = []
    used = 0
    for example in examples:
        sequence = list(example.tokens)[:limit]
        if len(sequence) < 2:
            continue
        learned = (
            list(example.learned)[: len(sequence)]
            if example.learned is not None
            else [False] + [True] * (len(sequence) - 1)
        )
        if used + len(sequence) > tokens:
            yield _batch(pending, tokens)
            pending, used = [], 0
        pending.append((sequence, learned))
        used += len(sequence)
    if pending:
        yield _batch(pending, tokens)


def _batch(sequences: list[tuple[list[int], list[bool]]], size: int) -> Batch:
    tokens: list[int] = []
    positions: list[int] = []
    segments: list[int] = []
    targets: list[int] = []
    mask: list[float] = []
    for index, (sequence, learned) in enumerate(sequences):
        tokens += sequence
        positions += range(len(sequence))
        segments += [index] * len(sequence)
        # Position `i` predicts token `i + 1`; the last predicts nothing.
        targets += [*sequence[1:], 0]
        mask += [float(flag) for flag in learned[1:]] + [0.0]
    padding = size - len(tokens)
    tokens += [0] * padding
    positions += [0] * padding
    segments += [len(sequences)] * padding
    targets += [0] * padding
    mask += [0.0] * padding
    return Batch(
        torch.tensor(tokens, dtype=torch.int32),
        torch.tensor(positions, dtype=torch.int32),
        torch.tensor(segments, dtype=torch.int32),
        torch.tensor(targets, dtype=torch.int64),
        torch.tensor(mask, dtype=torch.float32),
        len(sequences),
    )


def cosine_schedule(
    optimizer: torch.optim.Optimizer, warmup: int, total: int, floor: float = 0.1
) -> torch.optim.lr_scheduler.LambdaLR:
    """A linear warmup over `warmup` steps, then a cosine decay to `floor`
    times the learning rate at step `total`."""

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@dataclass
class Step:
    """What one optimizer step did."""

    step: int
    loss: float  # the mean over the step's counted positions
    tokens: int  # positions processed, padding included
    seconds: float
    learning_rate: float
    grad_norm: float | None = None


@dataclass
class History:
    steps: list[Step] = field(default_factory=lambda: list[Step]())

    @property
    def losses(self) -> list[float]:
        return [step.loss for step in self.steps]


def train(
    model: Any,
    batches: Iterable[Batch],
    *,
    optimizer: torch.optim.Optimizer,
    steps: int | None = None,
    accumulate: int = 1,
    clip: float | None = 1.0,
    schedule: Any = None,
    entry: str = "loss_packed",
    device: torch.device | str | None = None,
    save_every: int | None = None,
    save_to: str | Path | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> History:
    """Fits `model` on `batches` through its `entry` (`loss_packed`).

    Each optimizer step sums `accumulate` batches' gradients, every batch's
    loss weighed by the step's total count of learned positions, so the step
    follows the mean over all of them. `clip` rescales the gradients to that
    norm at most; `schedule` (a learning-rate scheduler) steps after each
    optimizer step. It stops after `steps` steps or when the batches run
    out. Every `save_every` steps, and at the end, it writes the weights to
    `save_to` (the adapters alone when the model has them).

    Under `torch.distributed`, every process trains a copy of the model on
    its own batches: the count of learned positions and the gradients are
    summed across processes before each step, so every copy takes the same
    step, the mean over every process's batches. Pass a
    `torch.distributed.optim.ZeroRedundancyOptimizer` to split the optimizer
    state across them. Only the first process saves; all stop when any runs
    out of batches."""
    import torch.distributed as dist

    if device is None:
        device = next(model.parameters()).device
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    trained = [p for group in optimizer.param_groups for p in group["params"]]
    history = History()
    iterator = iter(batches)
    while steps is None or len(history.steps) < steps:
        group = [batch for _, batch in zip(range(accumulate), iterator, strict=False)]
        local = float(sum(batch.count for batch in group))
        if distributed:
            # Every process steps together: any out of batches stops all.
            ready = torch.tensor([float(bool(group))], dtype=torch.float64, device=device)
            summed = torch.tensor([local], dtype=torch.float64, device=device)
            dist.all_reduce(ready, op=dist.ReduceOp.MIN)
            dist.all_reduce(summed)
            if not ready.item():
                break
            local = float(summed.item())
        elif not group:
            break
        count = max(1.0, local)
        begin = time.perf_counter()
        total = torch.zeros((), dtype=torch.float64, device=device)
        for batch in group:
            loss = model.run_entry(entry, batch.inputs(count, device))
            loss.backward()
            total += loss.detach().double()
        if distributed:
            _all_reduce_gradients(trained)
            dist.all_reduce(total)
        norm = float(torch.nn.utils.clip_grad_norm_(trained, clip)) if clip is not None else None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if schedule is not None:
            schedule.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record = Step(
            step=len(history.steps) + 1,
            loss=float(total.item()),
            tokens=sum(batch.tokens.numel() for batch in group),
            seconds=time.perf_counter() - begin,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            grad_norm=norm,
        )
        history.steps.append(record)
        if on_step is not None:
            on_step(record)
        first = not distributed or dist.get_rank() == 0
        if first and save_to is not None and save_every and record.step % save_every == 0:
            _save(model, Path(save_to))
    if (not distributed or dist.get_rank() == 0) and save_to is not None and history.steps:
        _save(model, Path(save_to))
    return history


def _all_reduce_gradients(parameters: list[torch.Tensor]) -> None:
    """Sums every gradient across processes: one collective per tensor, all
    in flight at once, so no buffer the size of the model is made."""
    import torch.distributed as dist

    pending = [
        dist.all_reduce(parameter.grad, async_op=True)
        for parameter in parameters
        if parameter.grad is not None
    ]
    for work in pending:
        work.wait()


def _save(model: Any, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if getattr(model, "lora", None) is not None:
        model.save_weights(
            directory / "adapters.safetensors", names="linnet", include=["*.lora_a", "*.lora_b"]
        )
    else:
        model.save_weights(directory / "model.safetensors")


__all__ = ["Batch", "Example", "History", "Step", "cosine_schedule", "pack", "train"]
