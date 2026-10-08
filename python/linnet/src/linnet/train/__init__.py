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
`examples/01-llama` have one.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch

from .. import lora, packing
from ..packing import Example
from ..runs import History, Step, unsaved
from ..torch.fsdp import sharded
from .checkpoints import load_checkpoint, save_checkpoint


@dataclass
class Batch:
    """Sequences packed into `P` positions: token `p` is at `positions[p]` of
    sequence `segments[p]` and predicts `targets[p]` (the next token of its
    sequence) where `mask[p]` is 1. Padding is one more sequence, masked
    out. `items[s]` is sequence `s`'s index among the examples packed."""

    tokens: torch.Tensor  # [P] i32
    positions: torch.Tensor  # [P] i32
    segments: torch.Tensor  # [P] i32
    targets: torch.Tensor  # [P] i64
    mask: torch.Tensor  # [P] f32
    items: list[int] = field(default_factory=lambda: list[int]())

    @property
    def sequences(self) -> int:
        return len(self.items)

    @property
    def count(self) -> int:
        """The positions that count toward the loss."""
        return int(self.mask.sum().item())

    @staticmethod
    def of(packed: packing.Packed) -> Batch:
        """The packed arrays as tensors."""
        return Batch(
            torch.as_tensor(packed.tokens),
            torch.as_tensor(packed.positions),
            torch.as_tensor(packed.segments),
            torch.as_tensor(packed.targets),
            torch.as_tensor(packed.mask),
            list(packed.items),
        )

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
    together: int = 1,
) -> Iterator[Batch]:
    """`linnet.packing.pack` as PyTorch tensors: examples packed in order
    into batches of exactly `tokens` positions, a sequence never split
    across two. See `linnet.packing.pack` for `max_length` and `together`."""
    for packed in packing.pack(examples, tokens, max_length=max_length, together=together):
        yield Batch.of(packed)


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
    checkpoint: str | Path | None = None,
    checkpoint_every: int | None = None,
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
    out of batches.

    A model split across processes (`load(tensor_parallel=...)`) is one copy
    over its group: those processes take the same batches, and the sums run
    across the copies, one process of each group to a sum. Its gradient's
    norm adds the split parameters' parts across the group.

    With `checkpoint`, a directory, training resumes from the latest
    checkpoint there (`load_checkpoint`), skipping the batches its steps
    took, and `steps` counts from the start of the run. A checkpoint is
    written every `checkpoint_every` steps and at the end
    (`save_checkpoint`)."""
    import torch.distributed as dist

    if device is None:
        device = next(model.parameters()).device
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    trained = [p for group in optimizer.param_groups for p in group["params"]]
    layout = copies(model) if distributed else Copies()
    # One process saves; a sharded model is gathered to be saved, by every one.
    saves = not distributed or dist.get_rank() == 0 or bool(getattr(model, "fully_sharded", ()))
    history = History()
    iterator = iter(batches)
    start = 0
    if checkpoint is not None:
        start = load_checkpoint(checkpoint, model, optimizer, schedule=schedule)
        for _ in range(start * accumulate):
            next(iterator, None)
    while steps is None or start + len(history.steps) < steps:
        group = [batch for _, batch in zip(range(accumulate), iterator, strict=False)]
        local = float(sum(batch.count for batch in group))
        if distributed:
            # Every process steps together: any out of batches stops all.
            ready = torch.tensor([float(bool(group))], dtype=torch.float64, device=device)
            summed = torch.tensor([local], dtype=torch.float64, device=device)
            dist.all_reduce(ready, op=dist.ReduceOp.MIN)
            dist.all_reduce(summed, group=layout.copies)
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
            reduce_gradients(trained, layout.copies)
            dist.all_reduce(total, group=layout.copies)
        norm = clip_gradients(trained, clip, layout) if clip is not None else None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if schedule is not None:
            schedule.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record = Step(
            step=start + len(history.steps) + 1,
            loss=float(total.item()),
            tokens=sum(batch.tokens.numel() for batch in group),
            seconds=time.perf_counter() - begin,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            grad_norm=norm,
        )
        history.steps.append(record)
        if on_step is not None:
            on_step(record)
        if saves and save_to is not None and save_every and record.step % save_every == 0:
            _save(model, Path(save_to))
        if checkpoint is not None and checkpoint_every and record.step % checkpoint_every == 0:
            save_checkpoint(checkpoint, model, optimizer, step=record.step, schedule=schedule)
    if saves and save_to is not None and history.steps:
        _save(model, Path(save_to))
    last = history.steps[-1].step if history.steps else 0
    if checkpoint is not None and history.steps and unsaved(checkpoint_every, last):
        save_checkpoint(checkpoint, model, optimizer, step=last, schedule=schedule)
    return history


@dataclass(frozen=True)
class Copies:
    """How processes hold a model they train: `copies`, the group a sum
    across copies runs over (None: every process); `shards`, the group a
    copy split across processes runs over; `split`, the ids of the
    parameters split across it."""

    copies: Any = None
    shards: Any = None
    split: frozenset[int] = frozenset()


def copies(model: Any) -> Copies:
    """The copies of a model under `torch.distributed`. Split across
    processes (`load(tensor_parallel=...)`), a copy is its group, and each
    of its processes sums with the same place in the other groups: every
    process makes those groups, in the same order, the first time."""
    shards = getattr(model, "shard_group", None)
    if shards is None:
        return Copies()
    found = getattr(model, "copies_layout", None)
    if isinstance(found, Copies):
        return found
    import torch.distributed as dist

    from ..torch.module import owner_of

    every: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(every, dist.get_process_group_ranks(shards))
    groups = sorted({tuple(ranks) for ranks in every})
    mine: Any = None
    for place in range(len(groups[0])):
        ranks = [group[place] for group in groups]
        made = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            mine = made
    split = frozenset(
        id(getattr(owner, leaf))
        for owner, leaf in (owner_of(model, path) for path in getattr(model, "shard_parts", {}))
    )
    layout = Copies(mine, shards, split)
    model.copies_layout = layout
    return layout


def reduce_gradients(parameters: list[torch.Tensor], group: Any = None) -> None:
    """Sums every gradient across the processes of `group` (all of
    `torch.distributed` by default): one collective per tensor, all in
    flight at once, so no buffer the size of the model is made. A parameter
    split by `fully_shard` has its part summed already."""
    import torch.distributed as dist

    if group is not None and dist.get_world_size(group) == 1:
        return
    pending = [
        dist.all_reduce(parameter.grad, group=group, async_op=True)
        for parameter in parameters
        if parameter.grad is not None and not sharded(parameter)
    ]
    for work in pending:
        work.wait()


def clip_gradients(
    parameters: list[torch.Tensor], limit: float, layout: Copies | None = None
) -> float:
    """Scales the gradients to a norm of `limit` at most; returns the norm
    before. The parts of parameters split by `fully_shard`, or across a
    copy's processes (`layout`), add up across processes."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if layout is not None and layout.shards is not None:
        return _clip_split(parameters, limit, layout)
    if not any(sharded(g) for g in grads):
        return float(torch.nn.utils.clip_grad_norm_(parameters, limit))
    import torch.distributed as dist

    # A part of a split gradient adds to the others; a scalar kept whole on
    # every process (`Replicate`) counts once.
    split = [cast(Any, g) for g in grads if sharded(g)]
    local = [g.to_local() for g in split if g.placements[0].is_shard()]
    whole = [g for g in grads if not sharded(g)]
    whole += [g.to_local() for g in split if not g.placements[0].is_shard()]
    device = [*local, *whole][0].device
    squares = torch.zeros((), dtype=torch.float32, device=device)
    for g in local:
        squares = squares + g.float().pow(2).sum()
    dist.all_reduce(squares)
    for g in whole:
        squares = squares + g.float().pow(2).sum().to(device)
    norm = squares.sqrt()
    scale = (limit / (norm + 1e-6)).clamp(max=1.0)
    for g in [*local, *whole]:
        g.mul_(scale.to(g.dtype))
    return float(norm)


def _clip_split(parameters: list[torch.Tensor], limit: float, layout: Copies) -> float:
    """`clip_gradients` for a copy split across processes: the split
    parameters' squares summed across them, the rest, which every process
    holds alike, counted once."""
    import torch.distributed as dist

    held = [p for p in parameters if p.grad is not None]
    if not held:
        return 0.0
    device = held[0].grad.device  # type: ignore[union-attr]
    parts = torch.zeros((), dtype=torch.float32, device=device)
    whole = torch.zeros((), dtype=torch.float32, device=device)
    for p in held:
        squares = cast(torch.Tensor, p.grad).float().pow(2).sum()
        if id(p) in layout.split:
            parts = parts + squares
        else:
            whole = whole + squares
    dist.all_reduce(parts, group=layout.shards)
    norm = (parts + whole).sqrt()
    scale = (limit / (norm + 1e-6)).clamp(max=1.0)
    for p in held:
        cast(torch.Tensor, p.grad).mul_(scale.to(cast(torch.Tensor, p.grad).dtype))
    return float(norm)


def _save(model: Any, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if getattr(model, "lora", None) is not None:
        model.save_weights(
            directory / "adapters.safetensors", names="linnet", include=list(lora.PATTERNS)
        )
    else:
        model.save_weights(directory / "model.safetensors")


__all__ = [
    "Batch",
    "Copies",
    "Example",
    "History",
    "Step",
    "clip_gradients",
    "copies",
    "cosine_schedule",
    "load_checkpoint",
    "pack",
    "reduce_gradients",
    "save_checkpoint",
    "train",
]
