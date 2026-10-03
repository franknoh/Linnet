"""A Linnet decoder as a causal language model for existing trainers.

Trainers built for transformers models (`transformers.Trainer`, TRL's, and
loops written against them) call `model(input_ids=..., attention_mask=...,
labels=..., position_ids=...)` and read the result's `loss`. `CausalLM`
answers that call from the model's `loss_packed` entry: it removes the
padding, packs the rows' tokens into one row, and keeps each sequence apart,
so no compute goes to padding and the logits are never formed whole.

```python
from linnet import nest
from linnet.torch import CausalLM

model = CausalLM(nest.load("llama-3.1-8b-instruct", backend="torch", device="cuda",
                           compile="inductor", trainable=True))
trainer = transformers.Trainer(model=model, args=..., train_dataset=..., data_collator=...)
```
"""

from __future__ import annotations

from typing import Any

import torch

from .module import LinnetModule

# transformers' label for a position to leave out of the loss.
IGNORE = -100


class CausalLM(torch.nn.Module):
    """`model(input_ids, attention_mask=None, labels=None, position_ids=None)`
    over a Linnet model with a `loss_packed` entry; returns `{"loss": ...}`,
    the mean cross-entropy over the labelled positions.

    `labels` follow transformers: the token each position's next one should
    be, `-100` to leave it out, shifted inside the model (position `p` is
    scored on `labels[p + 1]`). Without `labels` the inputs are learned
    whole. Rows are sequences, their padding given by `attention_mask`; with
    `position_ids`, a position 0 starts a new sequence within a row (the
    padding-free batches of TRL's `DataCollatorWithFlattening`). Packed
    lengths are padded up to a multiple of `bucket`, so few shapes compile."""

    def __init__(self, model: LinnetModule, *, entry: str = "loss_packed", bucket: int = 256):
        super().__init__()
        self.model = model
        self.entry = entry
        self.bucket = bucket

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        inputs, count = packed(input_ids, attention_mask, labels, position_ids, self.bucket)
        loss = self.model.run_entry(self.entry, inputs)
        return {"loss": loss, "num_items": count}


def packed(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    bucket: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """`loss_packed`'s inputs for a transformers-style batch, and the count
    of labelled positions the loss is the mean over."""
    device = input_ids.device
    rows, width = input_ids.shape
    keep = (
        attention_mask.bool()
        if attention_mask is not None
        else torch.ones(rows, width, dtype=torch.bool, device=device)
    )
    if position_ids is None:
        positions = (keep.cumsum(1) - 1).clamp_min(0)
    else:
        positions = position_ids.expand(rows, width)
    # A sequence starts at each row and at each position 0 within one.
    starts = (positions == 0) & keep
    starts[:, 0] = True
    segment = torch.cumsum(starts.flatten(), 0).reshape(rows, width) - 1
    if labels is None:
        labels = input_ids
    # Position p is scored on the label after it, within its own sequence.
    following = torch.full_like(labels, IGNORE)
    following[:, :-1] = labels[:, 1:]
    same = torch.zeros_like(keep)
    same[:, :-1] = (segment[:, 1:] == segment[:, :-1]) & keep[:, 1:]
    following = torch.where(same, following, torch.full_like(following, IGNORE))

    flat = keep.flatten()
    tokens = input_ids.flatten()[flat].to(torch.int32)
    pos = positions.flatten()[flat].to(torch.int32)
    seg = segment.flatten()[flat].to(torch.int32)
    targets = following.flatten()[flat]
    learned = targets != IGNORE
    count = learned.sum()
    weights = learned.float() / count.clamp_min(1).float()
    targets = torch.where(learned, targets, torch.zeros_like(targets)).to(torch.int64)

    size = tokens.numel()
    padding = (-size) % bucket if bucket > 1 else 0
    if padding:
        tokens = torch.cat([tokens, tokens.new_zeros(padding)])
        pos = torch.cat([pos, pos.new_zeros(padding)])
        seg = torch.cat([seg, seg.new_full((padding,), int(seg.max().item()) + 1)])
        targets = torch.cat([targets, targets.new_zeros(padding)])
        weights = torch.cat([weights, weights.new_zeros(padding)])
    return [tokens, pos, seg, targets, weights], count


__all__ = ["IGNORE", "CausalLM", "packed"]
