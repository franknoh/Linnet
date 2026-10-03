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

TRL's `SFTTrainer` computes the loss itself by default (`chunked_nll`), a
block of tokens at a time, from `base_model`'s hidden states and
`get_output_embeddings()`; both come from the model's `hidden_packed`
entry. With `loss_type="nll"` it reads the logits for its token accuracy
and entropy: `CausalLM(model, logits=True)` returns them.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import torch

from ..plan import PlanError
from .loss import linear_cross_entropy
from .module import LinnetModule, owner_of

# transformers' label for a position to leave out of the loss.
IGNORE = -100


class CausalLM(torch.nn.Module):
    """`model(input_ids, attention_mask=None, labels=None, position_ids=None)`
    over a Linnet model with a `loss_packed` entry; returns the loss, the
    mean cross-entropy over the labelled positions, as `output.loss` or
    `output["loss"]`.

    `labels` follow transformers: the token each position's next one should
    be, `-100` to leave it out, shifted inside the model (position `p` is
    scored on `labels[p + 1]`). Without `labels` the inputs are learned
    whole. Rows are sequences, their padding given by `attention_mask`; with
    `position_ids`, a position 0 starts a new sequence within a row (the
    padding-free batches of TRL's `DataCollatorWithFlattening`). Packed
    lengths are padded up to a multiple of `bucket`, so few shapes compile.

    Given `num_items_in_batch` (as `transformers.Trainer` passes it, the
    labelled positions of every micro-batch of a step), the loss is the sum
    over it instead, so accumulated steps take the mean over all of them.

    With `logits=True` the output also holds `logits`, `[rows, width,
    Vocab]` in the input's layout (zero at padding), computed without
    gradients from the `hidden` entry's states and the `head` weight, which
    then also give the loss. They cost the memory transformers' own models
    spend on them. `base_model` and `get_output_embeddings` give the states
    and the head to a trainer that computes the loss itself. `config` and
    `generation_config` hold the few fields trainers read and write (`name`
    is the model's name there)."""

    # `transformers.Trainer` then passes `num_items_in_batch`.
    accepts_loss_kwargs = True

    def __init__(
        self,
        model: LinnetModule,
        *,
        entry: str = "loss_packed",
        bucket: int = 256,
        logits: bool = False,
        hidden: str = "hidden_packed",
        head: str = "lm_head.weight",
        name: str = "",
    ):
        super().__init__()
        self.model = model
        self.entry = entry
        self.bucket = bucket
        self.return_logits = logits
        self.hidden = hidden
        self.head = head
        try:
            vocab: int | None = model.get_parameter("root." + head).shape[0]
        except AttributeError:
            vocab = None
        self.config = Config(name, vocab)
        self.generation_config = SimpleNamespace(
            eos_token_id=None, pad_token_id=None, bos_token_id=None
        )
        self.model_tags: list[str] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        num_items_in_batch: torch.Tensor | int | None = None,
        **_: Any,
    ) -> Output:
        inputs, _count = packed(
            input_ids, attention_mask, labels, position_ids, self.bucket, num_items_in_batch
        )
        if not self.return_logits:
            return Output(loss=self.model.run_entry(self.entry, inputs))
        states = self._states(inputs)
        weight = self.get_output_embeddings().weight
        loss = linear_cross_entropy(states, weight, inputs[3], inputs[4])
        keep = _kept(input_ids, attention_mask)
        with torch.no_grad():
            flat = states[: int(keep.sum())] @ weight.T
            logits = flat.new_zeros((*input_ids.shape, flat.shape[-1]))
            logits[keep] = flat
        return Output(loss=loss, logits=logits)

    @property
    def base_model(self) -> Callable[..., SimpleNamespace]:
        """The decoder without its output head, as TRL's `chunked_nll` loss
        calls it: `last_hidden_state`, `[rows, width, H]` in the input's
        layout (zero at padding), from the `hidden` entry. Not a submodule,
        so the weights are not saved twice."""

        def backbone(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            **_: Any,
        ) -> SimpleNamespace:
            inputs, _count = packed(input_ids, attention_mask, None, position_ids, self.bucket)
            states = self._states(inputs)
            keep = _kept(input_ids, attention_mask)
            shape = (*input_ids.shape, states.shape[-1])
            hidden = states.new_zeros(shape).index_put((keep,), states[: int(keep.sum())])
            return SimpleNamespace(last_hidden_state=hidden, hidden_states=None)

        return backbone

    def get_output_embeddings(self) -> SimpleNamespace:
        """The output head's `weight` (and `bias`, None when absent)."""
        if self.head in getattr(self.model, "lora_paths", []):
            raise PlanError(f"`{self.head}` is read alone here; leave the output head unadapted")
        owner, leaf = owner_of(self.model, self.head)
        bias = None
        if leaf == "weight" and "bias" in owner._parameters and "bias" not in owner.absent_params:
            bias = owner.get_parameter("bias")
        return SimpleNamespace(weight=owner.get_parameter(leaf), bias=bias)

    def _states(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if self.hidden not in self.model.entries:
            raise PlanError(f"the model has no `{self.hidden}` entry, which this needs")
        return self.model.run_entry(self.hidden, inputs[:3])

    def add_model_tags(self, tags: str | list[str]) -> None:
        """Trainers tag the model they train (TRL: `trl`, `sft`); the tags
        are kept in `model_tags`."""
        new = [tags] if isinstance(tags, str) else list(tags)
        self.model_tags = sorted({*self.model_tags, *new})

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: Any = None, **_: Any
    ) -> None:
        """Trainers call this for `gradient_checkpointing=True` (TRL's
        default). Linnet entries keep their activations; with
        `compile="inductor"`, `activation_memory_budget` recomputes them."""
        warnings.warn(
            "Linnet entries have no per-layer checkpointing; with compile='inductor', set "
            "torch._functorch.config.activation_memory_budget to recompute activations",
            stacklevel=2,
        )

    def gradient_checkpointing_disable(self) -> None:
        pass


class Output(dict[str, Any]):
    """The loss (and logits), by key or attribute, as transformers' model
    outputs are read."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            hint = "; construct CausalLM(..., logits=True) for them" if name == "logits" else ""
            raise AttributeError(f"the output has no `{name}`{hint}") from None


class Config:
    """The fields transformers-style trainers read off `model.config`, and
    write: the end-of-sequence and padding tokens."""

    model_type = "linnet"
    _attn_implementation = "linnet"

    def __init__(self, name: str, vocab_size: int | None) -> None:
        self._name_or_path = name
        self.vocab_size = vocab_size
        self.bos_token_id: int | None = None
        self.eos_token_id: int | list[int] | None = None
        self.pad_token_id: int | None = None
        self.use_cache = False

    def get_text_config(self, *_: Any, **__: Any) -> Config:
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": self.model_type, **vars(self)}

    def to_json_string(self, *_: Any, **__: Any) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _kept(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is not None:
        return attention_mask.bool()
    return torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)


def packed(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    bucket: int,
    over: torch.Tensor | int | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """`loss_packed`'s inputs for a transformers-style batch, and its count
    of labelled positions. The loss is the mean over them, or the sum over
    `over` when given."""
    device = input_ids.device
    rows, width = input_ids.shape
    keep = _kept(input_ids, attention_mask)
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
    divisor = count if over is None else torch.as_tensor(over, device=device)
    weights = learned.float() / divisor.clamp_min(1).float()
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


__all__ = ["IGNORE", "CausalLM", "Config", "Output", "packed"]
