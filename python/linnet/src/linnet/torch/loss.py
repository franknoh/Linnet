"""An output head and its loss, a block of tokens at a time.

`std.nn.loss::linear_cross_entropy` and `linear_token_log_probs` take the
hidden states and the output head's weight. Their bodies multiply first,
which holds the [N, V] logits: 2 GB in f32 for 4096 tokens over Llama 3's
128K vocabulary, and as much again for their gradient. Here each block of
rows (about `BLOCK_BYTES` of f32 logits) is multiplied, reduced and dropped:

- `linear_cross_entropy` returns one number, so its gradient is known in
  the forward pass: each block's gradient goes into the hidden states' and
  the weight's as it is computed, and backward only scales them.
- `linear_token_log_probs` returns one value per token, scaled in backward
  by a gradient known only then: forward keeps each row's log-sum-exp, and
  backward multiplies each block again.

Both accumulate the weight's gradient in f32.

`split_cross_entropy` and `split_token_log_probs` are the same over a
vocabulary split across the processes of a group (tensor parallelism): each
holds its rows of the output head, and each block's row maxima, sums and
target logits are combined across the processes, never the logits. The
hidden states' gradient each process computes is its part of the whole.
"""

# pyright: reportIncompatibleMethodOverride=false, reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed import ProcessGroup

    _CrossEntropyFn = Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ProcessGroup | None], torch.Tensor
    ]
    _LogProbsFn = Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, ProcessGroup | None], torch.Tensor
    ]


# The f32 logits of one block of rows.
BLOCK_BYTES = 1 << 30


def _rows(vocab: int) -> int:
    return max(1, BLOCK_BYTES // (4 * vocab))


def _block_logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (hidden @ weight.T).float()


def _part(
    targets: torch.Tensor, vocab: int, group: ProcessGroup | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whether each target lies in this process's `vocab` columns, and where
    (clamped into them when it does not). Without a group, all of them do."""
    low = 0
    if group is not None:
        import torch.distributed as dist

        low = dist.get_rank(group) * vocab
    local = targets - low
    inside = (local >= 0) & (local < vocab)
    return inside, local.clamp(0, vocab - 1)


def _whole(
    logits: torch.Tensor, targets: torch.Tensor, group: ProcessGroup | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Over the whole vocabulary, of which `logits` [rows, V] are this
    process's part: each row's log-sum-exp and target logit, and where the
    target lies in this part (`inside`, `local`)."""
    inside, local = _part(targets, logits.shape[1], group)
    picked = logits.gather(1, local[:, None]).squeeze(1)
    if group is None:
        return torch.logsumexp(logits, dim=-1), picked, inside, local
    import torch.distributed as dist

    peak = logits.amax(-1)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX, group=group)
    picked = torch.where(inside, picked, 0.0)
    sums = torch.stack([(logits - peak[:, None]).exp().sum(-1), picked])
    dist.all_reduce(sums, group=group)
    return peak + sums[0].log(), sums[1], inside, local


class _CrossEntropyCtx(Protocol):
    """What `_CrossEntropy` reads from and keeps on its autograd context."""

    weight_dtype: torch.dtype

    @property
    def needs_input_grad(self) -> tuple[bool, ...]: ...
    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...
    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _TokenLogProbsCtx(Protocol):
    """What `_TokenLogProbs` reads from and keeps on its autograd context."""

    group: ProcessGroup | None

    @property
    def needs_input_grad(self) -> tuple[bool, ...]: ...
    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...
    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _CrossEntropyCtx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        group: ProcessGroup | None,
    ) -> torch.Tensor:
        rows = _rows(weight.shape[0])
        grad_hidden = torch.empty_like(hidden) if ctx.needs_input_grad[0] else None
        grad_weight = (
            torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
            if ctx.needs_input_grad[1]
            else None
        )
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for start in range(0, hidden.shape[0], rows):
            block = hidden[start : start + rows]
            logits = _block_logits(block, weight)
            weight_of = weights[start : start + rows].float()
            norm, picked, inside, local = _whole(logits, targets[start : start + rows], group)
            total += (weight_of * (norm - picked)).sum()
            if grad_hidden is None and grad_weight is None:
                continue
            # d/dlogits of w * (logsumexp - logit[t]) = w * (softmax - onehot(t)),
            # this process's columns of it.
            probs = logits.sub_(norm[:, None]).exp_()
            probs.scatter_add_(1, local[:, None], -inside[:, None].to(probs.dtype))
            grad = probs.mul_(weight_of[:, None]).to(hidden.dtype)
            if grad_hidden is not None:
                grad_hidden[start : start + rows] = grad @ weight
            if grad_weight is not None:
                grad_weight += (grad.T @ block).float()
        ctx.save_for_backward(
            grad_hidden if grad_hidden is not None else torch.empty(0),
            grad_weight if grad_weight is not None else torch.empty(0),
        )
        ctx.weight_dtype = weight.dtype
        return total

    @staticmethod
    def backward(
        ctx: _CrossEntropyCtx, grad_total: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        grad_hidden, grad_weight = ctx.saved_tensors
        scale = grad_total.float()
        return (
            (grad_hidden * scale.to(grad_hidden.dtype)) if ctx.needs_input_grad[0] else None,
            (grad_weight * scale).to(ctx.weight_dtype) if ctx.needs_input_grad[1] else None,
            None,
            None,
            None,
        )


class _TokenLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _TokenLogProbsCtx | None,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        group: ProcessGroup | None,
    ) -> torch.Tensor:
        rows = _rows(weight.shape[0])
        out = torch.empty(hidden.shape[0], dtype=torch.float32, device=hidden.device)
        norms = torch.empty_like(out)
        for start in range(0, hidden.shape[0], rows):
            logits = _block_logits(hidden[start : start + rows], weight)
            norm, picked, _, _ = _whole(logits, targets[start : start + rows], group)
            norms[start : start + rows] = norm
            out[start : start + rows] = picked - norm
        if ctx is not None:
            ctx.save_for_backward(hidden, weight, targets, norms)
            ctx.group = group
        return out

    @staticmethod
    def backward(ctx: _TokenLogProbsCtx, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        hidden, weight, targets, norms = ctx.saved_tensors
        rows = _rows(weight.shape[0])
        grad_hidden = torch.empty_like(hidden) if ctx.needs_input_grad[0] else None
        grad_weight = (
            torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
            if ctx.needs_input_grad[1]
            else None
        )
        for start in range(0, hidden.shape[0], rows):
            block = hidden[start : start + rows]
            inside, local = _part(targets[start : start + rows], weight.shape[0], ctx.group)
            # d/dlogits of logit[t] - logsumexp = onehot(t) - softmax, this
            # process's columns of it.
            probs = _block_logits(block, weight).sub_(norms[start : start + rows, None]).exp_()
            probs.neg_().scatter_add_(1, local[:, None], inside[:, None].to(probs.dtype))
            grad = probs.mul_(grad_out[start : start + rows, None].float()).to(hidden.dtype)
            if grad_hidden is not None:
                grad_hidden[start : start + rows] = grad @ weight
            if grad_weight is not None:
                grad_weight += (grad.T @ block).float()
        return (
            grad_hidden,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            None,
            None,
        )


def linear_cross_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """`sum_n weights[n] * -log softmax(hidden[n] @ weight.T)[targets[n]]`,
    in f32, a block of rows at a time."""
    return split_cross_entropy(hidden, weight, targets, weights, None)


def linear_token_log_probs(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """`log softmax(hidden[n] @ weight.T)[targets[n]]` per row, in f32, a
    block of rows at a time."""
    return split_token_log_probs(hidden, weight, targets, None)


def split_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    group: ProcessGroup | None,
) -> torch.Tensor:
    """`linear_cross_entropy` over a vocabulary split across `group`'s
    processes, `weight` this one's rows of the output head, in order; one
    process without a group."""
    return _cross_entropy(hidden, weight, targets.long(), weights, group)


def split_token_log_probs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    group: ProcessGroup | None,
) -> torch.Tensor:
    """`linear_token_log_probs` over a vocabulary split across `group`'s
    processes, as `split_cross_entropy`."""
    return _log_probs(hidden, weight, targets.long(), group)


def _cross_entropy_eager(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    group: ProcessGroup | None,
) -> torch.Tensor:
    if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
        return _CrossEntropy.apply(hidden, weight, targets, weights, group)  # type: ignore[no-any-return]
    with torch.no_grad():
        picked = _TokenLogProbs.forward(None, hidden, weight, targets, group)
    return -(weights.float() * picked).sum()


def _log_probs_eager(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    group: ProcessGroup | None,
) -> torch.Tensor:
    return _TokenLogProbs.apply(hidden, weight, targets, group)  # type: ignore[no-any-return]


# `torch.compile` calls these as they are: their loops over blocks are the
# point, and tracing would unroll them.
# `disable` carries no types; what it returns takes the same arguments.
_cross_entropy = cast(
    "_CrossEntropyFn",
    torch.compiler.disable(_cross_entropy_eager),
)
_log_probs = cast(
    "_LogProbsFn",
    torch.compiler.disable(_log_probs_eager),
)


__all__ = [
    "BLOCK_BYTES",
    "linear_cross_entropy",
    "linear_token_log_probs",
    "split_cross_entropy",
    "split_token_log_probs",
]
