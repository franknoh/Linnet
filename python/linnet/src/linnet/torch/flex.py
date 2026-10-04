"""A copy Inductor cannot see through, for FlexAttention's query.

Inductor's FlexAttention reads a query that is a view starting inside its
storage (a prompt pass's step rows, sliced off its prompts) as if it started
at the storage's beginning. A copy made inside the compiled graph (a clone,
`contiguous`, arithmetic, a gather) can be folded back into such a view; a
custom operator is opaque to Inductor, so its result is a buffer of its own.
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import torch


@torch.library.custom_op("linnet::fresh_copy", mutates_args=())
def fresh_copy(x: torch.Tensor) -> torch.Tensor:
    """`x` copied into a contiguous buffer of its own."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    out.copy_(x)
    return out


@fresh_copy.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def _backward(_: object, grad: torch.Tensor) -> torch.Tensor:
    return grad


fresh_copy.register_autograd(_backward)
