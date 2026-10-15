"""Costs every process of a split model agrees on: each process measures
every region (no times kept from before, which could differ between them),
then the slowest process's time stands for each. The processes then plan
alike, so none runs a region with a collective the others leave out."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist

from ..backends import Backend, Cost, CostModel, Options, Region, Target


class Agreed:
    """`inner`'s costs, made the same on every process of `group`: the
    largest of their times, and unknown where any process does not know."""

    def __init__(
        self,
        inner: CostModel[torch.Tensor],
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        self.inner = inner
        self.group = group
        self.device = device

    def costs(
        self,
        asked: Sequence[tuple[Backend[torch.Tensor], Region]],
        target: Target,
        options: Options,
    ) -> list[Cost | None]:
        local = self.inner.costs(asked, target, options)
        seconds = agree(
            [c.seconds if c is not None else 0.0 for c in local], self.group, self.device
        )
        unknown = agree([1.0 if c is None else 0.0 for c in local], self.group, self.device)
        return [
            None if gap > 0 or c is None else Cost(s, c.bytes)
            for c, s, gap in zip(local, seconds, unknown, strict=True)
        ]


def agree(values: list[float], group: dist.ProcessGroup, device: torch.device) -> list[float]:
    """Each of `values` made the largest any process of `group` holds."""
    if not values:
        return []
    held = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(held, op=dist.ReduceOp.MAX, group=group)
    return [float(held[i].item()) for i in range(len(values))]


__all__ = ["Agreed", "agree"]
