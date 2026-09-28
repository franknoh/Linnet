"""Tensor parallelism for `linnet.torch`: weights and KV caches as DTensors.

`load(..., tensor_parallel=mesh)` turns each weight into a DTensor split over
a one-dimensional `DeviceMesh` along the axis `linnet.parallel` gives, and each
KV cache into one split by heads. Every process of the mesh loads the model
the same way and runs the same entries; DTensor adds the collectives the
splits need. Values the generated code makes itself (masks, rotary tables,
the inputs) are taken as replicated, and results come back whole.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..parallel import split_axis, state_axis
from .module import LinnetModule, owner_of


def distribute(
    module: LinnetModule, mesh: Any, rules: Mapping[str, int | None] | None = None
) -> None:
    """Splits `module`'s weights and states over `mesh`, in place."""
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    devices = mesh.size()
    for name, parameter in list(module.named_parameters()):
        path = name.removeprefix("root.")
        axis = split_axis(path, tuple(parameter.shape), devices, rules)
        placement = [Replicate()] if axis is None else [Shard(axis)]
        owner, leaf = owner_of(module, path)
        split = distribute_tensor(parameter.detach(), mesh, placement)
        setattr(owner, leaf, torch.nn.Parameter(split, requires_grad=parameter.requires_grad))
    for name, buffer in list(module.named_buffers()):
        path = name.removeprefix("root.")
        axis = state_axis(tuple(buffer.shape), devices)
        placement = [Replicate()] if axis is None else [Shard(axis)]
        owner, leaf = owner_of(module, path)
        setattr(owner, leaf, distribute_tensor(buffer, mesh, placement))
    module.tensor_parallel = mesh


def whole(value: Any) -> Any:
    """A DTensor result as a whole tensor on every process; anything else as it is."""
    from torch.distributed.tensor import DTensor

    if isinstance(value, DTensor):
        return value.full_tensor()
    if isinstance(value, tuple):
        parts: tuple[Any, ...] = value  # pyright: ignore[reportUnknownVariableType]
        return tuple(whole(part) for part in parts)
    return value
