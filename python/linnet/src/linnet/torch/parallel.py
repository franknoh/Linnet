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
from typing import TYPE_CHECKING, TypedDict, TypeVar, Unpack, overload

import torch

from ..parallel import split_axis, state_axis
from .module import LinnetModule, owner_of

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

_T = TypeVar("_T")


def distribute(
    module: LinnetModule, mesh: DeviceMesh, rules: Mapping[str, int | None] | None = None
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


class _AttentionOptions(TypedDict, total=False):
    """`scaled_dot_product_attention`'s keyword arguments past the mask."""

    dropout_p: float
    is_causal: bool
    scale: float | None
    enable_gqa: bool


class SplitFunctional:
    """`torch.nn.functional` for a generated module whose weights are split.

    DTensor splits an attention mask the way it splits the heads, along
    axis 1, which for a mask of `[queries, keys]` is the keys. The mask
    goes in expanded to the query's `[batch, heads, queries, keys]` (a view,
    no copy), so axis 1 is the heads. Everything else is `F` itself.
    """

    def __getattr__(self, name: str) -> object:
        return getattr(torch.nn.functional, name)

    @staticmethod
    def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **options: Unpack[_AttentionOptions],
    ) -> torch.Tensor:
        if attn_mask is not None and attn_mask.dim() < query.dim():
            attn_mask = attn_mask.expand(*query.shape[:-1], key.shape[-2])
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, **options
        )


@overload
def whole(value: torch.Tensor) -> torch.Tensor: ...
@overload
def whole(value: _T) -> _T: ...
def whole(value: object) -> object:
    """A DTensor result as a whole tensor on every process; anything else as it is."""
    from torch.distributed.tensor import DTensor

    if isinstance(value, DTensor):
        return value.full_tensor()
    if isinstance(value, tuple):
        parts: tuple[object, ...] = value  # pyright: ignore[reportUnknownVariableType]
        return tuple(whole(part) for part in parts)
    return value
