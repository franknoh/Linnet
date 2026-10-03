"""Fully sharded data parallelism (FSDP) for generated Linnet modules.

`fully_shard(model)` splits every parameter of a model across the processes
of a mesh, along its first dimension: each process keeps one part, in f32 by
default, and with it that part of the gradients and of the optimizer state.
The model's entries are compiled again to gather a block's parameters whole,
in the dtype the model computes in, just before the block runs, and to drop
them when it returns. The gradients are summed back into each process's part
(a reduce-scatter). Each process trains on its own batches, as with
`linnet.train` under `torch.distributed`.

PyTorch's own `fully_shard` cannot do this for a Linnet model. It gathers a
module's parameters in hooks around that module's `forward`, and generated
code computes every block inline in one function, calling none of them.

Backward needs the gathered weights again. Under `torch.compile`, each
gather is marked for recomputation, so backward gathers again rather than
keeping the whole model. Without it, saved-tensor hooks keep a part in place
of each whole weight and gather it again when backward reads it.
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import contextlib
import functools
import inspect
import weakref
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from ..plan import PlanError
from .module import LinnetModule, owner_of


def fully_shard(
    model: LinnetModule, mesh: DeviceMesh | None = None, *, dtype: torch.dtype = torch.float32
) -> list[str]:
    """Splits `model`'s parameters across the processes of `mesh` (one
    dimension; every process by default) and compiles its entries to gather
    them a block at a time. Each process keeps its part in `dtype`, the copy
    the optimizer updates, and gathers it in the dtype the model computes in;
    gradients are summed across processes in `dtype`. A unit is each element
    of the root block's lists (`layers.0`, `layers.1`, ...) and each other
    block the root holds. Returns the units.

    Load the model on this process's device with `compile=True` or
    `"inductor"`, without placement. Add adapters (`add_lora`) before: they
    stay whole on every process, and the weights they adapt, which do not
    train, are split in their own dtype. Parameters of the root block itself
    stay whole as well. `linnet.train` sums the gradients of whatever stays
    whole."""
    from .compiled import CompiledLinnetModule

    if not isinstance(model, CompiledLinnetModule):
        raise PlanError("sharding needs generated code: load with compile=True or 'inductor'")
    if model.tensor_parallel is not None or (
        model.placement is not None and not model.placement.trivial
    ):
        raise PlanError("a sharded model runs on one device per process")
    if model.fully_sharded:
        raise PlanError("the model is sharded already")
    if "forward_dtype" not in inspect.signature(DTensor.redistribute).parameters:
        raise PlanError("this PyTorch is too old to shard: DTensor.redistribute has no dtypes")
    if mesh is None:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise PlanError("sharding needs torch.distributed initialized (torchrun)")
        mesh = init_device_mesh(model.interpreter.device.type, (dist.get_world_size(),))
    if mesh.ndim != 1:
        raise PlanError("sharding takes a one-dimensional mesh")

    units: list[str] = []
    for name, child in model.root.named_children():
        if isinstance(child, nn.ModuleList):
            units += [f"{name}.{i}" for i in range(len(child))]
        else:
            units.append(name)
    replaced: dict[int, nn.Parameter] = {}
    with torch.no_grad():
        for path, parameter in list(model.root.named_parameters(remove_duplicate=False)):
            owner, leaf = owner_of(model, path)
            if "." not in path or leaf in owner.absent_params or leaf in ("lora_a", "lora_b"):
                continue
            split = replaced.get(id(parameter))
            if split is None:
                # A weight that does not train needs no f32 copy to update.
                kept = dtype if parameter.requires_grad else parameter.dtype
                split = nn.Parameter(
                    _split(parameter.detach(), mesh, kept, model.interpreter.device),
                    requires_grad=parameter.requires_grad,
                )
                replaced[id(parameter)] = split
            setattr(owner, leaf, split)
    model.fully_sharded = tuple(units)
    model._recompile()  # pyright: ignore[reportPrivateUsage]
    return units


def _split(
    whole: torch.Tensor, mesh: DeviceMesh, dtype: torch.dtype, device: torch.device
) -> DTensor:
    """This process's part of `whole` along dim 0, as `Shard(0)` cuts it
    (`torch.chunk`); a scalar is kept whole."""
    whole = whole.to(dtype).contiguous()
    if whole.dim() == 0:
        return DTensor.from_local(whole.to(device), mesh, [Replicate()], run_check=False)
    rank = mesh.get_local_rank()
    pieces = torch.chunk(whole, mesh.size(), dim=0)
    local = pieces[rank] if rank < len(pieces) else whole.new_empty((0, *whole.shape[1:]))
    # A copy: the chunk is a view, which would keep the whole alive.
    return DTensor.from_local(
        local.to(device, copy=True),
        mesh,
        [Shard(0)],
        run_check=False,
        shape=whole.shape,
        stride=whole.stride(),
    )


def gather(part: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """A sharded parameter whole, in `dtype`; its gradient is summed back
    into each process's part. Generated code calls this."""
    if not isinstance(part, DTensor):
        return part if part.dtype == dtype else part.to(dtype)
    if torch.compiler.is_compiling():
        return checkpoint(_whole, part, dtype, use_reentrant=False, context_fn=_recompute)
    whole = _whole(part, dtype)
    if _held is not None:
        _held[whole.untyped_storage().data_ptr()] = (weakref.ref(whole), part, dtype)
    return whole


def _whole(part: DTensor, dtype: torch.dtype) -> torch.Tensor:
    replicated = [Replicate()] * part.device_mesh.ndim
    return part.redistribute(
        part.device_mesh, replicated, forward_dtype=dtype, backward_dtype=part.dtype
    ).to_local(grad_placements=[Partial()] * part.device_mesh.ndim)


def _gather_everything(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
    return CheckpointPolicy.MUST_RECOMPUTE


_recompute = functools.partial(create_selective_checkpoint_contexts, _gather_everything)


# Without `torch.compile`: the whole weights gathered during the current
# call, by storage, while `regathered` is active.
_held: dict[int, tuple[weakref.ref[torch.Tensor], DTensor, torch.dtype]] | None = None


@dataclass(frozen=True)
class _Again:
    """A saved view of a gathered weight, kept as the part to gather again."""

    part: DTensor
    dtype: torch.dtype
    size: torch.Size
    stride: tuple[int, ...]
    offset: int


def _pack(value: torch.Tensor) -> Any:
    if _held is None or isinstance(value, DTensor):
        return value
    try:
        key = value.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError):
        return value
    found = _held.get(key)
    if found is None or found[0]() is None:
        return value
    _, part, dtype = found
    return _Again(part, dtype, value.size(), value.stride(), int(value.storage_offset()))


def _unpack(saved: Any) -> torch.Tensor:
    if not isinstance(saved, _Again):
        return saved
    with torch.no_grad():
        whole = _whole(saved.part, saved.dtype)
    return whole.as_strided(saved.size, saved.stride, saved.offset)


@contextlib.contextmanager
def regathered() -> Generator[None]:
    """Around a call to generated code that runs without `torch.compile`:
    backward keeps the parts of the weights it gathers, not the wholes."""
    global _held
    outer, _held = _held, {}
    try:
        with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
            yield
    finally:
        _held = outer


def sharded(parameter: torch.Tensor) -> bool:
    return isinstance(parameter, DTensor)


__all__ = ["fully_shard", "gather", "regathered", "sharded"]
