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

A pipeline's stage runs many micro-batches a step. There `Held` gathers its
weights once and keeps them whole for the step; each micro-batch's
gradients are summed into the parts as its backward finishes them.
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import contextlib
import functools
import inspect
import weakref
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from .. import lora
from ..parallel import units as parallel_units
from ..plan import PlanError
from .module import LinnetModule, owner_of

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup, Work
    from torch.utils.checkpoint import SelectiveCheckpointContext
    from typing_extensions import TypeIs


def fully_shard(
    model: LinnetModule,
    mesh: DeviceMesh | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    only: Callable[[str], bool] | None = None,
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

    chosen = [
        p for p, _ in model.root.named_parameters(remove_duplicate=False) if only is None or only(p)
    ]
    units = parallel_units(chosen)[0]
    replaced: dict[int, nn.Parameter] = {}
    computes: dict[str, torch.dtype] = {}
    with torch.no_grad():
        for path, parameter in list(model.root.named_parameters(remove_duplicate=False)):
            owner, leaf = owner_of(model, path)
            if "." not in path or leaf in owner.absent_params or leaf in (lora.DOWN, lora.UP):
                continue
            if only is not None and not only(path):
                continue
            split = replaced.get(id(parameter))
            if split is None:
                # A weight that does not train needs no f32 copy to update.
                kept = dtype if parameter.requires_grad else parameter.dtype
                split = nn.Parameter(
                    # Where the weight is: a pipeline's model is on `meta`, its
                    # own stage's weights on the device.
                    _split(parameter.detach(), mesh, kept, parameter.device),
                    requires_grad=parameter.requires_grad,
                )
                replaced[id(parameter)] = split
            computes[path] = parameter.dtype
            setattr(owner, leaf, split)
    model.fully_sharded = tuple(units)
    model.gathered_dtypes = computes
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


def _gather_everything(
    ctx: SelectiveCheckpointContext, op: object, *args: object, **kwargs: object
) -> CheckpointPolicy:
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


def _pack(value: torch.Tensor) -> torch.Tensor | _Again:
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


def _unpack(saved: torch.Tensor | _Again) -> torch.Tensor:
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


def sharded(parameter: torch.Tensor) -> TypeIs[DTensor]:
    return isinstance(parameter, DTensor)


class Held:
    """Sharded weights gathered whole once and kept through a pipeline
    stage's step: every micro-batch's forward and backward read the same
    wholes. When a micro-batch's backward finishes a whole's gradient, it is
    summed into each process's part (a reduce-scatter in the part's dtype),
    overlapping the rest of that backward; `finish` waits for the last."""

    def __init__(self, trains: bool) -> None:
        self.trains = trains
        self._wholes: dict[int, torch.Tensor] = {}
        self._summing: _Summing | None = None

    def whole(self, part: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """`part`'s whole in `dtype`, gathered on the first call of the step."""
        if not isinstance(part, DTensor):
            return part
        found = self._wholes.get(id(part))
        if found is not None:
            return found
        with torch.no_grad():
            whole = _joined(part, dtype)
        if self.trains and part.requires_grad:
            whole.requires_grad_(True)
            whole.register_post_accumulate_grad_hook(functools.partial(self._sum, part))
        self._wholes[id(part)] = whole
        return whole

    def finish(self) -> None:
        """Waits for the last sum and drops the wholes."""
        self._settle()
        self._wholes.clear()

    def _sum(self, part: DTensor, whole: torch.Tensor) -> None:
        grad = whole.grad
        whole.grad = None
        assert grad is not None
        # One sum in flight: waiting for the previous one here, rather than
        # at once, lets it run beside this gradient's backward.
        self._settle()
        self._summing = _start_sum(part, grad)

    def _settle(self) -> None:
        summing, self._summing = self._summing, None
        if summing is None:
            return
        local = summing.wait()
        part = summing.part
        with torch.no_grad():
            if part.grad is None:
                part.grad = DTensor.from_local(
                    local,
                    part.device_mesh,
                    part.placements,
                    run_check=False,
                    shape=part.shape,
                    stride=part.stride(),
                )
            else:
                summed = part.grad
                assert isinstance(summed, DTensor)
                summed.to_local().add_(local)


def _rows(part: DTensor) -> tuple[int, int]:
    """The whole's rows and those of each process's padded part (`Shard(0)`
    cuts them as `torch.chunk` does)."""
    rows = part.shape[0]
    return rows, -(-rows // part.device_mesh.size())


def _nccl(group: ProcessGroup) -> bool:
    import torch.distributed as dist

    return dist.get_backend(group) == "nccl"


def _joined(part: DTensor, dtype: torch.dtype) -> torch.Tensor:
    """The whole of `part` in `dtype`, outside autograd."""
    import torch.distributed as dist

    local = part.to_local()
    if not isinstance(part.placements[0], Shard):
        return local.to(dtype, copy=True)
    mesh = part.device_mesh
    group = mesh.get_group()
    rows, each = _rows(part)
    padded = local.new_zeros((each, *part.shape[1:]), dtype=dtype)
    padded[: local.shape[0]].copy_(local)
    joined = local.new_empty((mesh.size() * each, *part.shape[1:]), dtype=dtype)
    if _nccl(group):
        dist.all_gather_single(joined, padded, group=group)
    else:
        dist.all_gather(list(joined.chunk(mesh.size())), padded, group=group)
    return joined[:rows].detach()


@dataclass
class _Summing:
    part: DTensor
    work: Work
    out: torch.Tensor
    full: torch.Tensor  # kept until the sum is done
    rows: int | None  # of `out`, this process's; None: all of it

    def wait(self) -> torch.Tensor:
        self.work.wait()
        return self.out if self.rows is None else self.out[: self.rows]


def _start_sum(part: DTensor, grad: torch.Tensor) -> _Summing:
    """Starts summing a whole's gradient into each process's part."""
    import torch.distributed as dist

    mesh = part.device_mesh
    group = mesh.get_group()
    local = part.to_local()
    if not isinstance(part.placements[0], Shard):
        full = grad.to(part.dtype, copy=True)
        work = dist.all_reduce(full, group=group, async_op=True)
        return _Summing(part, work, full, full, None)
    rows, each = _rows(part)
    size = mesh.size()
    if rows == size * each:
        full = grad.to(part.dtype).contiguous()
    else:
        full = grad.new_zeros((size * each, *grad.shape[1:]), dtype=part.dtype)
        full[:rows].copy_(grad)
    if _nccl(group):
        out = full.new_empty((each, *grad.shape[1:]))
        work = dist.reduce_scatter_single(out, full, group=group, async_op=True)
    else:
        # Gloo sums whole; each process keeps its rows.
        work = dist.all_reduce(full, group=group, async_op=True)
        rank = mesh.get_local_rank()
        out = full[rank * each : (rank + 1) * each]
    return _Summing(part, work, out, full, local.shape[0])


__all__ = ["Held", "fully_shard", "gather", "regathered", "sharded"]
