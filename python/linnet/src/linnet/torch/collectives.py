"""Collectives the generated code of a sharded model calls.

`all_reduce` sums a tensor over the processes of a group. A decoding step's
messages are small -- one token's hidden state is 8 KiB -- and their cost is
latency, not bandwidth: NCCL takes about 10 us for one on two NVLink-joined
H100s. Up to `ONE_SHOT_BYTES` the sum is instead one Triton kernel over
symmetric memory, 7 to 10 us: each process copies its part into a buffer
its peers can read, flags them, waits for their flags, and adds. Larger
messages, CPU tensors, and anything else the kernel does not cover go to
NCCL (or whatever backend the group has).

Run eagerly, a call's host time outweighs the sum's. Outside
`torch.compile` the kernel is launched directly: the custom operator
compiled code needs costs the host about 10 us more. Outside compiled code,
graph capture and autograd, NCCL sums a copy in place: a functional
collective's result is a tensor subclass, and every operation on it goes
through Python, about 100 us a call.

The buffers are set up once per group by `prepare`, a collective every
process of the group calls together (`linnet.torch.load` does, when it
splits a model); without it, every sum goes to NCCL.

In training, a split computation reads its input whole on every process
(`shared`), and each process's gradient of that input is its part of the
whole: backward sums them. A sum's gradient is the gradient of the whole sum,
which every process holds; a gathered slice's is its slice of the gradient.
"""

# PyTorch's symmetric memory, functional collectives, and custom-op registry,
# and the Triton kernel, carry no complete types.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportIncompatibleMethodOverride=false

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import torch

from ..parallel import ONE_SHOT_BYTES

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx
    from torch.distributed import ProcessGroup

# The largest message the one-shot kernel sums (`linnet.parallel`). Past it
# NCCL is faster: the kernel reads every peer's whole part, where NCCL
# splits the work.
_BLOCK = 1024
_MAX_BLOCKS = ONE_SHOT_BYTES // 2 // _BLOCK  # of the narrowest dtype the kernel takes
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

_one_shots: dict[str, OneShot] = {}


class _Launcher(Protocol):
    """A Triton kernel: indexed by its grid, then called with its arguments."""

    def __getitem__(self, grid: tuple[int, ...], /) -> Callable[..., object]: ...


class OneShot:
    """A group's symmetric buffers: two slots of `ONE_SHOT_BYTES` each, the
    flags, and each block's call count. Made by `prepare`."""

    def __init__(self, group: ProcessGroup) -> None:
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        cuda = torch.device("cuda", torch.cuda.current_device())

        self.buffer = symm_mem.empty(ONE_SHOT_BYTES // 2, dtype=torch.float32, device=cuda)
        handle = symm_mem.rendezvous(self.buffer, group.group_name)
        self.rank: int = handle.rank
        self.world: int = handle.world_size
        self.buffers = torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=cuda)
        self.flags = symm_mem.empty(_MAX_BLOCKS * self.world, dtype=torch.int32, device=cuda)
        self.flags.zero_()
        flags = symm_mem.rendezvous(self.flags, group.group_name)
        self.flag_ptrs = torch.tensor(flags.buffer_ptrs, dtype=torch.int64, device=cuda)
        self.counters = torch.zeros(_MAX_BLOCKS, dtype=torch.int32, device=cuda)
        from .kernels import one_shot_all_reduce_kernel

        # `triton.jit` carries no types: to the checker the kernel is a plain function.
        self.kernel = cast(_Launcher, one_shot_all_reduce_kernel)
        # Every process's flags are zero before anyone can set one.
        torch.cuda.synchronize()
        dist.barrier(group)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        grid = (-(-n // _BLOCK),)
        self.kernel[grid](
            x,
            out,
            self.buffers,
            self.flag_ptrs,
            self.counters,
            n,
            ONE_SHOT_BYTES // x.element_size(),
            rank=self.rank,
            world=self.world,
            block_size=_BLOCK,
            num_warps=4,
        )
        return out


@torch.library.custom_op("linnet::one_shot_all_reduce", mutates_args=())
def one_shot_all_reduce(x: torch.Tensor, group: str) -> torch.Tensor:
    """The sum of `x` over the processes of group `group`, by the one-shot
    kernel. An opaque operation, so a compiled step calls it as it is."""
    return _one_shots[group](x)


@one_shot_all_reduce.register_fake
def _(x: torch.Tensor, group: str) -> torch.Tensor:
    return torch.empty_like(x)


def prepare(group: ProcessGroup) -> bool:
    """Sets up the one-shot kernel's buffers for `group`: a collective, which
    every process of the group calls together. Returns whether the kernel
    runs; where it cannot (no Triton, no CUDA, no symmetric memory), every
    sum goes to NCCL."""
    name = str(group.group_name)
    if name in _one_shots:
        return True
    if not torch.cuda.is_available():
        return False
    try:
        _one_shots[name] = OneShot(group)
    except (RuntimeError, ImportError, AttributeError):
        return False
    return True


def _eager(x: torch.Tensor) -> bool:
    """Whether `x` is summed outside compiled code, graph capture and
    autograd, where NCCL's eager calls apply."""
    return not (
        torch.compiler.is_compiling()
        or (x.is_cuda and torch.cuda.is_current_stream_capturing())
        or (x.requires_grad and torch.is_grad_enabled())
    )


def all_reduce(x: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    """The sum of `x` over the processes of `group`."""
    if x.requires_grad and torch.is_grad_enabled():
        return _Sum.apply(x, group)  # type: ignore[no-any-return]
    name = str(group.group_name)
    if (
        name in _one_shots
        and x.is_cuda
        and x.dtype in _DTYPES
        and x.numel() * x.element_size() <= ONE_SHOT_BYTES
    ):
        if torch.compiler.is_compiling():
            return one_shot_all_reduce(x.contiguous(), name)
        return _one_shots[name](x.contiguous())
    if _eager(x):
        import torch.distributed as dist

        out = x.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(out, group=group)
        return out
    from torch.distributed import _functional_collectives as funcol

    return funcol.all_reduce(x, "sum", group)


def all_gather(x: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    """The processes' `x` side by side along the last axis, in rank order."""
    if x.requires_grad and torch.is_grad_enabled():
        return _Gather.apply(x, group)  # type: ignore[no-any-return]
    if _eager(x):
        import torch.distributed as dist

        world = dist.get_world_size(group)
        stacked = x.new_empty((world * x.shape[0], *x.shape[1:]))
        dist.all_gather_single(stacked, x.contiguous(), group=group)
        return torch.cat(stacked.chunk(world), dim=x.dim() - 1)
    from torch.distributed import _functional_collectives as funcol

    return funcol.all_gather_tensor(x.contiguous(), x.dim() - 1, group)


def shared(x: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    """`x` as every process of `group` reads it whole: itself, and in
    backward the sum of the processes' gradients of it."""
    if x.requires_grad and torch.is_grad_enabled():
        return _Shared.apply(x, group)  # type: ignore[no-any-return]
    return x


class _SharedCtx(Protocol):
    group: ProcessGroup


class _GatherCtx(Protocol):
    rank: int
    world: int


class _Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
        return all_reduce(x, group)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad, None


class _Shared(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _SharedCtx, x: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx: _SharedCtx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return all_reduce(grad.contiguous(), ctx.group), None


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _GatherCtx, x: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
        import torch.distributed as dist

        ctx.rank, ctx.world = dist.get_rank(group), dist.get_world_size(group)
        return all_gather(x, group)

    @staticmethod
    def backward(ctx: _GatherCtx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad.chunk(ctx.world, dim=-1)[ctx.rank].contiguous(), None


__all__ = ["ONE_SHOT_BYTES", "all_gather", "all_reduce", "prepare", "shared"]
