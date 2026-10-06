"""Collectives the generated code of a sharded model calls.

`all_reduce` sums a tensor over the processes of a group. A decoding step's
messages are small -- one token's hidden state is 8 KiB -- and their cost is
latency, not bandwidth: NCCL takes about 11 us for one on two NVLink-joined
H100s. Up to `ONE_SHOT_BYTES` the sum is instead one Triton kernel over
symmetric memory, about 4.5 us: each process copies its part into a buffer
its peers can read, flags them, waits for their flags, and adds. Larger
messages, CPU tensors, and anything else the kernel does not cover go to
NCCL (or whatever backend the group has).

The buffers are set up once per group by `prepare`, a collective every
process of the group calls together (`linnet.torch.load` does, when it
splits a model); without it, every sum goes to NCCL.
"""

# PyTorch's symmetric memory, functional collectives, and custom-op registry,
# and the Triton kernel, carry no complete types.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any

import torch

from ..parallel import ONE_SHOT_BYTES

# The largest message the one-shot kernel sums (`linnet.parallel`). Past it
# NCCL is faster: the kernel reads every peer's whole part, where NCCL
# splits the work.
_BLOCK = 1024
_MAX_BLOCKS = 64  # ONE_SHOT_BYTES of the narrowest dtype the kernel takes, by _BLOCK
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

_one_shots: dict[str, OneShot] = {}


class OneShot:
    """A group's symmetric buffers: two slots of `ONE_SHOT_BYTES` each, the
    flags, and each block's call count. Made by `prepare`."""

    def __init__(self, group: Any) -> None:
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

        self.kernel: Any = one_shot_all_reduce_kernel
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


def prepare(group: Any) -> bool:
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


def all_reduce(x: torch.Tensor, group: Any) -> torch.Tensor:
    """The sum of `x` over the processes of `group`."""
    name = str(group.group_name)
    if (
        name in _one_shots
        and x.is_cuda
        and x.dtype in _DTYPES
        and x.numel() * x.element_size() <= ONE_SHOT_BYTES
    ):
        return one_shot_all_reduce(x.contiguous(), name)
    from torch.distributed import _functional_collectives as funcol

    return funcol.all_reduce(x, "sum", group)


def all_gather(x: torch.Tensor, group: Any) -> torch.Tensor:
    """The processes' `x` side by side along the last axis, in rank order."""
    from torch.distributed import _functional_collectives as funcol

    return funcol.all_gather_tensor(x.contiguous(), x.dim() - 1, group)


__all__ = ["ONE_SHOT_BYTES", "all_gather", "all_reduce", "prepare"]
