"""The one-shot all-reduce (`linnet.torch.collectives`) sums what NCCL sums,
call after call and replayed in a CUDA graph, and leaves larger messages to
NCCL; gathers lay the parts as NCCL does. Needs two CUDA devices."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="the one-shot all-reduce needs two CUDA devices"
)


def _sum(rank: int, port: int, out: str) -> None:
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    from torch.distributed import _functional_collectives as funcol

    from linnet.torch.collectives import ONE_SHOT_BYTES, all_gather, all_reduce, prepare

    group = dist.group.WORLD
    assert group is not None and prepare(group)
    problems: list[str] = []

    def nccl(x: torch.Tensor) -> torch.Tensor:
        return funcol.wait_tensor(funcol.all_reduce(x, "sum", group))

    generator = torch.Generator(device="cuda").manual_seed(rank)
    for dtype, count in (
        (torch.bfloat16, 4096),  # one token's hidden state
        (torch.float16, 8193),  # a length no block divides
        (torch.float32, ONE_SHOT_BYTES // 4),  # the largest f32 message it takes
        (torch.bfloat16, ONE_SHOT_BYTES),  # past the limit: NCCL
    ):
        x = torch.randn(count, device="cuda", generator=generator).to(dtype)
        got = all_reduce(x, group)
        expected = nccl(x.clone())
        if not torch.allclose(got.float(), expected.float(), atol=1e-2, rtol=1e-2):
            problems.append(f"{dtype} x {count}")
    # Many calls in a row: the epochs and the two slots.
    x = torch.randn(4096, device="cuda", generator=generator).to(torch.bfloat16)
    for i in range(500):
        y = all_reduce(x + i, group)
        if i % 50 == 0 and not torch.allclose(y.float(), nccl(x + i).float(), atol=0.5):
            problems.append(f"call {i}")
    # Gathered side by side, as NCCL lays the parts.
    part = torch.randn(4, 8, device="cuda", generator=generator)
    whole = funcol.wait_tensor(funcol.all_gather_tensor(part, 1, group))
    if not torch.equal(all_gather(part, group), whole):
        problems.append("gather")
    # Captured in a CUDA graph and replayed.
    static = x.clone()
    for _ in range(2):
        all_reduce(static, group)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = all_reduce(static, group)
        for _ in range(9):
            captured = all_reduce(captured * 0.5, group)
    for i in range(3):
        static.copy_(x + i)
        graph.replay()
        expected = nccl(static.clone())
        for _ in range(9):
            expected = nccl(expected * 0.5)
        if not torch.allclose(captured.float(), expected.float(), atol=0.5, rtol=1e-2):
            problems.append(f"replay {i}")
    torch.cuda.synchronize()
    if rank == 0:
        Path(out).write_text("\n".join(problems), encoding="utf-8")
    dist.destroy_process_group()


def test_one_shot_sums_as_nccl_does(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = tmp_path / "problems.txt"
    spawn(_sum, args=(port, str(out)), nprocs=2)
    assert out.read_text(encoding="utf-8") == ""
