"""What a process holds outside PyTorch's caching allocator on this machine.

    python -m linnet.resources.probe
    torchrun --nproc-per-node 2 -m linnet.resources.probe

The memory analysis estimates the CUDA context, the libraries' kernels and
NCCL's buffers from one H100 machine. This measures them here, step by step,
as the device memory in use minus what the allocator holds: pass the
context to `linnet memory --context-bytes` for a closer whole-process
prediction.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("linnet: the probe needs a CUDA device", file=sys.stderr)
        return 1
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    def outside() -> float:
        free, total = torch.cuda.mem_get_info()
        return (total - free - torch.cuda.memory_reserved()) / 2**20

    rows: list[tuple[str, float]] = []
    torch.zeros(1, device="cuda")
    rows.append(("CUDA context", outside()))
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    (a @ a).sum().item()
    q = a[None, None]
    torch.nn.functional.scaled_dot_product_attention(q, q, q).sum().item()
    rows.append(("CUDA libraries", outside() - sum(size for _, size in rows)))
    if distributed:
        import torch.distributed as dist

        dist.init_process_group("nccl")
        t = torch.ones(1024, device="cuda")
        dist.all_reduce(t)
        torch.cuda.synchronize()
        rows.append(("NCCL communicator", outside() - sum(size for _, size in rows)))
        peer = dist.get_rank() ^ 1
        if peer < dist.get_world_size():
            ops = [dist.P2POp(dist.isend, t, peer), dist.P2POp(dist.irecv, t.clone(), peer)]
            for work in dist.batch_isend_irecv(ops):
                work.wait()
            torch.cuda.synchronize()
            rows.append(("NCCL send and receive", outside() - sum(size for _, size in rows)))
        dist.destroy_process_group()
    rank = os.environ.get("RANK", "0")
    for name, size in rows:
        print(f"rank {rank}  {name:<24} {size:8.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
