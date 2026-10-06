"""A device's rates, measured: what `linnet fit --maximize throughput`
predicts with.

    python -m linnet.resources.calibrate --name my-h100 --output my-h100.json
    torchrun --nproc-per-node 2 -m linnet.resources.calibrate --output my-h100.json
    linnet fit model --maximize throughput --device my-h100.json

Each rate is the best of several timed runs after a warm-up:

- matrix products in bf16 and f32 (TF32 off, PyTorch's default), at the
  shapes of a transformer's projections and by the weight's size;
- the bandwidth of a large copy, of a sum of two tensors, and of a product
  with a broadcast operand;
- fused attention, forward and backward, at head widths 64 and 128, in
  bf16 and f32: causal
  (`is_causal`, the flash kernel) and under a boolean mask (the
  memory-efficient kernel with a bias), with grouped key/value heads;
- one query per row under a mask, as the generated code runs it, as the
  cache's bytes read per second;
- the host time of one eager operation, over a decoder layer's mix of calls
  on tiny tensors, and the device time of a small kernel issued eagerly;
- under `torchrun`, NCCL's all-reduce: a small message's time and a large
  one's bus bandwidth; and a small one as the generated code calls it,
  host included.

The rates are the device's, not a model's: predictions built on them are
compared with measured models in the docs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any


def _time(run: Callable[[], Any], repeats: int = 20) -> float:
    """The best of a few timings of `repeats` calls, in device seconds per call."""
    import torch

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            run()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1000 / repeats)
    return best


def _host(device: Any) -> float:
    """Host seconds per eager call: a decoder layer's calls on tensors too
    small for the device to be the limit."""
    import torch
    from torch.nn import functional

    width, heads = 64, 4
    x = torch.randn(1, 1, width, device=device, dtype=torch.bfloat16)
    w = torch.randn(width, width, device=device, dtype=torch.bfloat16)
    up = torch.randn(4 * width, width, device=device, dtype=torch.bfloat16)
    down = torch.randn(width, 4 * width, device=device, dtype=torch.bfloat16)
    norm = torch.ones(width, device=device, dtype=torch.bfloat16)
    cos = torch.randn(1, width // heads, device=device, dtype=torch.bfloat16)
    cache = torch.randn(1, heads, 64, width // heads, device=device, dtype=torch.bfloat16)
    mask = torch.ones(1, 64, dtype=torch.bool, device=device)
    position = torch.zeros(1, dtype=torch.long, device=device)

    def rotate(t: Any) -> Any:
        half = t.shape[-1] // 2
        return t * cos + torch.cat([-t[..., half:], t[..., :half]], dim=-1) * cos

    def layer() -> int:
        h = torch.rms_norm(x, [width], eps=1e-6) * norm
        q = functional.linear(h, w).reshape(1, 1, heads, -1).permute(0, 2, 1, 3)
        k = functional.linear(h, w).reshape(1, 1, heads, -1).permute(0, 2, 1, 3)
        v = functional.linear(h, w).reshape(1, 1, heads, -1).permute(0, 2, 1, 3)
        q, k = rotate(q), rotate(k)
        cache.index_copy_(2, position, k)
        cache.index_copy_(2, position, v)
        a = functional.scaled_dot_product_attention(q, cache, cache, attn_mask=mask)
        h = x + functional.linear(a.permute(0, 2, 1, 3).reshape(1, 1, width), w)
        g = torch.rms_norm(h, [width], eps=1e-6) * norm
        g = functional.linear(
            functional.silu(functional.linear(g, up)) * functional.linear(g, up), down
        )
        _ = h + g
        return 40  # the calls above

    calls = layer()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        begin = time.perf_counter()
        for _ in range(200):
            layer()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - begin) / 200 / calls)
    return best


def _products(device: Any) -> tuple[dict[str, float], dict[str, list[list[float]]]]:
    """The best bf16 and f32 rates, and both by size: `[n, FLOP/s]` for 8192
    rows times an `n` by `n` weight."""
    import torch
    from torch.nn import functional

    def rate(m: int, n: int, k: int, dtype: Any) -> float:
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(n, k, device=device, dtype=dtype)
        return 2 * m * n * k / _time(lambda: functional.linear(a, b))

    shapes = [(8192, 8192, 8192), (8192, 14336, 4096), (8192, 4096, 14336), (4096, 4096, 4096)]
    best = max(rate(m, n, k, torch.bfloat16) for m, n, k in shapes)
    sizes = {
        "bf16": [
            [float(n), rate(8192, n, n, torch.bfloat16)] for n in (128, 256, 512, 1024, 2048, 4096)
        ],
        "f32": [[float(n), rate(8192, n, n, torch.float32)] for n in (256, 512, 1024, 2048, 4096)],
    }
    peaks = {"bf16": best, "f16": best, "f32": max(r for _, r in sizes["f32"])}
    return peaks, sizes


def _bandwidth(device: Any) -> float:
    import torch

    x = torch.empty(1 << 30, device=device, dtype=torch.uint8)
    y = torch.empty_like(x)
    return 2 * x.numel() / _time(lambda: y.copy_(x), repeats=10)


def _elementwise(device: Any) -> tuple[float, float]:
    """Bytes per second of a bf16 sum of two same-shape tensors, and of a
    product whose second operand is broadcast over the leading axes (a
    rotary table over the heads)."""
    import torch

    a = torch.randn(8, 32, 4096, 64, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    table = torch.randn(4096, 64, device=device, dtype=torch.bfloat16)
    same = 3 * a.numel() * a.element_size() / _time(lambda: a + b)
    broadcast = 2 * a.numel() * a.element_size() / _time(lambda: a * table)
    return same, broadcast


def _attention(device: Any) -> dict[str, float]:
    """FLOP/s of fused attention with grouped key/value heads, by head
    width and in f32 (`causal/64`, `causal/64/f32`): `causal` counts the
    half of the scores its kernel computes, `masked` all; `_backward` the
    same count over the backward pass's time."""
    import torch
    from torch.nn import functional

    heads, kv_heads = 32, 8
    rates: dict[str, float] = {}
    for dtype, suffix, length in ((torch.bfloat16, "", 4096), (torch.float32, "/f32", 2048)):
        mask = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        for width in (64, 128):
            q = torch.randn(1, heads, length, width, device=device, dtype=dtype)
            k = torch.randn(1, kv_heads, length, width, device=device, dtype=dtype)
            v = torch.randn_like(k)
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            flops = 4 * heads * length * length * width
            for kind, options, work in (
                ("causal", {"is_causal": True}, flops / 2),
                ("masked", {"attn_mask": mask}, flops),
            ):

                def forward(
                    options: dict[str, Any] = options, q: Any = q, k: Any = k, v: Any = v
                ) -> Any:
                    return functional.scaled_dot_product_attention(
                        q, k, v, enable_gqa=True, **options
                    )

                with torch.no_grad():
                    rates[f"{kind}/{width}{suffix}"] = work / _time(forward)
                out = forward()
                grad = torch.randn_like(out)
                inputs = (q, k, v)
                rates[f"{kind}_backward/{width}{suffix}"] = work / _time(
                    lambda out=out, grad=grad, inputs=inputs: torch.autograd.grad(
                        out, inputs, grad, retain_graph=True
                    )
                )
    return rates


def _decode(device: Any) -> float:
    """Bytes of cache per second one query per row reads under a mask, as
    the generated code runs it: two products around an f32 softmax."""
    import torch

    rows, heads, kv_heads, cache, width = 16, 32, 8, 8192, 128
    query = torch.randn(
        rows, kv_heads, heads // kv_heads, width, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(rows, kv_heads, cache, width, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.ones(rows, 1, 1, cache, dtype=torch.bool, device=device)

    def attend() -> Any:
        scores = torch.matmul(query, key.transpose(-1, -2)).float() * 0.088
        scores = scores.masked_fill(~mask, -1e30)
        weights = torch.softmax(scores, dim=-1).to(value.dtype)
        return torch.matmul(weights, value)

    with torch.no_grad():
        seconds = _time(attend)
    return 2 * key.numel() * key.element_size() / seconds


def _kernel(device: Any) -> float:
    """Device seconds of a small kernel (one decoding step's hidden rows)
    issued eagerly, timed behind long products so that the host is ahead
    and the kernels run back to back."""
    import torch

    big = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    product = torch.empty_like(big)
    x = torch.randn(16, 4096, device=device, dtype=torch.bfloat16)
    best = float("inf")
    for _ in range(5):
        torch.cuda.synchronize()
        for _ in range(24):
            torch.matmul(big, big, out=product)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(2000):
            x.add_(1.0)
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1000 / 2000)
    return best


def _collectives(device: Any) -> tuple[float, float, float]:
    """A small NCCL all-reduce's seconds, a large one's bus bandwidth, and
    the seconds one token's sum takes as the generated code calls it
    (`linnet.torch.collectives`, its one-shot kernel where it runs) with the
    next call reading it, host included."""
    import torch
    import torch.distributed as dist

    from ..torch import collectives

    dist.init_process_group("nccl")
    group = dist.group.WORLD
    small = torch.ones(2048, device=device)
    latency = _time(lambda: dist.all_reduce(small), repeats=50)
    large = torch.ones(64 << 20, device=device)
    world = dist.get_world_size()
    seconds = _time(lambda: dist.all_reduce(large), repeats=5)
    collectives.prepare(group)
    hidden = torch.ones(1, 4096, device=device, dtype=torch.bfloat16)

    def reduced() -> Any:
        # The next call reads the sum, as in a layer.
        return collectives.all_reduce(hidden, group) + 1

    for _ in range(10):
        reduced()
    torch.cuda.synchronize()
    dist.barrier()
    begin = time.perf_counter()
    for _ in range(200):
        reduced()
    torch.cuda.synchronize()
    reduce = (time.perf_counter() - begin) / 200
    dist.destroy_process_group()
    # Bus bandwidth: what each link carries in a ring all-reduce.
    return latency, 2 * (world - 1) / world * large.numel() * 4 / seconds, reduce


def measure(name: str) -> dict[str, Any]:
    import torch

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    _, total = torch.cuda.mem_get_info()
    peaks, sizes = _products(device)
    same, broadcast = _elementwise(device)
    found: dict[str, Any] = {
        "name": name,
        "source": f"measured: {torch.cuda.get_device_name()}, PyTorch {torch.__version__}",
        "memory": int(total),
        "flops": peaks,
        "products": sizes,
        "bandwidth": _bandwidth(device),
        "elementwise": same,
        "broadcast": broadcast,
        "attention": _attention(device),
        "decode": _decode(device),
        "dispatch": _host(device),
        "kernel": _kernel(device),
        "link": 0.0,
        "latency": 0.0,
        "reduce": 0.0,
    }
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        found["latency"], found["link"], found["reduce"] = _collectives(device)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m linnet.resources.calibrate")
    parser.add_argument("--name", default="measured", help="the profile's device name")
    parser.add_argument("--output", help="write the profile here as JSON")
    args = parser.parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        print("linnet: calibration needs a CUDA device", file=sys.stderr)
        return 1
    found = measure(args.name)
    if os.environ.get("RANK", "0") != "0":
        return 0
    text = json.dumps(found, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
