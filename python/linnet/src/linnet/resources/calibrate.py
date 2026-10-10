"""A device's rates, measured: what `linnet fit --maximize throughput`
predicts with.

    python -m linnet.resources.calibrate --name my-h100 --output my-h100.json
    torchrun --nproc-per-node 2 -m linnet.resources.calibrate --output my-h100.json
    linnet fit model --maximize throughput --device my-h100.json

or let `linnet fit --device local` measure the GPU it runs on, once per
machine (`profile`).

Each rate is the best of several timed runs after a warm-up:

- matrix products in bf16 and f32 (TF32 off, PyTorch's default), at the
  shapes of a transformer's projections and over a grid of weight shapes,
  and the bandwidth a 16-row product reads its weight at;
- the bandwidth of a large copy, of a sum of two tensors, and of a product
  with a broadcast operand;
- fused attention, forward and backward, at head widths 64 and 128, in
  bf16 and f32: causal
  (`is_causal`, the flash kernel) and under a boolean mask (the
  memory-efficient kernel with a bias), with grouped key/value heads;
- one query per row under a mask, as the generated code runs it, as the
  cache's bytes read per second;
- the host time of one eager call: over a decoder layer's mix of calls on
  tiny tensors, and for each kind of call and autograd node alone; and the
  device time of a small kernel issued eagerly;
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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    import torch
    from torch.distributed import ProcessGroup

    from .performance import DeviceSpec


def _time(run: Callable[[], object], repeats: int = 20) -> float:
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


def _host(device: torch.device) -> float:
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

    def rotate(t: torch.Tensor) -> torch.Tensor:
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


def _host_by_kind(device: torch.device) -> dict[str, float]:
    """Host seconds of one eager call of each kind the generated code makes,
    on tensors too small for the device to matter; `backward` and
    `product_backward` are autograd's per node of a chain, each product by a
    weight of its own."""
    import torch
    from torch.nn import functional

    bf16 = torch.bfloat16
    x = torch.randn(1, 1, 64, device=device, dtype=bf16)
    w = torch.randn(64, 64, device=device, dtype=bf16)
    kv = torch.randn(1, 4, 64, 16, device=device, dtype=bf16)
    q = torch.randn(1, 2, 2, 16, device=device, dtype=bf16)
    cache = torch.randn(4, 2, 64, 16, device=device, dtype=bf16)
    value = torch.randn(4, 2, 16, device=device, dtype=bf16)
    rows = torch.arange(4, device=device)
    at = torch.zeros(4, dtype=torch.long, device=device)
    scores = torch.randn(1, 2, 2, 64, device=device)
    mask = torch.ones(1, 1, 1, 64, dtype=torch.bool, device=device)
    tokens = torch.zeros(1, 1, dtype=torch.long, device=device)
    table = torch.randn(100, 64, device=device, dtype=bf16)
    calls: dict[str, Callable[[], object]] = {
        "view": lambda: x.reshape(1, 4, 16),
        "elementwise": lambda: x * x,
        "product": lambda: functional.linear(x, w),
        "batched": lambda: torch.matmul(q, kv[:, :2].transpose(-1, -2)),
        "attention": lambda: functional.scaled_dot_product_attention(kv, kv, kv, is_causal=True),
        "normalization": lambda: torch.rms_norm(x, [64], eps=1e-6),
        "write": lambda: torch.ops.aten.index_put_(cache, [rows, None, at], value),
        "gather": lambda: functional.embedding(tokens, table),
        "join": lambda: torch.cat([x, x], dim=-1),
        "cast": lambda: x.float(),
        "reduction": lambda: torch.softmax(scores, dim=-1),
        "mask": lambda: scores.masked_fill(mask, -1e30),
    }
    found: dict[str, float] = {}
    for kind, call in calls.items():
        for _ in range(50):
            call()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(5):
            begin = time.perf_counter()
            for _ in range(500):
                call()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - begin) / 500)
        found[kind] = best

    def backward(steps: list[Callable[[torch.Tensor], torch.Tensor]]) -> float:
        best = float("inf")
        for _ in range(5):
            y = x.detach().requires_grad_(True)
            for step in steps:
                y = step(y)
            loss = y.float().sum()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            torch.autograd.backward(loss)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - begin)
        return best

    def per_node(make: Callable[[int], Callable[[torch.Tensor], torch.Tensor]]) -> float:
        """A node's share of a long chain's backward against a short one's,
        so that starting the backward pass counts for neither."""
        long, short = 200, 50
        slope = backward([make(i) for i in range(long)]) - backward([make(i) for i in range(short)])
        return slope / (long - short)

    weights = [w.detach().clone().requires_grad_(True) for _ in range(200)]
    found["backward"] = per_node(lambda i: lambda y: y * 1.5)
    found["product_backward"] = per_node(
        lambda i: lambda y, weight=weights[i]: functional.linear(y, weight)
    )
    return found


# The decoder `_generated` runs: small enough that every step waits on the
# host, with the calls of a full-size one.
_DECODER = {"Vocab": 256, "H": 64, "Heads": 4, "KvHeads": 2, "Inner": 128, "Layers": 4}


def _generated(device: torch.device, host: Mapping[str, float], dispatch: float) -> float:
    """How much more the generated code's calls cost the host than the
    kinds' costs measured alone add up to: a small decoder's eager decoding
    step (`calibration.linnet`) on the host, against the sum
    `linnet.resources.performance` makes of the same step's calls."""
    import torch

    from .. import torch as linnet_torch
    from .analysis import MemoryModel
    from .config import ExecutionConfig
    from .performance import DeviceSpec, _costs  # pyright: ignore[reportPrivateUsage]

    source = Path(__file__).with_name("calibration.linnet")
    generics: dict[str, int | str] = {**_DECODER, "Batch": 1, "MaxSeq": 64, "T": "bf16"}
    module = linnet_torch.load(source, generics=generics, device=device, compile=True)
    decode: Callable[..., object] = getattr(module, "decode")  # noqa: B009
    token = torch.zeros(1, 1, dtype=torch.int32, device=device)
    pos = torch.tensor(0, dtype=torch.int32, device=device)
    for _ in range(10):
        decode(token, pos)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        begin = time.perf_counter()
        for _ in range(50):
            decode(token, pos)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - begin) / 50)
    model = MemoryModel(
        source,
        ExecutionConfig(entry="decode", batch=1, cache=64, bindings=dict(_DECODER), dtype="bf16"),
    )
    unit = DeviceSpec(
        "unit", 0, {"bf16": 1e30, "f32": 1e30}, 1e30, 1e30, 0.0, host=host, dispatch=dispatch
    )
    counted = sum(op.host for op in _costs(model.graph, model.env(), unit, 1, False)[0])
    return best / counted


def _products(
    device: torch.device,
) -> tuple[dict[str, float], dict[str, list[list[float]]], float]:
    """The best bf16 and f32 rates; both by the weight's shape, `[n, k,
    FLOP/s]` for 8192 rows times an `n` by `k` weight; and the bytes per
    second a product of 16 rows reads its weight at (a decoding step's)."""
    import torch
    from torch.nn import functional

    def seconds(m: int, n: int, k: int, dtype: torch.dtype) -> float:
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(n, k, device=device, dtype=dtype)
        return _time(lambda: functional.linear(a, b))

    def rate(m: int, n: int, k: int, dtype: torch.dtype) -> float:
        return 2 * m * n * k / seconds(m, n, k, dtype)

    shapes = [(8192, 8192, 8192), (8192, 14336, 4096), (8192, 4096, 14336), (4096, 4096, 4096)]
    best = max(rate(m, n, k, torch.bfloat16) for m, n, k in shapes)
    grid = [(n, k) for n in (256, 1024, 4096, 16384) for k in (256, 1024, 4096)]
    sizes = {
        "bf16": [[float(n), float(k), rate(8192, n, k, torch.bfloat16)] for n, k in grid],
        "f32": [[float(n), float(k), rate(8192, n, k, torch.float32)] for n, k in grid],
    }
    peaks = {"bf16": best, "f16": best, "f32": max(r for _, _, r in sizes["f32"])}
    skinny = 14336 * 4096 * 2 / seconds(16, 14336, 4096, torch.bfloat16)
    return peaks, sizes, skinny


def _bandwidth(device: torch.device) -> float:
    import torch

    x = torch.empty(1 << 30, device=device, dtype=torch.uint8)
    y = torch.empty_like(x)
    return 2 * x.numel() / _time(lambda: y.copy_(x), repeats=10)


def _elementwise(device: torch.device) -> tuple[float, float]:
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


class _Masking(TypedDict, total=False):
    """How `_attention` masks the scores: causally, or by a boolean mask."""

    is_causal: bool
    attn_mask: torch.Tensor


def _attention(device: torch.device) -> dict[str, float]:
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
                ("causal", cast(_Masking, {"is_causal": True}), flops / 2),
                ("masked", cast(_Masking, {"attn_mask": mask}), flops),
            ):

                def forward(
                    options: _Masking = options,
                    q: torch.Tensor = q,
                    k: torch.Tensor = k,
                    v: torch.Tensor = v,
                ) -> torch.Tensor:
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


def _decode(device: torch.device) -> float:
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

    def attend() -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-1, -2)).float() * 0.088
        scores = scores.masked_fill(~mask, -1e30)
        weights = torch.softmax(scores, dim=-1).to(value.dtype)
        return torch.matmul(weights, value)

    with torch.no_grad():
        seconds = _time(attend)
    return 2 * key.numel() * key.element_size() / seconds


def _kernel(device: torch.device) -> tuple[float, float]:
    """Device seconds of a small kernel (one decoding step's rows) issued
    eagerly, timed behind long products so that the host is ahead and the
    kernels run back to back: a contiguous one, and one reading a permuted
    view with a broadcast operand (PyTorch's unvectorized kernel)."""
    import torch

    big = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    product = torch.empty_like(big)
    x = torch.randn(16, 4096, device=device, dtype=torch.bfloat16)
    heads = torch.randn(16, 1, 32, 128, device=device, dtype=torch.bfloat16).permute(0, 2, 1, 3)
    row = torch.randn(1, 128, device=device, dtype=torch.bfloat16)

    def queued(call: Callable[[], object]) -> float:
        best = float("inf")
        for _ in range(5):
            torch.cuda.synchronize()
            for _ in range(24):
                torch.matmul(big, big, out=product)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(2000):
                call()
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end) / 1000 / 2000)
        return best

    return queued(lambda: x.add_(1.0)), queued(lambda: heads * row)


def _collectives(device: torch.device) -> tuple[float, float, float]:
    """In an initialized NCCL group: a small all-reduce's seconds, a large
    one's bus bandwidth, and the seconds one token's sum takes as the
    generated code calls it (`linnet.torch.collectives`, its one-shot kernel
    where it runs) with the next call reading it, host included."""
    import torch
    import torch.distributed as dist

    from ..torch import collectives

    group = cast("ProcessGroup", dist.group.WORLD)
    small = torch.ones(2048, device=device)
    latency = _time(lambda: dist.all_reduce(small), repeats=50)
    large = torch.ones(64 << 20, device=device)
    world = dist.get_world_size()
    seconds = _time(lambda: dist.all_reduce(large), repeats=5)
    collectives.prepare(group)
    hidden = torch.ones(1, 4096, device=device, dtype=torch.bfloat16)

    def reduced() -> torch.Tensor:
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
    # Bus bandwidth: what each link carries in a ring all-reduce.
    return latency, 2 * (world - 1) / world * large.numel() * 4 / seconds, reduce


def _spawned(rank: int, world: int, port: int, out: str) -> None:
    """One process of a group `profile` starts to measure the collectives."""
    import torch
    import torch.distributed as dist

    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    # A group of its own, even when started from a `torchrun` process.
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    try:
        found = _collectives(device)
    finally:
        dist.destroy_process_group()
    if rank == 0:
        Path(out).write_text(json.dumps(found), encoding="utf-8")


def measure(name: str, *, collectives: bool = True) -> dict[str, object]:
    """This device's rates; under `torchrun`, with `collectives`, those of
    the collectives between its processes too."""
    import torch

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    _, total = torch.cuda.mem_get_info()
    # The host first: autograd's nodes measure slower after long runs on
    # the device.
    host = _host_by_kind(device)
    dispatch = _host(device)
    scale = _generated(device, host, dispatch)
    small, strided = _kernel(device)
    peaks, sizes, skinny = _products(device)
    same, broadcast = _elementwise(device)
    found: dict[str, object] = {
        "name": name,
        "source": f"measured: {torch.cuda.get_device_name()}, PyTorch {torch.__version__}",
        "memory": int(total),
        "flops": peaks,
        "products": sizes,
        "skinny": skinny,
        "bandwidth": _bandwidth(device),
        "elementwise": same,
        "broadcast": broadcast,
        "attention": _attention(device),
        "decode": _decode(device),
        "dispatch": dispatch,
        # Each kind alone costs the host less than in the generated code:
        # the kinds (not autograd's, measured in a chain already) scaled to a
        # small decoder's step.
        "host": {
            kind: cost if kind in ("backward", "product_backward") else cost * scale
            for kind, cost in host.items()
        },
        "kernel": small,
        "strided_kernel": strided,
        "link": 0.0,
        "latency": 0.0,
        "reduce": 0.0,
    }
    if collectives and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist

        dist.init_process_group("nccl")
        try:
            found["latency"], found["link"], found["reduce"] = _collectives(device)
        finally:
            dist.destroy_process_group()
    return found


def cache_directory() -> Path:
    """Where `profile` keeps the profiles it measured: `$LINNET_CACHE/devices`,
    else `$XDG_CACHE_HOME/linnet/devices`, else `~/.cache/linnet/devices`."""
    base = os.environ.get("LINNET_CACHE")
    if base:
        return Path(base) / "devices"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg or Path.home() / ".cache") / "linnet" / "devices"


def profile(devices: int = 1, *, refresh: bool = False) -> DeviceSpec:
    """The rates of the CUDA device this process uses, measured once on this
    machine and kept: the GPU, PyTorch and the host all count, so the
    profile is named by the three. With `devices` above one and as many
    GPUs here, the collectives between them are measured too (in processes
    of their own); otherwise the H100's figures stand in for them.
    `refresh` measures again."""
    import platform

    import torch

    from .performance import DeviceSpec

    if not torch.cuda.is_available():
        raise RuntimeError("calibration needs a CUDA device")
    count = min(devices, torch.cuda.device_count())
    name = torch.cuda.get_device_name()
    key = "-".join(
        "".join(c if c.isalnum() else "-" for c in part.lower()).strip("-")
        for part in (name, torch.__version__, platform.node())
    )
    path = cache_directory() / f"{key}{'' if count < 2 else f'-x{count}'}.json"
    if path.is_file() and not refresh:
        return DeviceSpec.load(path)
    # The collectives in processes of their own, below: this one may be
    # one of a group already (`validate --time` under `torchrun`).
    found = measure("local", collectives=False)
    found["source"] = f"{found['source']}, on {platform.node()}"
    if count > 1:
        import socket
        import tempfile

        from torch.multiprocessing.spawn import spawn  # pyright: ignore[reportUnknownVariableType]

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as work:
            out = str(Path(work) / "collectives.json")
            spawn(_spawned, args=(count, port, out), nprocs=count)
            found["latency"], found["link"], found["reduce"] = json.loads(
                Path(out).read_text(encoding="utf-8")
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
    return DeviceSpec.load(path)


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
