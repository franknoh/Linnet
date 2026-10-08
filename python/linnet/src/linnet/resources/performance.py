"""How long a step takes on a device, operation by operation.

Each step of the graph is costed at the device's measured rates
(`python -m linnet.resources.calibrate`):

- matrix products at the rate of their weight's size, or reading their
  operands at memory bandwidth, whichever is longer;
- fused attention at its kernel's rate (causal, or under a mask), and one
  query under a mask as the generated code runs it: two products around a
  softmax, bound by memory;
- collectives at the link's bandwidth plus a latency each;
- everything else that makes a tensor reads its inputs and writes its
  outputs at the bandwidth an elementwise kernel reaches, lower when an
  operand is broadcast;
- views take no device time.

The calls are the generated code's: what it computes once per shape is left
out, an inference entry's products that read the same input run as one
(their weights joined), and a causal mask runs as `is_causal`.

The generated code runs eagerly: the host issues one call after another,
`dispatch` seconds each, and the device runs each kernel once it is issued
and the previous one is done. A step with small kernels waits on the host;
a step with large ones keeps the host ahead. Training adds the backward pass
(products twice, attention at its backward rate, other kernels reading the
gradient and writing one) and one optimizer pass a step. A pipeline adds its
bubble, `(stages - 1) / (micro-batches + stages - 1)` of the step.

What is not modeled: kernels below their measured rate at unusual shapes,
the host's own variance, overlap of communication with computation, and
`torch.compile`'s fusion (compiled models drop the host time only).
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from . import expr as ex
from .graph import Category, TensorGraph

MIB = 1 << 20


class _Profile(TypedDict):
    """A profile as `python -m linnet.resources.calibrate` writes it."""

    name: str
    memory: int
    flops: dict[str, float]
    bandwidth: float
    link: NotRequired[float | None]
    latency: NotRequired[float | None]
    attention: NotRequired[dict[str, float]]
    decode: NotRequired[float]
    elementwise: NotRequired[float]
    broadcast: NotRequired[float]
    products: NotRequired[dict[str, list[list[float]]]]  # [n, k, FLOP/s] each
    skinny: NotRequired[float]
    host: NotRequired[dict[str, float]]
    dispatch: NotRequired[float]
    kernel: NotRequired[float]
    strided_kernel: NotRequired[float]
    reduce: NotRequired[float | None]
    source: NotRequired[str]


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """A device's rates, as `python -m linnet.resources.calibrate` measures
    them.

    - `flops`: matrix products' FLOP/s by compute dtype; `products`: the
      same by the weight's shape (`(n, k, FLOP/s)` for 8192 rows times an
      `n` by `k` weight); `skinny`: the bytes per second a product of a few
      rows (a decoding step's) reads its weight at.
    - `bandwidth`: a large copy's bytes read plus written per second;
      `elementwise` and `broadcast`: the same for an operation on two
      same-shape tensors and for one with a broadcast operand.
    - `attention`: fused attention's FLOP/s by kind, head width and, past
      bf16, dtype (`causal/64`, `causal/64/f32`): `causal` counts the half of the scores its kernel
      computes, `masked` all of them, and `_backward` rates the same count
      over the backward pass's time. `decode`: the bytes of cache one query
      per row reads per second under a mask, as the generated code runs it.
    - `dispatch`: the host's seconds per eager call, over a decoder layer's
      mix; `host`: the same by kind of call (`view`, `elementwise`,
      `product`, ...) and per autograd node (`backward`,
      `product_backward`); `kernel`: a small kernel's device seconds,
      issued eagerly, and `strided_kernel` one reading a strided view.
    - `link`: an all-reduce's bus bandwidth; `latency`: a small one's
      seconds; `reduce`: one token's as the generated code calls it with the
      next call reading it, host included.
    - `memory`: what the driver reports; `source`: where the figures come
      from."""

    name: str
    memory: int
    flops: Mapping[str, float]
    bandwidth: float
    link: float
    latency: float
    attention: Mapping[str, float] = field(default_factory=dict[str, float])
    decode: float = 0.0
    elementwise: float = 0.0
    broadcast: float = 0.0
    products: Mapping[str, tuple[tuple[float, float, float], ...]] = field(
        default_factory=dict[str, tuple[tuple[float, float, float], ...]]
    )
    skinny: float = 0.0
    host: Mapping[str, float] = field(default_factory=dict[str, float])
    dispatch: float = 0.0
    kernel: float = 0.0
    strided_kernel: float = 0.0
    reduce: float = 0.0
    source: str = "data sheet"

    def peak(self, dtype: str) -> float:
        return self.flops.get(dtype, self.flops["f32"])

    def product(self, dtype: str, outputs: int, width: int) -> float:
        """A matrix product's FLOP/s by a weight of `outputs` rows of
        `width`: between the measured shapes on a log scale, the nearest
        edge past them, and in proportion below them."""
        peak = self.peak(dtype)
        measured = self.products.get("bf16" if dtype == "f16" else dtype, ())
        if not measured:
            return peak
        table = {(n, k): rate for n, k, rate in measured}

        def axis(values: list[float], x: float) -> tuple[float, float, float, float]:
            """The grid points around `x`, its share of the way between
            them, and the scale below the grid."""
            if x <= values[0]:
                return values[0], values[0], 0.0, x / values[0]
            for low, high in itertools.pairwise(values):
                if x <= high:
                    return low, high, math.log(x / low) / math.log(high / low), 1.0
            return values[-1], values[-1], 0.0, 1.0

        n0, n1, tn, sn = axis(sorted({n for n, _, _ in measured}), outputs)
        k0, k1, tk, sk = axis(sorted({k for _, k, _ in measured}), width)

        def at(n: float, k: float) -> float:
            return table.get((n, k), peak)

        rate = (
            (1 - tn) * (1 - tk) * at(n0, k0)
            + tn * (1 - tk) * at(n1, k0)
            + (1 - tn) * tk * at(n0, k1)
            + tn * tk * at(n1, k1)
        )
        return min(rate * sn * sk, peak)

    def host_of(self, *kinds: str) -> float:
        """The host's seconds for eager calls of these kinds."""
        return sum(self.host.get(kind, self.dispatch) for kind in kinds)

    def attention_rate(self, kind: str, width: int, dtype: str = "bf16") -> float:
        """Fused attention's rate in `dtype` at the measured head width
        nearest `width`."""
        suffix = "" if dtype in ("bf16", "f16") else f"/{dtype}"
        widths = [
            int(parts[1])
            for parts in (key.split("/") for key in self.attention)
            if len(parts) > 1 and parts[0] == kind and "/".join(["", *parts[2:]]) == suffix
        ]
        if widths:
            nearest = min(widths, key=lambda w: abs(math.log(w / max(width, 1))))
            return self.attention[f"{kind}/{nearest}{suffix}"]
        return self.attention.get(kind) or self.peak(dtype)

    @classmethod
    def load(cls, path: str | Path) -> DeviceSpec:
        """A profile `python -m linnet.resources.calibrate` wrote."""
        data: _Profile = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=str(data["name"]),
            memory=int(data["memory"]),
            flops={k: float(v) for k, v in data["flops"].items()},
            bandwidth=float(data["bandwidth"]),
            link=float(data.get("link") or 0.0) or _H100.link,
            latency=float(data.get("latency") or 0.0) or _H100.latency,
            attention={k: float(v) for k, v in data.get("attention", {}).items()},
            decode=float(data.get("decode", 0.0)),
            elementwise=float(data.get("elementwise", 0.0)),
            broadcast=float(data.get("broadcast", 0.0)),
            products={
                dtype: tuple((float(n), float(k), float(r)) for n, k, r in points)
                for dtype, points in data.get("products", {}).items()
            },
            skinny=float(data.get("skinny", 0.0)),
            host={k: float(v) for k, v in data.get("host", {}).items()},
            dispatch=float(data.get("dispatch", 0.0)),
            kernel=float(data.get("kernel", 0.0)),
            strided_kernel=float(data.get("strided_kernel", 0.0)),
            reduce=float(data.get("reduce") or 0.0) or _H100.reduce,
            source=str(data.get("source", "measured")),
        )


# Measured on an H100 SXM with PyTorch 2.14.1 and CUDA 13 by
# `python -m linnet.resources.calibrate`; link, latency and reduce between
# two of them under torchrun (NVLink, NCCL 2.30).
_H100 = DeviceSpec(
    "h100-80gb",
    85017493504,
    {"bf16": 749.3e12, "f16": 749.3e12, "f32": 41.17e12},
    3.037e12,
    286.1e9,
    17.4e-6,
    attention={
        "causal/64": 305.18e12,
        "causal_backward/64": 107.18e12,
        "masked/64": 193.85e12,
        "masked_backward/64": 76.31e12,
        "causal/128": 449.64e12,
        "causal_backward/128": 142.55e12,
        "masked/128": 272.78e12,
        "masked_backward/128": 101.91e12,
        "causal/64/f32": 22.60e12,
        "causal_backward/64/f32": 9.33e12,
        "masked/64/f32": 24.63e12,
        "masked_backward/64/f32": 11.25e12,
        "causal/128/f32": 33.38e12,
        "causal_backward/128/f32": 10.54e12,
        "masked/128/f32": 33.73e12,
        "masked_backward/128/f32": 12.09e12,
    },
    decode=1.946e12,
    elementwise=3.033e12,
    broadcast=1.415e12,
    products={
        "bf16": (
            (256, 256, 112.4e12),
            (256, 1024, 392.0e12),
            (256, 4096, 528.8e12),
            (1024, 256, 302.9e12),
            (1024, 1024, 543.7e12),
            (1024, 4096, 723.5e12),
            (4096, 256, 414.0e12),
            (4096, 1024, 650.7e12),
            (4096, 4096, 762.0e12),
            (16384, 256, 481.3e12),
            (16384, 1024, 664.1e12),
            (16384, 4096, 729.4e12),
        ),
        "f32": (
            (256, 256, 27.2e12),
            (256, 1024, 38.5e12),
            (256, 4096, 40.8e12),
            (1024, 256, 35.7e12),
            (1024, 1024, 40.1e12),
            (1024, 4096, 40.5e12),
            (4096, 256, 30.3e12),
            (4096, 1024, 39.9e12),
            (4096, 4096, 40.8e12),
            (16384, 256, 36.9e12),
            (16384, 1024, 40.3e12),
            (16384, 4096, 41.2e12),
        ),
    },
    skinny=2.725e12,
    host={
        "view": 1.21e-6,
        "elementwise": 6.26e-6,
        "product": 14.46e-6,
        "batched": 28.44e-6,
        "attention": 20.02e-6,
        "normalization": 10.10e-6,
        "write": 11.07e-6,
        "gather": 9.33e-6,
        "join": 7.87e-6,
        "cast": 7.17e-6,
        "reduction": 6.50e-6,
        "mask": 14.71e-6,
        "backward": 9.39e-6,
        "product_backward": 45.19e-6,
    },
    dispatch=5.95e-6,
    kernel=2.07e-6,
    strided_kernel=3.25e-6,
    reduce=20.6e-6,
    source="measured: H100 SXM, PyTorch 2.14.1+cu130",
)


def _from_sheet(
    name: str, memory: int, bf16: float, f32: float, bandwidth: float, link: float
) -> DeviceSpec:
    """A device known by its data sheet: the H100's measured rates scaled by
    the ratio of the two data sheets, and the H100 host's dispatch."""
    compute = bf16 / 989.4e12
    memory_ratio = bandwidth / 3.35e12
    return DeviceSpec(
        name,
        memory,
        {
            "bf16": _H100.flops["bf16"] * compute,
            "f16": _H100.flops["f16"] * compute,
            "f32": _H100.flops["f32"] * f32 / 66.9e12,
        },
        _H100.bandwidth * memory_ratio,
        _H100.link * link / 450e9,
        _H100.latency,
        attention={
            k: v * (f32 / 66.9e12 if k.endswith("/f32") else compute)
            for k, v in _H100.attention.items()
        },
        decode=_H100.decode * memory_ratio,
        elementwise=_H100.elementwise * memory_ratio,
        broadcast=_H100.broadcast * memory_ratio,
        products={
            dtype: tuple(
                (n, k, r * (compute if dtype == "bf16" else f32 / 66.9e12)) for n, k, r in points
            )
            for dtype, points in _H100.products.items()
        },
        skinny=_H100.skinny * memory_ratio,
        host=_H100.host,
        dispatch=_H100.dispatch,
        kernel=_H100.kernel,
        strided_kernel=_H100.strided_kernel,
        reduce=_H100.reduce,
        source="data sheet, at the H100's measured share of its own",
    )


DEVICES: dict[str, DeviceSpec] = {
    d.name: d
    for d in (
        _H100,
        _from_sheet("h200", 143771 * MIB, 989.4e12, 66.9e12, 4.8e12, 450e9),
        _from_sheet("a100-80gb", 81920 * MIB, 312e12, 19.5e12, 2.039e12, 300e9),
        _from_sheet("a100-40gb", 40960 * MIB, 312e12, 19.5e12, 1.555e12, 300e9),
        _from_sheet("l4", 23034 * MIB, 121e12, 30.3e12, 300e9, 32e9),
    )
}


def device(name: str, devices: int = 1, *, refresh: bool = False) -> DeviceSpec:
    """A device by its name in `DEVICES`, a calibrated profile's path, or
    `local`: the GPU this process uses, measured once on this machine and
    kept (`linnet.resources.calibrate.profile`, with the collectives
    between `devices` of them; `refresh` measures again)."""
    if name == "local":
        from .calibrate import profile

        return profile(devices, refresh=refresh)
    if name in DEVICES:
        return DEVICES[name]
    if Path(name).is_file():
        return DeviceSpec.load(name)
    known = ", ".join(sorted(DEVICES))
    raise ValueError(
        f"unknown device `{name}`: one of {known}, `local`, or a calibrated profile's path"
    )


@dataclass(frozen=True, slots=True)
class Throughput:
    """A step's predicted time and what sets it.

    `tokens` are the positions one step processes over every replica;
    `compute`, `memory` and `communication` are the slowest device's
    kernel seconds limited by each, `host` the seconds it waits on the
    host to issue them, and `bubble` the share of a pipelined step its
    stages sit idle."""

    tokens: int
    seconds: float
    compute: float
    memory: float
    communication: float
    bubble: float
    limit: Literal["compute", "memory", "communication", "host"]
    host: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": self.tokens,
            "seconds": self.seconds,
            "tokens_per_second": self.tokens_per_second,
            "compute_seconds": self.compute,
            "memory_seconds": self.memory,
            "communication_seconds": self.communication,
            "host_seconds": self.host,
            "bubble": self.bubble,
            "limit": self.limit,
        }


_PRODUCTS = ("torch.nn.functional.linear", "torch.matmul")
_WRITES = ("torch.Tensor.index_copy", "torch.Tensor.index_put", "torch.Tensor.index_put(tokens)")
_COLLECTIVES = ("torch.distributed.all_reduce", "torch.distributed.all_gather")
_LOSSES = ("linnet.linear_cross_entropy", "linnet.linear_token_log_probs")
# A vocabulary split across processes: each runs the loss over its part.
_SPLIT_LOSSES = {
    "linnet.split_cross_entropy": "linnet.linear_cross_entropy",
    "linnet.split_token_log_probs": "linnet.linear_token_log_probs",
}
# The chunked loss's passes over its f32 logits, bytes per logit: making
# them (bf16, then f32) and their log-sum-exp; with gradients also the
# softmax, its scaling and its cast back.
_LOSS_BYTES = 16
_LOSS_GRAD_BYTES = 46
_LOSS_BLOCK_BYTES = 1 << 30  # f32 logits per block, as `linnet.torch.loss`


@dataclass(slots=True)
class _Op:
    """Device work and the host time it takes to issue: `seconds` on the
    device, which of compute, memory or communication sets it, and `host`
    seconds of eager calls."""

    host: float
    seconds: float
    by: Literal["compute", "memory", "communication"]


@dataclass(slots=True)
class _Clock:
    """The host issuing calls and the device running them, in order."""

    eager: bool
    host: float = 0.0
    device: float = 0.0
    spent: dict[str, float] = field(
        default_factory=lambda: {"compute": 0.0, "memory": 0.0, "communication": 0.0}
    )

    def run(self, op: _Op) -> None:
        if self.eager:
            self.host += op.host
        self.device = max(self.device, self.host) + op.seconds
        self.spent[op.by] += op.seconds


def _bytes(graph: TensorGraph, id: int, env: Mapping[str, int]) -> int:
    return ex.evaluate(graph.objects[id].nbytes, env)


def _numel(graph: TensorGraph, id: int, env: Mapping[str, int]) -> int:
    return math.prod(ex.evaluate(d, env) for d in graph.objects[id].shape)


def _dims(graph: TensorGraph, id: int, env: Mapping[str, int]) -> list[int]:
    return [ex.evaluate(d, env) for d in graph.objects[id].shape]


def _causal(graph: TensorGraph, mask: int) -> bool:
    """A square mask `causal_mask` made, which the generated code passes as
    `is_causal`."""
    owner = graph.objects[graph.objects[mask].storage]
    made = graph.steps[owner.producer] if owner.producer is not None else None
    shape = graph.objects[mask].shape
    return made is not None and made.label == "causal_mask" and shape[-1] == shape[-2]


def _hoisted(graph: TensorGraph) -> set[int]:
    """The steps the generated code runs once per shape (`constants`), not
    per call: those reading neither an input nor a persistent object."""
    varying = set(graph.inputs) | {o.id for o in graph.objects if o.persistent}
    hoisted: set[int] = set()
    for step in graph.steps:
        if any(i in varying or graph.objects[i].storage in varying for i in step.inputs):
            varying.update(step.outputs)
        else:
            hoisted.add(step.index)
    return hoisted


def _strided(graph: TensorGraph, seeds: set[int]) -> set[int]:
    """Values whose elements do not lie in order: `seeds`, a permute, a
    slice, a broadcast, and views of those."""
    found = set(seeds)
    for step in graph.steps:
        for out in step.outputs:
            if graph.objects[out].owns_storage:
                continue
            if step.kind in ("permute", "slice", "broadcast") or any(
                i in found for i in step.inputs
            ):
                found.add(out)
    return found


def _fused(graph: TensorGraph) -> dict[int, list[int]]:
    """Products the generated inference code runs as one: those reading the
    same input, each by a weight of its own, whose weights `prepare` joins
    (the query, key and value projections, say). Keyed by the first."""
    groups: dict[int, list[int]] = {}
    for step in graph.steps:
        if step.implementation != "torch.nn.functional.linear" or len(step.inputs) != 2:
            continue
        weight = graph.objects[step.inputs[1]]
        if weight.category == Category.PARAMETER and weight.persistent:
            groups.setdefault(step.inputs[0], []).append(step.index)
    return {members[0]: members for members in groups.values() if len(members) > 1}


def _costs(
    graph: TensorGraph,
    env: Mapping[str, int],
    device: DeviceSpec,
    processes: int,
    training: bool,
    fuse: bool = True,
) -> tuple[list[_Op], list[_Op]]:
    """The device work of each call the generated code makes for a step,
    and the host's time to issue it: forward, and in training backward."""
    forward: list[_Op] = []
    backward: list[_Op] = []
    hoisted = _hoisted(graph)
    # Split models and pipeline stages keep their weights apart.
    fused = _fused(graph) if fuse and not training and processes == 1 else {}
    joined = {i for members in fused.values() for i in members[1:]}

    stream = device.elementwise or device.bandwidth
    spread = device.broadcast or stream
    host = device.host_of
    node = host("backward")
    # A joined product's parts are slices of its result.
    strided = _strided(
        graph, {graph.steps[i].outputs[0] for members in fused.values() for i in members}
    )

    def memory(nbytes: float, kernels: int = 1, rate: float = stream) -> float:
        # The unvectorized kernel's floor is its own.
        floor = device.strided_kernel if rate == spread and device.strided_kernel else device.kernel
        return max(nbytes / rate, kernels * floor)

    def bounded(
        flops: float,
        rate: float,
        nbytes: float,
        issued: float,
        kernels: int = 1,
        read_rate: float = 0.0,
    ) -> _Op:
        compute = flops / rate
        moved = nbytes / (read_rate or device.bandwidth)
        floor = kernels * device.kernel
        if compute >= moved:
            return _Op(issued, max(compute, floor), "compute")
        return _Op(issued, max(moved, floor), "memory")

    for step in graph.steps:
        if step.index in hoisted or step.index in joined:
            continue
        implementation = (step.implementation or "").split("(")[0]
        implementation = _SPLIT_LOSSES.get(implementation, implementation)
        full = step.implementation or ""
        owned = [o for o in step.outputs if graph.objects[o].owns_storage]
        read = sum(_bytes(graph, i, env) for i in dict.fromkeys(step.inputs))
        written = sum(_bytes(graph, o, env) for o in owned)
        flops = float(ex.evaluate(step.flops, env))
        operands = len(step.inputs)
        if implementation in _PRODUCTS and operands >= 2:
            members = fused.get(step.index, [step.index])
            if len(members) > 1:
                steps = [graph.steps[i] for i in members]
                flops = sum(float(ex.evaluate(s.flops, env)) for s in steps)
                read = _bytes(graph, step.inputs[0], env) + sum(
                    _bytes(graph, s.inputs[1], env) for s in steps
                )
                written = sum(_bytes(graph, o, env) for s in steps for o in s.outputs)
            weight = _dims(graph, step.inputs[1], env)
            source = _dims(graph, step.inputs[0], env)
            batched = implementation == "torch.matmul" and len(weight) > 2
            if implementation == "torch.matmul":
                outputs, width = (weight[-1], weight[-2]) if len(weight) >= 2 else (1, 1)
            else:
                outputs = sum(_dims(graph, graph.steps[i].inputs[1], env)[-2] for i in members)
                width = weight[-1] if weight else 1
            rows = math.prod(source[:-1]) if source else 1
            dtype = graph.objects[step.inputs[0]].dtype
            # A joined product is one call, then a slice for each part.
            issued = host("batched" if batched else "product")
            if len(members) > 1:
                issued += host(*["view"] * len(members))
            # A few rows (a decoding step's) read the weight at its own rate.
            skinny = device.skinny if rows <= 64 else 0.0
            op = bounded(
                flops,
                device.product(dtype, outputs, width),
                read + written,
                issued,
                read_rate=skinny,
            )
            forward.append(op)
            backward.append(_Op(host("product_backward"), 2 * op.seconds, op.by))
        elif implementation == "torch.nn.functional.scaled_dot_product_attention":
            inputs = step.inputs
            query = _dims(graph, inputs[0], env)
            masked = operands > 3 and graph.objects[inputs[3]].dtype == "bool"
            causal = masked and _causal(graph, inputs[3])
            cache = sum(_bytes(graph, i, env) for i in inputs[1:3])
            if masked and not causal and query[-2] == 1:
                # One query under a mask: two products around an f32
                # softmax, as the generated code writes it.
                calls = host(
                    *["view"] * 4,
                    "batched",
                    "batched",
                    "cast",
                    "cast",
                    "elementwise",
                    "elementwise",
                    "mask",
                    "reduction",
                )
                seconds = cache / (device.decode or device.bandwidth)
                forward.append(_Op(calls, max(seconds, 8 * device.kernel), "memory"))
                backward.append(_Op(8 * node, 2 * seconds, "memory"))
                continue
            kind = "masked" if masked and not causal else "causal"
            work = flops / 2 if causal else flops
            # Exact numerics run it in f32.
            dtype = graph.objects[inputs[0]].dtype if "(input dtype)" in full else "f32"
            rate = device.attention_rate(kind, query[-1], dtype)
            backward_rate = device.attention_rate(f"{kind}_backward", query[-1], dtype)
            moved = (read + written) / device.bandwidth
            compute = work / rate
            forward.append(
                _Op(
                    host("attention"),
                    max(compute, moved, device.kernel),
                    "compute" if compute >= moved else "memory",
                )
            )
            backward.append(
                _Op(
                    node + host("attention"),
                    max(work / backward_rate, 2 * moved, device.kernel),
                    "compute",
                )
            )
        elif implementation == "torch.distributed.shared":
            if processes <= 1 or not training:
                continue  # its argument, as it is
            # The shards' parts of its gradient summed, in backward.
            size = _bytes(graph, step.inputs[0], env)
            seconds = device.latency + 2 * (processes - 1) / processes * size / device.link
            backward.append(_Op(host() + node, seconds, "communication"))
        elif implementation in _COLLECTIVES:
            if processes <= 1:
                continue  # the generated helper returns its argument
            size = _bytes(graph, step.outputs[0], env)
            reduce = implementation == "torch.distributed.all_reduce"
            share = (processes - 1) / processes
            carried = (2 if reduce else 1) * share
            seconds = device.latency + carried * size / device.link
            # The generated code's sum costs the host more than a call:
            # `reduce` is measured with the call that reads it.
            issued = (
                max(device.reduce - host("elementwise"), 0.0)
                if reduce and device.reduce
                else host()
            )
            forward.append(_Op(issued, seconds, "communication"))
            backward.append(_Op(issued + node, seconds, "communication"))
        elif implementation in _LOSSES:
            hidden, weight = _dims(graph, step.inputs[0], env), _dims(graph, step.inputs[1], env)
            rows = math.prod(hidden[:-1])
            logits = rows * weight[0]
            rate = device.product(graph.objects[step.inputs[0]].dtype, weight[0], weight[1])
            gradients = implementation == "linnet.linear_cross_entropy" and training
            # A block of rows at a time: its products, then passes over its
            # f32 logits, one kernel after another; with gradients the
            # weight's gradient is summed in f32 every block as well.
            blocks = -(-rows // max(1, _LOSS_BLOCK_BYTES // (4 * weight[0])))
            calls = (3 if gradients else 1) * host("product") + (11 if gradients else 5) * host(
                "elementwise"
            )
            summed = blocks * weight[0] * weight[1] * 20
            with_grads = _Op(
                blocks * calls,
                3 * flops / rate + (logits * _LOSS_GRAD_BYTES + summed) / stream,
                "compute",
            )
            if gradients:
                forward.append(with_grads)
                backward.append(_Op(node + 2 * host("elementwise"), memory(2 * read), "memory"))
            else:
                forward.append(
                    _Op(blocks * calls, flops / rate + logits * _LOSS_BYTES / stream, "compute")
                )
                if training:
                    # Log-probabilities keep their log-sum-exp and recompute
                    # the logits for the gradients.
                    backward.append(with_grads)
        elif implementation == "torch.nn.functional.embedding":
            forward.append(_Op(host("gather"), memory(2 * written), "memory"))
            # A dense gradient the table's size, rows added into it.
            table = _bytes(graph, step.inputs[1], env)
            backward.append(
                _Op(node + 2 * host("elementwise"), memory(table + 3 * written, 2), "memory")
            )
        elif implementation in _WRITES:
            update = _bytes(graph, step.inputs[1], env)
            # The generated write's index arguments are calls of their own:
            # a position as a long, and per row an `arange` and a slice.
            calls = {
                "torch.Tensor.index_copy": ("write", "view", "cast"),
                "torch.Tensor.index_put": ("write", "elementwise", "cast", "view"),
            }.get(full, ("write", "cast", "cast", "view", "view"))
            forward.append(_Op(host(*calls), memory(2 * update), "memory"))
        elif implementation in ("torch.rms_norm", "torch.nn.functional.layer_norm"):
            # The fast normalizations are two calls: the normalization, then
            # times the (broadcast) weight.
            seconds = memory(2 * written) + memory(2 * written, rate=spread)
            forward.append(_Op(host("normalization", "elementwise"), seconds, "memory"))
            backward.append(_Op(2 * node, 2 * seconds, "memory"))
        elif owned:
            # An operand broadcast, or a strided view (a permute, a slice of
            # a joined product), takes PyTorch's unvectorized kernel.
            largest = max(_numel(graph, o, env) for o in owned)
            slow = any(0 < _numel(graph, i, env) < largest or i in strided for i in step.inputs)
            seconds = memory(read + written, rate=spread if slow else stream)
            kind = {"cast": "cast", "concat": "join"}.get(step.kind, "elementwise")
            forward.append(_Op(host(kind), seconds, "memory"))
            backward.append(_Op(node, 2 * seconds, "memory"))
        else:
            forward.append(_Op(host("view"), 0.0, "memory"))  # a view
            backward.append(_Op(node, 0.0, "memory"))
    return forward, backward


@dataclass(frozen=True, slots=True)
class _Part:
    seconds: float  # one micro-batch through the stage, forward and backward
    host: float
    spent: Mapping[str, float]
    optimizer: float  # once a step
    once: float = 0.0  # also once a step: a stage's sharded weights gathered


def _part(
    graph: TensorGraph,
    env: Mapping[str, int],
    device: DeviceSpec,
    *,
    training: bool,
    processes: int,
    compiled: bool,
    received: int = 0,
    recomputed: float = 0.0,
    optimizer_states: int = 2,
    staged: bool = False,
    shards: int = 1,
) -> _Part:
    """One device's share of one (micro-)step. `received` is what crosses
    into it from the stage before, which in training sends a gradient as
    large back; `recomputed` is the share of the forward pass checkpointing
    runs again. With `shards`, each weight is gathered where it is read,
    forward and backward, and its gradient summed back into the parts
    (`linnet.torch.fsdp`); a pipeline's stage (`staged`) gathers its
    weights once a step and sums each micro-batch's gradients."""
    forward, backward = _costs(graph, env, device, processes, training, fuse=not staged)
    clock = _Clock(not compiled)
    once = 0.0
    summing: _Op | None = None
    if shards > 1 and training:
        owned = [o for o in graph.objects if o.category == Category.PARAMETER and o.owns_storage]
        weights = sum(ex.evaluate(o.nbytes, env) for o in owned)
        # Summed back in f32, the parts' dtype.
        summed = sum(4 * ex.evaluate(ex.product(o.shape), env) for o in owned)
        share = (shards - 1) / shards
        if staged:
            once = share * weights / device.link + len(owned) * device.latency
            summing = _Op(
                len(owned) * device.dispatch,
                share * summed / device.link + len(owned) * device.latency,
                "communication",
            )
        else:
            # Gathered twice in the model's dtype.
            moved = share * (2 * weights + summed) / device.link
            clock.run(_Op(0.0, moved + 3 * len(owned) * device.latency, "communication"))
    transfer = _Op(device.host_of(), device.latency + received / device.link, "communication")
    if received:
        clock.run(transfer)
    for op in forward:
        clock.run(op)
    optimizer = 0.0
    if training:
        if recomputed:
            for op in forward:
                clock.run(_Op(op.host * recomputed, op.seconds * recomputed, op.by))
        for op in reversed(backward):
            clock.run(op)
        if summing is not None:
            clock.run(summing)
        if received:
            clock.run(transfer)
        weights = sum(
            ex.evaluate(o.nbytes, env)
            for o in graph.objects
            if o.category == Category.PARAMETER and o.owns_storage
        )
        # The update reads each weight and its gradient, writes the weight,
        # and reads and writes each state.
        optimizer = (3 + 2 * optimizer_states) * weights / device.bandwidth
    busy = sum(clock.spent.values())
    return _Part(clock.device, max(clock.device - busy, 0.0), dict(clock.spent), optimizer, once)


def step_time(
    parts: Sequence[tuple[TensorGraph, Mapping[str, int], int, float]],
    device: DeviceSpec,
    *,
    tokens: int,
    training: bool,
    processes: int = 1,
    microbatches: int = 1,
    replicas: int = 1,
    gradient_bytes: int = 0,
    compiled: bool = False,
    optimizer_states: int = 2,
    shards: int = 1,
) -> Throughput:
    """The time of a step whose devices run `parts`: one per pipeline stage
    (its graph, the environment of one micro-batch, the bytes it receives
    and the share of its forward pass checkpointing repeats), or one for
    the whole step. `replicas` copies of the layout each take their own
    `tokens`; in training they then sum `gradient_bytes` over the link."""
    stages = [
        _part(
            graph,
            env,
            device,
            training=training,
            processes=processes,
            compiled=compiled,
            received=received,
            recomputed=recomputed,
            optimizer_states=optimizer_states,
            staged=len(parts) > 1,
            shards=shards,
        )
        for graph, env, received, recomputed in parts
    ]
    slowest = max(stages, key=lambda p: p.seconds)
    count = microbatches if len(stages) > 1 else 1
    slots = count + len(stages) - 1
    seconds = slots * slowest.seconds + slowest.optimizer + slowest.once
    bubble = (len(stages) - 1) / slots
    totals = {k: v * slots for k, v in slowest.spent.items()}
    totals["memory"] += slowest.optimizer
    totals["communication"] = totals.get("communication", 0.0) + slowest.once
    totals["host"] = slowest.host * slots
    if training and replicas > 1 and gradient_bytes:
        # Summing gradients overlaps the backward pass; the longer one sets the step.
        summed = 2 * (replicas - 1) / replicas * gradient_bytes / device.link
        if summed > seconds:
            totals["communication"] += summed - seconds
            seconds = summed
    limit: Literal["compute", "memory", "communication", "host"] = max(
        ("compute", "memory", "communication", "host"), key=lambda k: totals[k]
    )  # type: ignore[assignment]
    return Throughput(
        tokens * replicas,
        seconds,
        totals["compute"],
        totals["memory"],
        totals["communication"],
        bubble,
        limit,
        totals["host"],
    )


__all__ = ["DEVICES", "DeviceSpec", "Throughput", "device", "step_time"]
