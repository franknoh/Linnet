"""Routed MXFP4 experts on CUDA: each (row, choice) pair's input times its
expert, the experts in MXFP4 as gpt-oss publishes them (`blocks`
[E, Out, G, 16] u8, `scales` [E, Out, G] u8).

With OpenAI's `triton_kernels` installed -- the Triton package of grouped
matrix products for mixtures of experts that vLLM runs gpt-oss with -- its
MXFP4 product reads the four-bit weights in a layout swizzled for the GPU.
Otherwise the experts are dequantized to bf16 and `torch._grouped_mm`
multiplies them. Either copy is made on the first call and kept for the
weights it came from, until they are written again.

`mxfp4_grouped` is a custom op, so a compiled step calls it as it is and a
CUDA graph captures its kernels.
"""

# PyTorch's custom-op registry and `triton_kernels` carry no complete types.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingImports=false, reportMissingTypeStubs=false

from __future__ import annotations

import importlib.util
from typing import Any

import torch

_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# The weights' storage -> (their versions, the copy made from them).
_swizzled: dict[int, tuple[tuple[int, int], Any, Any]] = {}
_dequantized: dict[int, tuple[tuple[int, int], torch.Tensor]] = {}
_kernels: list[bool] = []


def available(x: torch.Tensor) -> bool:
    """Whether `mxfp4_grouped` runs for `x`: bf16 on a CUDA GPU of compute
    capability 9 or later (Hopper), where `triton_kernels`' product or
    `torch._grouped_mm` does."""
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and hasattr(torch, "_grouped_mm")
        and torch.cuda.get_device_capability(x.device)[0] >= 9
    )


def _triton_kernels() -> bool:
    if not _kernels:
        _kernels.append(importlib.util.find_spec("triton_kernels") is not None)
    return _kernels[0]


def _versions(blocks: torch.Tensor, scales: torch.Tensor) -> tuple[int, int]:
    # Binding new weights copies into the same tensors, which bumps these.
    return (blocks._version, scales._version)  # pyright: ignore[reportPrivateUsage]


def swizzled(blocks: torch.Tensor, scales: torch.Tensor) -> tuple[Any, Any]:
    """The experts as `triton_kernels` reads them: the bytes `[E, In / 2,
    Out]` and the scales `[E, In / 32, Out]`, each in its Hopper layout, and
    the precision config that pairs them."""
    versions = _versions(blocks, scales)
    entry = _swizzled.get(blocks.data_ptr())
    if entry is None or entry[0] != versions:
        from triton_kernels.matmul import PrecisionConfig
        from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
        from triton_kernels.tensor_details import layout

        count, out_features, groups, _ = blocks.shape
        quant = blocks.reshape(count, out_features, groups * 16).transpose(-2, -1)
        weight = convert_layout(
            wrap_torch_tensor(quant, dtype=FP4),
            layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2),
        )
        scale = convert_layout(
            wrap_torch_tensor(scales.transpose(-2, -1)),
            layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=-2, num_warps=8),
        )
        precision = PrecisionConfig(b_mx_scale=scale, b_microblock_size=32)
        entry = (versions, weight, precision)
        _swizzled[blocks.data_ptr()] = entry
    return entry[1], entry[2]


def dequantized(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Every expert's weight in bf16, `[E, Out, G * 32]`, one expert at a
    time so the f32 values of only one are ever there."""
    versions = _versions(blocks, scales)
    entry = _dequantized.get(blocks.data_ptr())
    if entry is None or entry[0] != versions:
        count, out_features, groups, _ = blocks.shape
        table = torch.tensor(_FP4, dtype=torch.float32, device=blocks.device)
        weight = torch.empty(
            count, out_features, groups * 32, dtype=torch.bfloat16, device=blocks.device
        )
        for e in range(count):
            nibbles = torch.stack([blocks[e] & 15, blocks[e] >> 4], dim=-1).long()
            factor = torch.exp2(scales[e].float() - 127)[..., None]
            values = table[nibbles].reshape(out_features, groups, 32) * factor
            weight[e] = values.reshape(out_features, groups * 32).to(torch.bfloat16)
        entry = (versions, weight)
        _dequantized[blocks.data_ptr()] = entry
    return entry[1]


@torch.library.custom_op("linnet::mxfp4_grouped", mutates_args=())
def mxfp4_grouped(
    x: torch.Tensor,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    experts: torch.Tensor,
    shared: bool,
) -> torch.Tensor:
    """`y[r, k, o] = sum_i x[row, i] * W[experts[r, k], o, i]`, `[R, K, Out]`,
    with `row` the pair's row of `x` ([R, In]) when `shared` and the pair
    itself otherwise (`x` [R, K, In]). The pairs are sorted by expert, each
    expert multiplied by its pairs' inputs, and the products put back in
    order. No host synchronization: a CUDA graph holds it."""
    rows, chosen = experts.shape
    count, out_features = blocks.shape[0], blocks.shape[1]
    flat = experts.reshape(-1)
    pairs = flat.numel()
    # Each pair's place among the pairs sorted by expert, from a running
    # count per expert rather than a sort: `order[i]` is the pair at sorted
    # position `i`.
    # Experts by pairs, so the running count is a scan along the inner axis
    # (along the outer one, PyTorch's scan is many times slower).
    one_hot = torch.arange(count, device=flat.device)[:, None] == flat[None, :]
    counts = one_hot.sum(1, dtype=torch.int32)
    rank = one_hot.cumsum(1, dtype=torch.int32).gather(0, flat[None, :]).reshape(-1) - 1
    starts = counts.cumsum(0, dtype=torch.int32) - counts
    place = (starts[flat] + rank).long()
    order = torch.empty(pairs, dtype=torch.long, device=flat.device)
    order.scatter_(0, place, torch.arange(pairs, device=flat.device))
    inputs = x if shared else x.reshape(pairs, -1)
    sources = order // chosen if shared else order
    if _triton_kernels():
        from triton_kernels.matmul import matmul
        from triton_kernels.tensor_details.ragged_tensor import make_ragged_tensor_metadata

        weight, precision = swizzled(blocks, scales)
        # Row `i` of the product reads input `sources[i]` and is written to
        # pair `order[i]`: the products come back in the pairs' order.
        y = matmul(
            inputs,
            weight,
            None,
            a_ragged_metadata=make_ragged_tensor_metadata(counts, pairs),
            gather_indx=sources.to(torch.int32),
            scatter_indx=order.to(torch.int32),
            precision_config=precision,
        )
        return y.reshape(rows, chosen, out_features)
    weight = dequantized(blocks, scales)
    grouped_mm: Any = getattr(torch, "_grouped_mm")  # noqa: B009 - private, checked by `available`
    y = grouped_mm(
        inputs[sources], weight.transpose(-2, -1), offs=counts.cumsum(0, dtype=torch.int32)
    )
    unsorted = torch.empty_like(y).index_copy_(0, order, y)
    return unsorted.reshape(rows, chosen, out_features)


@mxfp4_grouped.register_fake
def _(
    x: torch.Tensor,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    experts: torch.Tensor,
    shared: bool,
) -> torch.Tensor:
    rows, chosen = experts.shape
    return x.new_empty(rows, chosen, blocks.shape[1])


__all__ = ["available", "dequantized", "mxfp4_grouped", "swizzled"]
