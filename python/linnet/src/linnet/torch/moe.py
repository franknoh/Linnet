"""Routed MXFP4 experts on CUDA: each (row, choice) pair's input times its
expert, the experts in MXFP4 as gpt-oss publishes them (`blocks`, FP4
E2M1 held as their bytes [E, Out, G, 16] u8, `scales` [E, Out, G] u8).

With OpenAI's `triton_kernels` installed -- the Triton package of grouped
matrix products for mixtures of experts that vLLM runs gpt-oss with -- its
MXFP4 product reads the four-bit weights in a layout swizzled for the GPU.
Otherwise the experts are dequantized to bf16 and `torch._grouped_mm`
multiplies them. Either copy is made on the first call and kept for the
weights it came from, until they are written again.

`mxfp4_grouped` routes the pairs with plain tensor arithmetic, which a
compiled step fuses, and multiplies them in the custom op
`linnet::mxfp4_grouped`, which it calls as it is and a CUDA graph captures.
"""

# PyTorch's custom-op registry and `triton_kernels` carry no complete types.
# pyright: reportMissingImports=false, reportUnknownVariableType=false

from __future__ import annotations

import importlib.util
from typing import Protocol

import torch

# The values of the sixteen FP4 (E2M1) codes.
FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# The weights' storage -> (their versions, the copy made from them). The
# swizzled copy is `triton_kernels`' own weight and precision config, which
# are only passed back to its `matmul`.
_swizzled: dict[int, tuple[tuple[int, int], object, object]] = {}
_dequantized: dict[int, tuple[tuple[int, int], torch.Tensor]] = {}
_kernels: list[bool] = []


def grouped_mm_native(device: torch.device) -> bool:
    """Whether `torch._grouped_mm` has a kernel of its own on `device`:
    Hopper and the data-center Blackwells (compute capability 9 and 10).
    Elsewhere (the GB10's 12.1, say) PyTorch runs it group by group from
    the host, which reads the offsets back and so fails in a CUDA graph."""
    return (
        hasattr(torch, "_grouped_mm")
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] in (9, 10)
    )


def available(x: torch.Tensor) -> bool:
    """Whether `mxfp4_grouped` runs for `x`: bf16 on a CUDA GPU of compute
    capability 9 or later (Hopper), where `triton_kernels`' product runs,
    or `torch._grouped_mm` without it."""
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(x.device)[0] >= 9
        and (_triton_kernels() or grouped_mm_native(x.device))
    )


def mxfp4_values(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """MXFP4 weights in f32: `blocks` `[..., G, 16]` u8, two codes a byte,
    the low nibble first, and `scales` `[..., G]` u8, a power of two biased by
    127 for each group of 32, to `[..., G * 32]`."""
    table = torch.tensor(FP4, dtype=torch.float32, device=blocks.device)
    values = torch.stack([table[(blocks & 15).long()], table[(blocks >> 4).long()]], dim=-1)
    factor = torch.exp2(scales.float() - 127)[..., None]
    return (values.reshape(*blocks.shape[:-1], 32) * factor).reshape(*blocks.shape[:-2], -1)


def _triton_kernels() -> bool:
    if not _kernels:
        _kernels.append(importlib.util.find_spec("triton_kernels") is not None)
    return _kernels[0]


def _versions(blocks: torch.Tensor, scales: torch.Tensor) -> tuple[int, int]:
    # Binding new weights copies into the same tensors, which bumps these.
    return (blocks._version, scales._version)


def swizzled(blocks: torch.Tensor, scales: torch.Tensor) -> tuple[object, object]:
    """The experts as `triton_kernels` reads them: the bytes `[E, In / 2,
    Out]` and the scales `[E, In / 32, Out]`, each in the layout of the GPU's
    generation, and the precision config that pairs them. On compute
    capability 12 (the GB10, the RTX 50s) they stay strided: `triton_kernels`
    takes those for a data-center Blackwell and lays the data out for its
    persistent kernel, whose TMA scatters (`.tile::scatter4`) they lack."""
    versions = _versions(blocks, scales)
    entry = _swizzled.get(blocks.data_ptr())
    if entry is None or entry[0] != versions:
        from triton_kernels.matmul import PrecisionConfig
        from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
        from triton_kernels.tensor_details import layout

        count, out_features, groups, _ = blocks.shape
        quant = blocks.reshape(count, out_features, groups * 16).transpose(-2, -1)
        if torch.cuda.get_device_capability(blocks.device)[0] == 12:
            weight_layout = scale_layout = layout.StridedLayout(-2)  # pyright: ignore[reportUnknownMemberType]
        else:
            weight_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2)  # pyright: ignore[reportUnknownMemberType]
            scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=-2, num_warps=8)  # pyright: ignore[reportUnknownMemberType]
        weight = convert_layout(wrap_torch_tensor(quant, dtype=FP4), weight_layout)
        scale = convert_layout(wrap_torch_tensor(scales.transpose(-2, -1)), scale_layout)
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
        weight = torch.empty(
            count, out_features, groups * 32, dtype=torch.bfloat16, device=blocks.device
        )
        for e in range(count):
            weight[e] = mxfp4_values(blocks[e], scales[e]).to(torch.bfloat16)
        entry = (versions, weight)
        _dequantized[blocks.data_ptr()] = entry
    return entry[1]


def routes(experts: torch.Tensor, count: int, shared: bool) -> tuple[torch.Tensor, ...]:
    """The (row, choice) pairs of `experts` ([R, K]) sorted by expert: the
    pair at each sorted position (`order`), the row of the input it reads
    (`sources`: the pair's row when `shared`, the pair itself otherwise), and
    each expert's number of pairs (`counts`), all int32. Each pair is placed
    by a running count per expert rather than a sort. Plain tensor arithmetic,
    so a compiled step fuses it."""
    chosen = experts.shape[1]
    flat = experts.reshape(-1)
    pairs = flat.numel()
    # Experts by pairs, so the running count is a scan along the inner axis
    # (along the outer one, PyTorch's scan is many times slower).
    one_hot = torch.arange(count, device=flat.device)[:, None] == flat[None, :]
    counts = one_hot.sum(1, dtype=torch.int32)
    rank = one_hot.cumsum(1, dtype=torch.int32).gather(0, flat[None, :]).reshape(-1) - 1
    starts = counts.cumsum(0, dtype=torch.int32) - counts
    place = (starts[flat] + rank).long()
    order = torch.empty(pairs, dtype=torch.long, device=flat.device)
    order.scatter_(0, place, torch.arange(pairs, device=flat.device))
    sources = order // chosen if shared else order
    return order.to(torch.int32), sources.to(torch.int32), counts


class _GroupedMM(Protocol):
    """`torch._grouped_mm`, which older PyTorch lacks."""

    def __call__(
        self, input: torch.Tensor, mat2: torch.Tensor, *, offs: torch.Tensor
    ) -> torch.Tensor: ...


@torch.library.custom_op("linnet::mxfp4_grouped", mutates_args=())
def mxfp4_grouped_routed(
    x: torch.Tensor,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    order: torch.Tensor,
    sources: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Each sorted pair's input row `x[sources[i]]` times its expert's MXFP4
    weight, written to row `order[i]` of the `[pairs, Out]` result, the
    pairs sorted as `routes` gives them. No host synchronization: a CUDA
    graph holds it."""
    pairs = order.numel()
    if _triton_kernels():
        from triton_kernels.matmul import matmul
        from triton_kernels.tensor_details.ragged_tensor import make_ragged_tensor_metadata

        weight, precision = swizzled(blocks, scales)
        return matmul(
            x,
            weight,
            None,
            a_ragged_metadata=make_ragged_tensor_metadata(counts, pairs),
            gather_indx=sources,
            scatter_indx=order,
            precision_config=precision,
        )
    weight = dequantized(blocks, scales)
    grouped_mm: _GroupedMM = getattr(torch, "_grouped_mm")  # noqa: B009 - private, checked by `available`
    y = grouped_mm(x[sources], weight.transpose(-2, -1), offs=counts.cumsum(0, dtype=torch.int32))
    return torch.empty_like(y).index_copy_(0, order.long(), y)


@mxfp4_grouped_routed.register_fake  # pyright: ignore[reportUnknownMemberType]
def _(
    x: torch.Tensor,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    order: torch.Tensor,
    sources: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty(order.numel(), blocks.shape[1])


def mxfp4_grouped(
    x: torch.Tensor,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    experts: torch.Tensor,
    shared: bool,
) -> torch.Tensor:
    """`y[r, k, o] = sum_i x[row, i] * W[experts[r, k], o, i]`, `[R, K, Out]`,
    with `row` the pair's row of `x` ([R, In]) when `shared` and the pair
    itself otherwise (`x` [R, K, In]): the pairs routed (`routes`, which a
    compiled step fuses) and each expert multiplied by its pairs' inputs
    (`linnet::mxfp4_grouped`, called as it is)."""
    rows, chosen = experts.shape
    order, sources, counts = routes(experts, blocks.shape[0], shared)
    inputs = x if shared else x.reshape(rows * chosen, -1)
    y = mxfp4_grouped_routed(inputs, blocks, scales, order, sources, counts)
    return y.reshape(rows, chosen, blocks.shape[1])


__all__ = [
    "available",
    "dequantized",
    "grouped_mm_native",
    "mxfp4_grouped",
    "mxfp4_grouped_routed",
    "routes",
    "swizzled",
]
