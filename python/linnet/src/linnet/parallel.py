"""Tensor parallelism: which axis of each weight is split across devices.

A model runs tensor-parallel when its large weights are split across the
devices of a mesh, each device computing its share and the framework adding
the collectives the split needs (XLA's partitioner for `linnet.jax`, DTensor
for `linnet.torch`). Any split computes the same numbers; the split only
decides how much crosses between devices. The defaults follow the usual
layout for decoders: projections into heads or into the feed-forward width
are split by output (each device computes some heads, some of the hidden
units), projections back out of them by input (each device holds the rows
its heads produce, and the partial sums are added), and everything else --
embeddings, norms, the output head -- is copied to every device.

Rules are glob patterns over parameter paths, mapped to the axis to split or
`None` for a copy; the first match wins, and a path no rule matches is
copied. An axis a weight's extent does not divide evenly is not split.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping, Sequence

# Linear weights are [out, in]: axis 0 splits the output, axis 1 the input.
DEFAULT_RULES: dict[str, int | None] = {
    "*q_proj.weight": 0,
    "*k_proj.weight": 0,
    "*v_proj.weight": 0,
    "*qkv.weight": 0,
    "*gate_proj.weight": 0,
    "*up_proj.weight": 0,
    "*gate_up.weight": 0,
    "*q_proj.bias": 0,
    "*k_proj.bias": 0,
    "*v_proj.bias": 0,
    "*qkv.bias": 0,
    "*o_proj.weight": 1,
    "*down_proj.weight": 1,
    "*mlp.down.weight": 1,
    "*mlp.up.weight": 0,
    "*mlp.gate.weight": 0,
}

# A KV cache is [rows, heads, positions, width]: split by heads, as the
# key and value projections that fill it are.
STATE_AXIS = 1


def split_axis(
    path: str,
    shape: Sequence[int],
    devices: int,
    rules: Mapping[str, int | None] | None = None,
) -> int | None:
    """The axis of the weight at `path` to split over `devices`, or `None`
    to copy it to each."""
    table = DEFAULT_RULES if rules is None else rules
    for pattern, axis in table.items():
        if fnmatch.fnmatchcase(path, pattern):
            if axis is None or axis >= len(shape) or shape[axis] % devices != 0:
                return None
            return axis
    return None


def state_axis(shape: Sequence[int], devices: int) -> int | None:
    """The axis of a state member (a KV cache) to split, or `None`."""
    if len(shape) == 4 and shape[STATE_AXIS] % devices == 0:
        return STATE_AXIS
    return None


__all__ = ["DEFAULT_RULES", "split_axis", "state_axis"]
