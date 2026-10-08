"""Low-rank adapters (LoRA), the same under every framework: a linear
weight `W` [out, in] gains `A` [rank, in] (random) and `B` [out, rank]
(zero) in its block, named `lora_a` and `lora_b`, and the layer computes
`x @ W.T + (x @ A.T) @ B.T * alpha / rank`. `B` starting at zero leaves
the output unchanged until it trains."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DOWN = "lora_a"
UP = "lora_b"
# Glob patterns over parameter paths that pick every adapter.
PATTERNS = (f"*.{DOWN}", f"*.{UP}")


def is_adapter(path: str) -> bool:
    """Whether the parameter at `path` is an adapter's `A` or `B`."""
    return path.endswith((f".{DOWN}", f".{UP}"))


def initial_down(
    rank: int, in_features: int, seed_draws: np.random.Generator
) -> NDArray[np.float32]:
    """`A`'s first values: uniform in +-1/sqrt(in), in f32. Every framework
    draws them from the same NumPy generator, so one seed gives the same
    adapters everywhere."""
    bound = 1.0 / np.sqrt(in_features)
    return seed_draws.uniform(-bound, bound, (rank, in_features)).astype(np.float32)


__all__ = ["DOWN", "PATTERNS", "UP", "initial_down", "is_adapter"]
