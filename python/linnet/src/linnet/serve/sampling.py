"""Each row's next token from its logits: the most likely one, or one drawn at
a temperature -- from the `top_k` most likely tokens and then the fewest
whose probabilities add up to `top_p`, when those are set (the order
Transformers' `generate` filters in).

A drawn token is the argmax of the scaled, filtered logits plus Gumbel noise,
which draws from their softmax. The noise is a hash of the request's seed,
the position the token takes, and the token's id, computed alike in PyTorch,
JAX, and NumPy: what a request draws depends on its seed alone, not on the
requests that share its batch or on when it was admitted.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import jax
    import numpy as np
    import torch
    from numpy.typing import ArrayLike, NDArray

GREEDY, SAMPLED, FILTERED = 0, 1, 2  # what a pass has to compute, by `mode`

_MASK = 0xFFFFFFFF
_FIRST = 0x7FEB352D
_SECOND = 0x846CA68B
# The second multiplier less 2**32: the same low 32 bits in a product, and
# a 32-bit value times it stays inside int64, which PyTorch computes in.
_SECOND_SIGNED = _SECOND - (1 << 32)
# A hash's top 23 bits, plus a half, times this is uniform in (0, 1) and exact
# in f32: 1 itself never comes up, whose Gumbel noise would be infinite.
_STEP = 1.0 / (1 << 23)


@dataclass(frozen=True)
class Sampling:
    """How a request's tokens are drawn: greedily at temperature 0."""

    temperature: float = 0.0
    top_k: int = 0  # 0 keeps every token
    top_p: float = 1.0
    seed: int = 0

    @property
    def key(self) -> int:
        """The seed, hashed to the 32 bits the noise starts from."""
        return mix(self.seed & _MASK)

    def check(self) -> None:
        if not (math.isfinite(self.temperature) and self.temperature >= 0):
            raise ValueError(f"temperature must be at least 0, not {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be at least 0, not {self.top_k}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], not {self.top_p}")


def mode(rows: Iterable[Sampling]) -> int:
    """What drawing for `rows` needs: `GREEDY` (an argmax), `SAMPLED`
    (noise), or `FILTERED` (noise and a sort for top-k or top-p)."""
    need = GREEDY
    for row in rows:
        if row.temperature > 0:
            if row.top_k > 0 or row.top_p < 1:
                return FILTERED
            need = SAMPLED
    return need


def mix(x: int) -> int:
    """A 32-bit integer hash (Wellons' lowbias32); a bijection."""
    x ^= x >> 16
    x = (x * _FIRST) & _MASK
    x ^= x >> 15
    x = (x * _SECOND) & _MASK
    return x ^ (x >> 16)


# ---- PyTorch: int64 tensors holding 32-bit values


def _mix_torch(x: torch.Tensor) -> torch.Tensor:
    x = x ^ (x >> 16)
    x = (x * _FIRST) & _MASK
    x = x ^ (x >> 15)
    x = (x * _SECOND_SIGNED) & _MASK
    return x ^ (x >> 16)


def draw_torch(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    keys: torch.Tensor,
    at: torch.Tensor,
    need: int,
) -> torch.Tensor:
    """Each row's token: `logits` `[N, Vocab]`, one temperature, top-k, top-p,
    seed key (int64), and position `at` for each row."""
    import torch

    greedy = logits.argmax(-1)
    if need == GREEDY:
        return greedy
    sampled = temperature > 0
    scores = logits.float() / torch.where(sampled, temperature, 1.0)[:, None]
    vocab = scores.shape[-1]
    if need == FILTERED:
        ordered = scores.sort(-1, descending=True).values
        rank = torch.arange(vocab, device=scores.device)
        kept = rank[None, :] < torch.where(top_k > 0, top_k, vocab)[:, None]
        probabilities = ordered.masked_fill(~kept, -math.inf).softmax(-1)
        kept &= (probabilities.cumsum(-1) - probabilities) < top_p[:, None]
        floor = ordered.gather(-1, (kept.sum(-1) - 1)[:, None])
        scores = scores.masked_fill(scores < floor, -math.inf)
    rows = _mix_torch(keys ^ at.long())
    ids = torch.arange(vocab, device=scores.device)
    bits = _mix_torch(rows[:, None] ^ ids[None, :])
    uniform = ((bits >> 9).float() + 0.5) * _STEP
    drawn = (scores - torch.log(-torch.log(uniform))).argmax(-1)
    return torch.where(sampled, drawn, greedy)


# ---- JAX: uint32 arrays, which wrap as the hash wants


class _JaxNumPy(Protocol):
    """`jax.numpy`, as `_mix_jax` uses it."""

    def uint32(self, value: int, /) -> jax.Array: ...


def _mix_jax(jnp: _JaxNumPy, x: jax.Array) -> jax.Array:
    x = x ^ (x >> 16)
    x = x * jnp.uint32(_FIRST)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(_SECOND)
    return x ^ (x >> 16)


def draw_jax(
    logits: jax.Array,
    temperature: jax.Array,
    top_k: jax.Array,
    top_p: jax.Array,
    keys: jax.Array,
    at: jax.Array,
    need: int,
) -> jax.Array:
    """`draw_torch` for JAX, with `keys` as uint32; `need` is static."""
    import jax
    import jax.numpy as jnp_module

    # JAX's stubs leave `jnp.arange` partially unknown, which strict checking refuses.
    jnp: Any = jnp_module
    greedy = jnp.argmax(logits, -1).astype(jnp.int32)
    if need == GREEDY:
        return greedy
    sampled = temperature > 0
    scores = logits.astype(jnp.float32) / jnp.where(sampled, temperature, 1.0)[:, None]
    vocab = scores.shape[-1]
    if need == FILTERED:
        ordered = -jnp.sort(-scores, axis=-1)
        rank = jnp.arange(vocab)
        kept = rank[None, :] < jnp.where(top_k > 0, top_k, vocab)[:, None]
        probabilities = jax.nn.softmax(jnp.where(kept, ordered, -jnp.inf), axis=-1)
        kept &= (jnp.cumsum(probabilities, -1) - probabilities) < top_p[:, None]
        count = jnp.sum(kept, -1)
        floor = jnp.take_along_axis(ordered, (count - 1)[:, None], -1)
        scores = jnp.where(scores < floor, -jnp.inf, scores)
    rows = _mix_jax(jnp, keys ^ at.astype(jnp.uint32))
    ids = jnp.arange(vocab, dtype=jnp.uint32)
    bits = _mix_jax(jnp, rows[:, None] ^ ids[None, :])
    uniform = ((bits >> 9).astype(jnp.float32) + 0.5) * _STEP
    drawn = jnp.argmax(scores - jnp.log(-jnp.log(uniform)), -1).astype(jnp.int32)
    return jnp.where(sampled, drawn, greedy)


# ---- NumPy, for logits on the host


class _NumPy(Protocol):
    """`numpy`, as `_mix_numpy` uses it."""

    @property
    def uint32(self) -> type[np.uint32]: ...


def _mix_numpy(np: _NumPy, x: NDArray[np.uint32]) -> NDArray[np.uint32]:
    x = x ^ (x >> np.uint32(16))
    x = x * np.uint32(_FIRST)
    x = x ^ (x >> np.uint32(15))
    x = x * np.uint32(_SECOND)
    return x ^ (x >> np.uint32(16))


def draw_numpy(
    logits: NDArray[np.floating], rows: list[Sampling], at: ArrayLike, need: int
) -> NDArray[np.intp]:
    """`draw_torch` over NumPy, each row's sampling as it is."""
    import numpy as np

    greedy: NDArray[np.intp] = np.argmax(logits, -1)
    if need == GREEDY:
        return greedy
    temperature = np.asarray([r.temperature for r in rows], dtype=np.float32)
    sampled = temperature > 0
    scores = logits.astype(np.float32) / np.where(sampled, temperature, 1.0)[:, None]
    vocab: int = scores.shape[-1]
    if need == FILTERED:
        top_k = np.asarray([r.top_k for r in rows], dtype=np.int64)
        top_p = np.asarray([r.top_p for r in rows], dtype=np.float32)
        ordered = -np.sort(-scores, axis=-1)
        kept = np.arange(vocab)[None, :] < np.where(top_k > 0, top_k, vocab)[:, None]
        masked = np.where(kept, ordered, -np.inf)
        exponents = np.exp(masked - masked[:, :1])
        probabilities = exponents / exponents.sum(-1, keepdims=True)
        kept &= (np.cumsum(probabilities, -1) - probabilities) < top_p[:, None]
        floor = np.take_along_axis(ordered, (kept.sum(-1) - 1)[:, None], -1)
        scores = np.where(scores < floor, -np.inf, scores)
    keys = np.asarray([r.key for r in rows], dtype=np.uint32)
    with np.errstate(over="ignore"):
        mixed = _mix_numpy(np, keys ^ np.asarray(at).astype(np.uint32))
        bits = _mix_numpy(np, mixed[:, None] ^ np.arange(vocab, dtype=np.uint32)[None, :])
    uniform = ((bits >> np.uint32(9)).astype(np.float32) + 0.5) * np.float32(_STEP)
    drawn: NDArray[np.intp] = np.argmax(scores - np.log(-np.log(uniform)), -1)
    return np.where(sampled, drawn, greedy)
