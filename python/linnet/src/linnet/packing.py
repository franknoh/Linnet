"""Packing training sequences into batches of a fixed number of positions,
with no framework: numpy arrays that `linnet.train` (PyTorch) and
`linnet.jax.train` both take.

`pack(examples, tokens)` lays examples end to end, never splitting one;
each position knows its place in its sequence (`positions`), its sequence
(`segments`), the token it predicts (`targets`) and whether that counts
(`mask`). The rest is padding, so every batch has one shape.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Example:
    """One training sequence: `tokens`, and which of them the model learns
    to predict (`learned[i]` for `tokens[i]`; every token after the first
    when omitted)."""

    tokens: Sequence[int]
    learned: Sequence[bool] | None = None

    @staticmethod
    def prompted(prompt: Sequence[int], completion: Sequence[int]) -> Example:
        """A completion to learn after a prompt that is only read."""
        return Example([*prompt, *completion], [False] * len(prompt) + [True] * len(completion))


@dataclass
class Packed:
    """Sequences packed into `P` positions: token `p` is at `positions[p]` of
    sequence `segments[p]` and predicts `targets[p]` (the next token of its
    sequence) where `mask[p]` is 1. Padding is one more sequence, masked
    out. `items[s]` is sequence `s`'s index among the examples packed."""

    tokens: np.ndarray  # [P] i32
    positions: np.ndarray  # [P] i32
    segments: np.ndarray  # [P] i32
    targets: np.ndarray  # [P] i64
    mask: np.ndarray  # [P] f32
    items: list[int] = field(default_factory=lambda: list[int]())

    @property
    def sequences(self) -> int:
        return len(self.items)

    @property
    def count(self) -> int:
        """The positions that count toward the loss."""
        return int(self.mask.sum())

    def arrays(self, count: float) -> list[np.ndarray]:
        """`loss_packed`'s inputs, the loss weighed to a mean over `count`
        positions (this batch's, or every micro-batch's of one step)."""
        weights = (self.mask / count).astype(np.float32)
        return [self.tokens, self.positions, self.segments, self.targets, weights]


def empty(size: int) -> Packed:
    """A batch that learns nothing, for a process with fewer batches than
    the others to make the same calls."""
    zeros = np.zeros(size, dtype=np.int32)
    return Packed(zeros, zeros, zeros, zeros.astype(np.int64), np.zeros(size, dtype=np.float32))


def pack(
    examples: Iterable[Example],
    tokens: int,
    *,
    max_length: int | None = None,
    together: int = 1,
) -> Iterator[Packed]:
    """Packs `examples` in order into batches of exactly `tokens` positions,
    a sequence never split across two. A sequence longer than `max_length`
    (`tokens` when omitted) is cut to it. Each batch's unused positions are
    padding, so every batch has one shape and the model compiles once.

    `together` keeps each run of that many consecutive examples in one
    batch (a preference pair's two answers, say), each cut to `tokens //
    together` at most, and none left out. Otherwise an example of fewer than
    two tokens, with nothing to predict, is left out."""
    limit = min(tokens, max_length) if max_length is not None else tokens
    if together > 1:
        limit = min(limit, tokens // together)
    pending: list[tuple[int, list[int], list[bool]]] = []
    used = 0
    run: list[tuple[int, list[int], list[bool]]] = []
    for item, example in enumerate(examples):
        sequence = list(example.tokens)[:limit]
        if together == 1 and len(sequence) < 2:
            continue
        learned = (
            list(example.learned)[: len(sequence)]
            if example.learned is not None
            else [False] + [True] * (len(sequence) - 1)
        )
        run.append((item, sequence, learned))
        if len(run) < together:
            continue
        size = sum(len(sequence) for _, sequence, _ in run)
        if used + size > tokens and pending:
            yield _batch(pending, tokens)
            pending, used = [], 0
        pending += run
        used += size
        run = []
    if run:
        size = sum(len(sequence) for _, sequence, _ in run)
        if used + size > tokens and pending:
            yield _batch(pending, tokens)
            pending, used = [], 0
        pending += run
    if pending:
        yield _batch(pending, tokens)


def _batch(sequences: list[tuple[int, list[int], list[bool]]], size: int) -> Packed:
    tokens: list[int] = []
    positions: list[int] = []
    segments: list[int] = []
    targets: list[int] = []
    mask: list[float] = []
    for index, (_, sequence, learned) in enumerate(sequences):
        if not sequence:
            continue  # kept in `items`, with no positions
        tokens += sequence
        positions += range(len(sequence))
        segments += [index] * len(sequence)
        # Position `i` predicts token `i + 1`; the last predicts nothing.
        targets += [*sequence[1:], 0]
        mask += [float(flag) for flag in learned[1:]] + [0.0]
    padding = size - len(tokens)
    tokens += [0] * padding
    positions += [0] * padding
    segments += [len(sequences)] * padding
    targets += [0] * padding
    mask += [0.0] * padding
    return Packed(
        np.asarray(tokens, dtype=np.int32),
        np.asarray(positions, dtype=np.int32),
        np.asarray(segments, dtype=np.int32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(mask, dtype=np.float32),
        [item for item, _, _ in sequences],
    )


# ------------------------------------------------- preferences and prompts


@dataclass(frozen=True)
class Pair:
    """A prompt, the answer preferred, and the one not."""

    prompt: Sequence[int]
    chosen: Sequence[int]
    rejected: Sequence[int]


@dataclass(frozen=True)
class Prompt:
    """A prompt's tokens, and what `reward` needs to score its completions
    (the answer, say)."""

    tokens: Sequence[int]
    data: object = None


Reward = Callable[[Prompt, list[int]], float]


def group_advantages(rewards: Sequence[float], group: int, *, scale: bool = True) -> list[float]:
    """Each reward less its group's mean (`group` consecutive rewards), over
    the group's standard deviation when `scale`."""
    advantages: list[float] = []
    for start in range(0, len(rewards), group):
        chunk = rewards[start : start + group]
        mean = sum(chunk) / len(chunk)
        spread = (
            math.sqrt(sum((r - mean) ** 2 for r in chunk) / (len(chunk) - 1))
            if len(chunk) > 1
            else 0.0
        )
        advantages += [(r - mean) / (spread + 1e-4) if scale else r - mean for r in chunk]
    return advantages


__all__ = ["Example", "Packed", "Pair", "Prompt", "Reward", "empty", "group_advantages", "pack"]
