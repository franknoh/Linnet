"""What the PyTorch trainers (`linnet.train`) and the JAX ones
(`linnet.jax.train`) share apart from their arithmetic: the record of each
step, the directories checkpoints go in, how GRPO samples and scores a
step's completions, and how DPO lays out its pairs."""

from __future__ import annotations

import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .packing import Example, Pair, Prompt, Reward, group_advantages
from .serve import Completion, Request

# ------------------------------------------------------------------ steps


@dataclass
class Step:
    """What one optimizer step did."""

    step: int
    loss: float  # the mean over the step's counted positions
    tokens: int  # positions processed, padding included
    seconds: float
    learning_rate: float | None = None  # as a PyTorch optimizer has it
    grad_norm: float | None = None


@dataclass
class History:
    steps: list[Step] = field(default_factory=lambda: list[Step]())

    @property
    def losses(self) -> list[float]:
        return [step.loss for step in self.steps]


@dataclass
class GrpoStep:
    """What one GRPO step did: `reward` and `reward_std` over its
    completions, `length` their mean token count, `clipped` the share of
    learned tokens whose ratio was clipped, `kl` the mean estimate against
    the reference, `uniform` the share of groups whose rewards were all
    equal (no advantage, so nothing to learn), `mismatch` the mean absolute
    difference of the policy's and the engine's log-probabilities of the
    sampled tokens (with `correction_cap`)."""

    step: int
    reward: float
    reward_std: float
    length: float
    loss: float
    clipped: float
    kl: float | None
    sample_seconds: float
    train_seconds: float
    learning_rate: float | None
    grad_norm: float | None
    uniform: float = 0.0
    mismatch: float | None = None


@dataclass
class DpoStep:
    """What one DPO step did, over its pairs: `accuracy` is the share whose
    chosen answer the model now rewards more, `margin` the mean difference of
    the rewards, `chosen` and `rejected` their means (a reward is `beta`
    times an answer's log-probability less the reference's)."""

    step: int
    loss: float
    accuracy: float
    margin: float
    chosen: float
    rejected: float
    seconds: float
    learning_rate: float | None
    grad_norm: float | None


# ------------------------------------------------------------ checkpoints

PREFIX = "step-"


def checkpoint_path(root: Path, step: int) -> Path:
    """Where the checkpoint of `step` goes under `root`."""
    return root / f"{PREFIX}{step:08d}"


def complete(root: Path, marker: str) -> list[Path]:
    """The checkpoints under `root` that finished writing (`marker`, written
    last, is there), oldest first."""
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.iterdir() if path.name.startswith(PREFIX) and (path / marker).exists()
    )


def prune(root: Path, marker: str, keep: int | None) -> None:
    """Removes all but the `keep` latest complete checkpoints under `root`
    (None keeps every one)."""
    if keep is None:
        return
    for old in complete(root, marker)[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def unsaved(every: int | None, step: int) -> bool:
    """Whether a run that stopped after `step` has yet to write it: one
    written every `every` steps did not just."""
    return not (every and step % every == 0)


# ------------------------------------------------------------------ GRPO


def check_grpo(
    beta: float,
    reference: object,
    correction_cap: float | None,
    temperature: float,
    top_k: int,
    top_p: float,
) -> None:
    """Rejects GRPO options that cannot go together."""
    if beta and reference is None:
        raise ValueError("a KL penalty (`beta`) needs a `reference`")
    if correction_cap is not None and (temperature != 1.0 or top_k or top_p < 1.0):
        raise ValueError(
            "the engine's log-probabilities are of the model's own distribution: correcting "
            "for them takes temperature 1 and no top_k or top_p"
        )


def as_prompt(value: Prompt | Sequence[int]) -> Prompt:
    return value if isinstance(value, Prompt) else Prompt(value)


def sample_requests(
    prompts: Sequence[Prompt],
    group: int,
    draws: random.Random,
    *,
    max_new_tokens: int,
    eos: frozenset[int],
    temperature: float,
    top_k: int,
    top_p: float,
    logprobs: bool,
) -> list[Request]:
    """`group` requests for each prompt, each with a seed of its own from
    `draws`; with `logprobs`, the engine reports its log-probabilities."""
    return [
        Request(
            prompt=list(prompt.tokens),
            max_new_tokens=max_new_tokens,
            eos=eos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=draws.getrandbits(32),
            logprobs=0 if logprobs else None,
        )
        for prompt in prompts
        for _ in range(group)
    ]


@dataclass
class Scored:
    """A step's completions scored: each one's reward, whether each group's
    rewards were all equal, the completions learned from (`kept`), and their
    examples and advantages."""

    rewards: list[float]
    uniform: list[bool]
    kept: list[int]
    examples: list[Example]
    advantages: list[float]


def score(
    prompts: Sequence[Prompt],
    completions: Sequence[Completion],
    reward: Reward,
    group: int,
    *,
    scale: bool,
    drop_uniform: bool,
) -> Scored:
    """Rewards `group` completions of each prompt and takes each one's
    advantage over its group; with `drop_uniform`, a group whose rewards
    are all equal is not learned from."""
    rewards = [
        float(reward(prompts[index // group], list(completion.tokens)))
        for index, completion in enumerate(completions)
    ]
    advantages = group_advantages(rewards, group, scale=scale)
    groups = [rewards[i : i + group] for i in range(0, len(rewards), group)]
    uniform = [max(g) == min(g) for g in groups]
    kept = [i for i in range(len(completions)) if not (drop_uniform and uniform[i // group])]
    examples = [
        Example.prompted(completions[i].request.prompt, completions[i].tokens) for i in kept
    ]
    return Scored(rewards, uniform, kept, examples, [advantages[i] for i in kept])


def learned_positions(
    items: Sequence[int],
    segments: np.ndarray,
    positions: np.ndarray,
    mask: np.ndarray,
    advantages: Sequence[float],
    count: float,
    sampled_by: Sequence[tuple[int, Sequence[float]]] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """A packed batch's weights (its learned positions over the step's
    count), each position's advantage, and with `sampled_by` (each
    example's prompt length and the engine's log-probabilities of its
    completion) the engine's log-probability of each position's target."""
    segments = np.asarray(segments)
    mask = np.asarray(mask)
    weights = (mask / count).astype(np.float32)
    # Padding is the segment after the last sequence: no advantage.
    per_sequence = np.asarray([advantages[i] for i in items] + [0.0], dtype=np.float32)
    advantage = per_sequence[segments]
    if sampled_by is None:
        return weights, advantage, None
    # Completion token k is the target of the position before it: the
    # prompt's length - 1 + k within its sequence.
    rows = [sampled_by[i] for i in items]
    width = max([len(values) for _, values in rows] + [1])
    table = np.zeros((len(rows) + 1, width), np.float32)
    for s, (_, values) in enumerate(rows):
        table[s, : len(values)] = values
    lengths = np.asarray([length for length, _ in rows] + [1])
    k = np.clip(np.asarray(positions) - (lengths[segments] - 1), 0, width - 1)
    engine = np.where(mask > 0, table[segments, k], 0.0).astype(np.float32)
    return weights, advantage, engine


# ------------------------------------------------------------------- DPO


def pair_examples(pairs: Sequence[Pair]) -> list[Example]:
    """Each pair's chosen answer then its rejected one, after its prompt:
    `pack(..., together=2)` keeps the two in one batch."""
    return [
        Example.prompted(pair.prompt, answer)
        for pair in pairs
        for answer in (pair.chosen, pair.rejected)
    ]


def pair_slots(items: Sequence[int]) -> tuple[list[int], list[int]]:
    """In a batch of `pair_examples` (`items`, their indices among them),
    the sequence of each chosen answer (its rejected one is the next) and
    its pair's index."""
    first = [s for s, item in enumerate(items) if item % 2 == 0]
    return first, [items[s] // 2 for s in first]


__all__ = [
    "DpoStep",
    "GrpoStep",
    "History",
    "Scored",
    "Step",
    "as_prompt",
    "check_grpo",
    "checkpoint_path",
    "complete",
    "learned_positions",
    "pair_examples",
    "pair_slots",
    "prune",
    "sample_requests",
    "score",
    "unsaved",
]
