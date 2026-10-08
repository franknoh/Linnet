"""Drawing tokens (`linnet.serve.sampling`): draws follow the softmax of the
scaled, filtered logits, greedy rows keep their argmax, and PyTorch, JAX,
and NumPy draw the same tokens from the same logits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pytest
import torch
from numpy.typing import ArrayLike

from linnet.serve.sampling import (
    FILTERED,
    GREEDY,
    SAMPLED,
    Sampling,
    draw_numpy,
    draw_torch,
    mode,
)

if TYPE_CHECKING:
    import jax
    from jax.typing import DTypeLike

    class _JaxNumPy(Protocol):
        """`jax.numpy`, for `asarray`: JAX's stubs leave one of its
        parameters (`device`) partially unknown."""

        def asarray(self, a: ArrayLike, /, dtype: DTypeLike | None = ...) -> jax.Array: ...


PROBABILITIES = np.array([0.5, 0.25, 0.15, 0.1], dtype=np.float32)


def _torch(logits: np.ndarray, rows: list[Sampling], at: np.ndarray) -> np.ndarray:
    drawn = draw_torch(
        torch.as_tensor(logits),
        torch.tensor([r.temperature for r in rows], dtype=torch.float32),
        torch.tensor([r.top_k for r in rows], dtype=torch.int64),
        torch.tensor([r.top_p for r in rows], dtype=torch.float32),
        torch.tensor([r.key for r in rows], dtype=torch.int64),
        torch.as_tensor(at.astype(np.int32)),
        mode(rows),
    )
    return drawn.numpy()


def _numpy(logits: np.ndarray, rows: list[Sampling], at: np.ndarray) -> np.ndarray:
    return np.asarray(draw_numpy(logits, rows, at, mode(rows)))


def _jax(logits: np.ndarray, rows: list[Sampling], at: np.ndarray) -> np.ndarray:
    import jax.numpy as jnp

    from linnet.serve.sampling import draw_jax

    typed = cast("_JaxNumPy", jnp)
    drawn = draw_jax(
        typed.asarray(logits),
        typed.asarray([r.temperature for r in rows], dtype=jnp.float32),
        typed.asarray([r.top_k for r in rows], dtype=jnp.int32),
        typed.asarray([r.top_p for r in rows], dtype=jnp.float32),
        typed.asarray(np.array([r.key for r in rows], dtype=np.uint32)),
        typed.asarray(at.astype(np.int32)),
        mode(rows),
    )
    return np.asarray(drawn)


@pytest.mark.parametrize("draw", [_torch, _numpy])
@pytest.mark.parametrize(
    ("sampling", "expected"),
    [
        (Sampling(temperature=1.0), PROBABILITIES),
        (Sampling(temperature=0.5), PROBABILITIES**2 / (PROBABILITIES**2).sum()),
        (Sampling(temperature=1.0, top_k=2), [2 / 3, 1 / 3, 0, 0]),
        # The fewest tokens reaching 0.7 are the first two; 0.8 takes three.
        (Sampling(temperature=1.0, top_p=0.7), [2 / 3, 1 / 3, 0, 0]),
        (Sampling(temperature=1.0, top_p=0.8), [0.5 / 0.9, 0.25 / 0.9, 0.15 / 0.9, 0]),
        (Sampling(temperature=1.0, top_k=3, top_p=0.5), [1, 0, 0, 0]),
    ],
)
def test_draws_follow_the_softmax(
    draw: Callable[[np.ndarray, list[Sampling], np.ndarray], np.ndarray],
    sampling: Sampling,
    expected: np.ndarray | Sequence[float],
) -> None:
    count = 20000
    rows = [
        Sampling(sampling.temperature, sampling.top_k, sampling.top_p, seed)
        for seed in range(count)
    ]
    logits = np.ones((count, 1), dtype=np.float32) * np.log(PROBABILITIES)
    drawn = draw(logits, rows, np.zeros(count, dtype=np.int64))
    frequencies = np.bincount(drawn, minlength=4) / count
    np.testing.assert_allclose(frequencies, expected, atol=0.015)


def test_positions_draw_afresh() -> None:
    """One seed over many positions: the draws are as varied as over many
    seeds at one position."""
    count = 20000
    rows = [Sampling(temperature=1.0, seed=7)] * count
    logits = np.ones((count, 1), dtype=np.float32) * np.log(PROBABILITIES)
    drawn = _numpy(logits, rows, np.arange(count))
    np.testing.assert_allclose(np.bincount(drawn, minlength=4) / count, PROBABILITIES, atol=0.015)


def _mixed(count: int, vocab: int) -> tuple[np.ndarray, list[Sampling], np.ndarray]:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(count, vocab)).astype(np.float32) * 3
    kinds = [
        Sampling(),
        Sampling(temperature=0.7),
        Sampling(temperature=1.3, top_k=40),
        Sampling(temperature=0.9, top_p=0.9),
        Sampling(temperature=1.0, top_k=5, top_p=0.5),
    ]
    rows = [
        Sampling(k.temperature, k.top_k, k.top_p, int(rng.integers(1 << 40)))
        for k in (kinds[i % len(kinds)] for i in range(count))
    ]
    return logits, rows, rng.integers(0, 4096, size=count)


def test_backends_draw_alike() -> None:
    logits, rows, at = _mixed(200, 1000)
    np.testing.assert_array_equal(_torch(logits, rows, at), _numpy(logits, rows, at))


def test_jax_draws_alike() -> None:
    pytest.importorskip("jax")
    logits, rows, at = _mixed(200, 1000)
    np.testing.assert_array_equal(_jax(logits, rows, at), _numpy(logits, rows, at))


def test_greedy_rows_keep_their_argmax() -> None:
    logits, rows, at = _mixed(200, 1000)
    greedy = [i for i, row in enumerate(rows) if row.temperature == 0]
    for draw in (_torch, _numpy):
        np.testing.assert_array_equal(draw(logits, rows, at)[greedy], logits[greedy].argmax(-1))


def test_mode() -> None:
    assert mode([]) == GREEDY
    assert mode([Sampling(), Sampling(temperature=0.5)]) == SAMPLED
    assert (
        mode([Sampling(top_k=3), Sampling(temperature=0.5)]) == SAMPLED
    )  # greedy rows filter nothing
    assert mode([Sampling(temperature=0.5, top_p=0.9)]) == FILTERED


@pytest.mark.parametrize(
    "sampling",
    [Sampling(temperature=-1.0), Sampling(top_k=-1), Sampling(top_p=0.0), Sampling(top_p=1.5)],
)
def test_out_of_range(sampling: Sampling) -> None:
    with pytest.raises(ValueError):
        sampling.check()
