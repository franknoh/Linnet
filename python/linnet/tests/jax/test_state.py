"""Entries with `state` members run in JAX functionally: state in, state out,
by parameter path, agreeing with the Torch materializer's stateful view."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from linnet.jax import load

from .test_round_trip import REPO, STDLIB

SOURCE = REPO / "spec-tests/valid/020_state.linnet"


def test_state_is_threaded_through_calls() -> None:
    step = load(SOURCE, generics={"B": 1, "Max": 4, "D": 3}, weights={}, std_root=STDLIB)
    key0 = jnp.arange(3, dtype=jnp.float32).reshape(1, 1, 3)
    key1 = key0 + 10

    out0, state = step(key0, jnp.int32(0))  # every state starts at zeros
    assert set(state) == {"cache.keys", "caches.0.keys", "caches.1.keys"}
    out1, state = step(key1, jnp.int32(2), state=state)

    # Three caches (`cache` and the two `caches`) each hold both positions,
    # and the entry adds `cache.keys` once more.
    expected = np.zeros((1, 4, 3), np.float32)
    expected[0, 0] = np.asarray(key0[0, 0]) * 4
    expected[0, 2] = np.asarray(key1[0, 0]) * 4
    np.testing.assert_allclose(np.asarray(out1), expected)
    np.testing.assert_allclose(np.asarray(out0)[0, 2], np.zeros(3))
    np.testing.assert_allclose(np.asarray(state["caches.1.keys"])[0, 2], np.asarray(key1[0, 0]))

    # Restarting from fresh state reproduces the first call.
    again, _ = step(key0, jnp.int32(0))
    np.testing.assert_allclose(np.asarray(again), np.asarray(out0))
