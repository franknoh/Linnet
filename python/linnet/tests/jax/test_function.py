"""Functions: module-level entries (`examples/09-functions`) as JAX
functions. They agree with references written in `jax.numpy`, bind their
generics from each call's inputs, and compose with `jax.grad`, `jax.jit`,
and `jax.vmap`."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linnet.jax import LinnetError, load_function

from .test_round_trip import REPO, STDLIB

EXAMPLE = REPO / "examples" / "09-functions" / "functions.linnet"


def _cross_entropy(logits: Any, labels: Any) -> Any:
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=-1)[:, 0])


def test_cross_entropy_and_its_gradient() -> None:
    cross_entropy = load_function(EXAMPLE, "cross_entropy", std_root=STDLIB)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    logits = jax.random.normal(keys[0], (6, 11), dtype=jnp.float32)
    labels = jax.random.randint(keys[1], (6,), 0, 11).astype(jnp.int64)
    loss = cross_entropy(logits, labels)
    assert loss.shape == () and loss.dtype == jnp.float32
    np.testing.assert_allclose(loss, _cross_entropy(logits, labels), rtol=1e-5, atol=1e-6)
    grad = jax.grad(cross_entropy)(logits, labels)
    np.testing.assert_allclose(grad, jax.grad(_cross_entropy)(logits, labels), rtol=1e-5, atol=1e-6)
    # bf16 logits bind `T` to bf16; the loss is still computed in f32.
    half = logits.astype(jnp.bfloat16)
    np.testing.assert_allclose(
        cross_entropy(half, labels), _cross_entropy(half, labels), rtol=1e-5, atol=1e-6
    )


def test_functions_compose_with_jit_and_vmap() -> None:
    log_probs = load_function(EXAMPLE, "token_log_probs", std_root=STDLIB)
    keys = jax.random.split(jax.random.PRNGKey(1), 2)
    logits = jax.random.normal(keys[0], (4, 2, 5, 7), dtype=jnp.float32)
    tokens = jax.random.randint(keys[1], (4, 2, 5), 0, 7).astype(jnp.int64)
    expected = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), tokens[..., None], -1)[
        ..., 0
    ]
    # Under `vmap` the function sees one element of the leading axis at a
    # time: its generics bind to [2, 5, 7], not [4, 2, 5, 7].
    batched = jax.jit(jax.vmap(log_probs))(logits, tokens)
    np.testing.assert_allclose(batched, expected, rtol=1e-5, atol=1e-6)

    def objective(values: Any) -> Any:
        return -jnp.sum(log_probs(values, tokens[0]))

    def reference(values: Any) -> Any:
        chosen = jnp.take_along_axis(jax.nn.log_softmax(values, -1), tokens[0][..., None], -1)
        return -jnp.sum(chosen)

    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(logits[0]),
        jax.grad(reference)(logits[0]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_preprocessing_and_a_reward_with_a_scalar_input() -> None:
    normalize = load_function(EXAMPLE, "normalize_images", std_root=STDLIB)
    pixels = np.random.default_rng(0).integers(0, 256, (2, 5, 4, 3), dtype=np.uint8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    expected = ((pixels.astype(np.float32) / 255.0 - mean) / std).transpose(0, 3, 1, 2)
    np.testing.assert_allclose(normalize(pixels, mean, std), expected, rtol=1e-6, atol=1e-6)

    reward = load_function(EXAMPLE, "match_reward", std_root=STDLIB)
    generated = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    reference = np.array([[1, 2, 0, 4], [5, 0, 0, 0]], dtype=np.int64)
    lengths = np.array([4, 2], dtype=np.int64)
    np.testing.assert_allclose(
        reward(generated, reference, lengths, 0.5), [(3 - 0.5) / 4, (1 - 0.5) / 2], rtol=1e-6
    )


def test_mistakes_are_reported() -> None:
    with pytest.raises(LinnetError, match="4 module-level entries"):
        load_function(EXAMPLE, std_root=STDLIB)
    cross_entropy = load_function(EXAMPLE, "cross_entropy", std_root=STDLIB)
    with pytest.raises(LinnetError, match="`T` of `cross_entropy` is float, not i64"):
        cross_entropy(np.zeros((2, 3), dtype=np.int64), np.zeros(2, dtype=np.int64))
    with pytest.raises(LinnetError, match="has size 4 where `B` is 2"):
        cross_entropy(np.zeros((2, 3), dtype=np.float32), np.zeros(4, dtype=np.int64))
