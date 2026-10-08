"""An output head and its loss, a block of tokens at a time, in JAX.

`std.nn.loss::linear_cross_entropy` and `linear_token_log_probs` take the
hidden states and the output head's weight. Their bodies multiply first,
which holds the `[N, V]` logits: 2.5 GB in f32 for 4096 tokens over a
150K vocabulary, kept again for backward. Here `lax.scan` runs a block of
rows (about `BLOCK_BYTES` of f32 logits) at a time:

- `linear_cross_entropy` returns one number, so its gradient is known in
  the forward pass: each block's gradient goes into the hidden states' and
  the weight's as it is computed, and backward only scales them.
- `linear_token_log_probs` returns one value per token, scaled in backward
  by a gradient known only then: forward keeps each row's log-sum-exp, and
  backward multiplies each block again.

Both accumulate the weight's gradient in f32. The PyTorch backend's
`linnet.torch.loss` computes the same.
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    # What `linear_cross_entropy`'s forward pass keeps for backward: the
    # hidden states' and the weight's gradients, each row's loss, the
    # targets, and empty arrays of the weight's and the weights' dtypes.
    _CrossEntropySaved: TypeAlias = tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, tuple[jax.Array, jax.Array]
    ]
    # What `linear_token_log_probs`'s forward pass keeps: its inputs and each
    # row's log-sum-exp.
    _LogProbsSaved: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]

# The f32 logits of one block of rows.
BLOCK_BYTES = 1 << 30


def _blocks(n: int, vocab: int) -> tuple[int, int]:
    """Rows per block, and blocks."""
    rows = max(1, min(n, BLOCK_BYTES // (4 * vocab)))
    return rows, -(-n // rows)


def _split(value: jax.Array, rows: int, count: int) -> jax.Array:
    """`value` padded to `rows * count` rows, as `[count, rows, ...]`."""
    padding = rows * count - value.shape[0]
    widths = [(0, padding)] + [(0, 0)] * (value.ndim - 1)
    return jnp.pad(value, widths).reshape(count, rows, *value.shape[1:])


def _logits(block: jax.Array, weight: jax.Array) -> jax.Array:
    return (block @ weight.T).astype(jnp.float32)


def _zero_like_integers(value: jax.Array) -> NDArray[np.void]:
    return np.zeros(value.shape, dtype=jax.dtypes.float0)


# ---------------------------------------------------------------- cross entropy


@jax.custom_vjp
def linear_cross_entropy(
    hidden: jax.Array, weight: jax.Array, targets: jax.Array, weights: jax.Array
) -> jax.Array:
    """`sum_n weights[n] * -log softmax(hidden[n] @ weight.T)[targets[n]]`,
    in f32, a block of rows at a time."""
    total, _ = _cross_entropy(hidden, weight, targets, weights, grads=False)
    return total


def _cross_entropy(
    hidden: jax.Array, weight: jax.Array, targets: jax.Array, weights: jax.Array, *, grads: bool
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array] | None]:
    n, width = hidden.shape
    vocab = weight.shape[0]
    rows, count = _blocks(n, vocab)
    blocks = (
        _split(hidden, rows, count),
        _split(targets.astype(jnp.int32), rows, count),
        _split(weights.astype(jnp.float32), rows, count),
    )

    def body(
        carry: tuple[jax.Array, jax.Array], block: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        total, grad_weight = carry
        x, target, weight_of = block
        logits = _logits(x, weight)
        norm = jax.nn.logsumexp(logits, axis=-1)
        picked = jnp.take_along_axis(logits, target[:, None], axis=1)[:, 0]
        losses = norm - picked
        total = total + jnp.sum(weight_of * losses)
        if not grads:
            return (total, grad_weight), (jnp.zeros((0,), hidden.dtype), losses)
        # d/dlogits of w * (logsumexp - logit[t]) = w * (softmax - onehot(t)).
        probs = jnp.exp(logits - norm[:, None])
        probs = probs.at[jnp.arange(rows), target].add(-1.0)
        grad = (probs * weight_of[:, None]).astype(hidden.dtype)
        grad_x = grad @ weight
        grad_weight = grad_weight + (grad.T @ x).astype(jnp.float32)
        return (total, grad_weight), (grad_x, losses)

    start = (
        jnp.zeros((), jnp.float32),
        jnp.zeros((vocab, width), jnp.float32) if grads else jnp.zeros((), jnp.float32),
    )
    (total, grad_weight), (grad_x, losses) = jax.lax.scan(body, start, blocks)
    if not grads:
        return total, None
    grad_hidden = grad_x.reshape(rows * count, width)[:n]
    return total, (grad_hidden, grad_weight, losses.reshape(-1)[:n])


def _cross_entropy_forward(
    hidden: jax.Array, weight: jax.Array, targets: jax.Array, weights: jax.Array
) -> tuple[jax.Array, _CrossEntropySaved]:
    total, found = _cross_entropy(hidden, weight, targets, weights, grads=True)
    assert found is not None
    grad_hidden, grad_weight, losses = found
    # Empty arrays carry the dtypes: residuals are arrays.
    dtypes = (jnp.zeros((0,), weight.dtype), jnp.zeros((0,), weights.dtype))
    return total, (grad_hidden, grad_weight, losses, targets, dtypes)


def _cross_entropy_backward(
    saved: _CrossEntropySaved, grad: jax.Array
) -> tuple[jax.Array, jax.Array, NDArray[np.void], jax.Array]:
    grad_hidden, grad_weight, losses, targets, (weight_like, weights_like) = saved
    scale = grad.astype(jnp.float32)
    return (
        grad_hidden * scale.astype(grad_hidden.dtype),
        (grad_weight * scale).astype(weight_like.dtype),
        _zero_like_integers(targets),
        (losses * scale).astype(weights_like.dtype),
    )


linear_cross_entropy.defvjp(_cross_entropy_forward, _cross_entropy_backward)


# ------------------------------------------------------------ token log-probs


@jax.custom_vjp
def linear_token_log_probs(hidden: jax.Array, weight: jax.Array, targets: jax.Array) -> jax.Array:
    """`log softmax(hidden[n] @ weight.T)[targets[n]]` per row, in f32, a
    block of rows at a time."""
    out, _ = _log_probs(hidden, weight, targets)
    return out


def _log_probs(
    hidden: jax.Array, weight: jax.Array, targets: jax.Array
) -> tuple[jax.Array, jax.Array]:
    n = hidden.shape[0]
    rows, count = _blocks(n, weight.shape[0])
    blocks = (_split(hidden, rows, count), _split(targets.astype(jnp.int32), rows, count))

    def body(
        carry: None, block: tuple[jax.Array, jax.Array]
    ) -> tuple[None, tuple[jax.Array, jax.Array]]:
        x, target = block
        logits = _logits(x, weight)
        norm = jax.nn.logsumexp(logits, axis=-1)
        picked = jnp.take_along_axis(logits, target[:, None], axis=1)[:, 0]
        return carry, (picked - norm, norm)

    _, (out, norms) = jax.lax.scan(body, None, blocks)
    return out.reshape(-1)[:n], norms.reshape(-1)[:n]


def _log_probs_forward(
    hidden: jax.Array, weight: jax.Array, targets: jax.Array
) -> tuple[jax.Array, _LogProbsSaved]:
    out, norms = _log_probs(hidden, weight, targets)
    return out, (hidden, weight, targets, norms)


def _log_probs_backward(
    saved: _LogProbsSaved, grad: jax.Array
) -> tuple[jax.Array, jax.Array, NDArray[np.void]]:
    hidden, weight, targets, norms = saved
    n, width = hidden.shape
    vocab = weight.shape[0]
    rows, count = _blocks(n, vocab)
    blocks = (
        _split(hidden, rows, count),
        _split(targets.astype(jnp.int32), rows, count),
        _split(norms, rows, count),
        _split(grad.astype(jnp.float32), rows, count),
    )

    def body(
        grad_weight: jax.Array, block: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        x, target, norm, scale = block
        # d/dlogits of logit[t] - logsumexp = onehot(t) - softmax.
        probs = -jnp.exp(_logits(x, weight) - norm[:, None])
        probs = probs.at[jnp.arange(rows), target].add(1.0)
        step = (probs * scale[:, None]).astype(hidden.dtype)
        grad_weight = grad_weight + (step.T @ x).astype(jnp.float32)
        return grad_weight, step @ weight

    grad_weight, grad_x = jax.lax.scan(body, jnp.zeros((vocab, width), jnp.float32), blocks)
    return (
        grad_x.reshape(rows * count, width)[:n],
        grad_weight.astype(weight.dtype),
        _zero_like_integers(targets),
    )


linear_token_log_probs.defvjp(_log_probs_forward, _log_probs_backward)


__all__ = ["BLOCK_BYTES", "linear_cross_entropy", "linear_token_log_probs"]
