"""How the blockwise output head and loss (`linnet.torch.loss`,
`linnet.jax.loss`) split their rows, shared with the estimates in
`linnet.resources`, which import neither torch nor JAX."""

from __future__ import annotations

# The f32 logits of one block of rows.
BLOCK_BYTES = 1 << 30

# A vocabulary split across processes: each runs the loss over its part.
SPLIT_LOSSES = {
    "linnet.split_cross_entropy": "linnet.linear_cross_entropy",
    "linnet.split_token_log_probs": "linnet.linear_token_log_probs",
}


def block_rows(vocab: int, block_bytes: int = BLOCK_BYTES) -> int:
    """Rows per block: `block_bytes` of f32 logits, and at least one row."""
    return max(1, block_bytes // (4 * vocab))


__all__ = ["BLOCK_BYTES", "SPLIT_LOSSES", "block_rows"]
