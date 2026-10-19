"""`std.nn.recurrent::gated_delta_rule` for generated PyTorch: the gated delta
rule (Gated DeltaNet) over whole sequences, in blocks of tokens rather than
one token at a time.

With `fla` (flash-linear-attention) installed and the inputs on CUDA, its
chunked Triton kernel runs; otherwise the same chunked algorithm in PyTorch:
within a block of 64 tokens the updates are condensed by a triangular solve
(the WY/UT transform), and the state is carried from block to block. Both
are wrapped as the custom op `linnet::gated_delta_rule`, which `torch.compile`
calls as it is and a CUDA graph captures.
"""

# PyTorch's custom-op registry and `fla` carry no complete types.
# pyright: reportMissingImports=false, reportUnknownVariableType=false

from __future__ import annotations

import importlib.util
from typing import Protocol, cast

import torch
import torch.nn.functional as functional

_CHUNK = 64


class _SolveTriangular(Protocol):
    """`torch.linalg.solve_triangular`, which PyTorch's stubs leave untyped."""

    def __call__(self, a: torch.Tensor, b: torch.Tensor, *, upper: bool) -> torch.Tensor: ...


_solve: _SolveTriangular = getattr(torch.linalg, "solve_triangular")  # noqa: B009 - untyped
_fla: list[bool] = []


def _has_fla() -> bool:
    if not _fla:
        _fla.append(importlib.util.find_spec("fla") is not None)
    return _fla[0]


def _repeat(x: torch.Tensor, heads: int) -> torch.Tensor:
    """`[B, S, Hk, K]` with each head repeated for the value heads it serves."""
    group = heads // x.shape[2]
    return x if group == 1 else x.repeat_interleave(group, dim=2)


def _chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The chunked algorithm in f32: `query`/`key` `[B, S, H, K]`, `value`
    `[B, S, H, V]`, `decay`/`beta` `[B, S, H]`, `initial` `[B, H, K, V]`.
    Returns the outputs `[B, S, H, V]` and the final state."""
    batch, length, heads, _ = key.shape
    q, k, v, b, g = (
        x.transpose(1, 2).float().contiguous() for x in (query, key, value, beta, decay)
    )
    q = q * scale
    pad = (_CHUNK - length % _CHUNK) % _CHUNK
    q, k, v = (functional.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    b, g = (functional.pad(x, (0, pad)) for x in (b, g))
    chunks = (length + pad) // _CHUNK
    v_beta = v * b[..., None]
    k_beta = k * b[..., None]
    q, k, k_beta, v_beta = (
        x.reshape(batch, heads, chunks, _CHUNK, x.shape[-1]) for x in (q, k, k_beta, v_beta)
    )
    g = g.reshape(batch, heads, chunks, _CHUNK).cumsum(-1)
    upper = torch.ones(_CHUNK, _CHUNK, dtype=torch.bool, device=q.device).triu(1)
    pairwise = (g[..., :, None] - g[..., None, :]).masked_fill(upper, float("-inf")).exp()
    system = (k_beta @ k.transpose(-1, -2)) * pairwise
    within = (q @ k.transpose(-1, -2)) * pairwise
    lower = system.masked_fill(upper | torch.eye(_CHUNK, dtype=torch.bool, device=q.device), 0)
    identity = torch.eye(_CHUNK, dtype=q.dtype, device=q.device)
    written = _solve(identity + lower, v_beta, upper=False)
    reads = _solve(identity + lower, k_beta * g.exp()[..., None], upper=False)
    q = q * g.exp()[..., None]
    k = k * (g[..., -1:] - g).exp()[..., None]
    block_decay = g[..., -1].exp()[..., None, None]
    state = initial.float()
    out = torch.empty_like(written)
    for i in range(chunks):
        fresh = written[:, :, i] - reads[:, :, i] @ state
        out[:, :, i] = q[:, :, i] @ state + within[:, :, i] @ fresh
        state = state * block_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ fresh
    out = out.reshape(batch, heads, chunks * _CHUNK, -1)[:, :, :length]
    return out.transpose(1, 2), state


@torch.library.custom_op("linnet::gated_delta_rule", mutates_args=())
def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """`std.nn.recurrent::gated_delta_rule`: `[B, S + K, Hv, V]` f32, each
    token's output and then the final state (`[b, S + k, h, v]` is
    `state[b, h, k, v]`)."""
    heads = value.shape[2]
    q, k = _repeat(query, heads), _repeat(key, heads)
    if _has_fla() and query.is_cuda and query.dtype in (torch.bfloat16, torch.float16):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        out, state = cast(
            "tuple[torch.Tensor, torch.Tensor]",
            chunk_gated_delta_rule(
                q,
                k,
                value,
                g=decay.float(),
                beta=beta.to(value.dtype),
                scale=scale,
                initial_state=initial.float().contiguous(),
                output_final_state=True,
            ),
        )
    else:
        out, state = _chunked(q, k, value, decay, beta, initial, scale)
    return torch.cat([out.float(), state.permute(0, 2, 1, 3)], dim=1)


@gated_delta_rule.register_fake  # pyright: ignore[reportUnknownMemberType]
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch, length, heads, width = value.shape
    return value.new_empty(batch, length + key.shape[3], heads, width, dtype=torch.float32)


def _update_math(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """One token's update in PyTorch: the new state, `[B, Hv, K, V]` f32."""
    heads = value.shape[1]
    k = key.float().repeat_interleave(heads // key.shape[1], dim=1)
    kept = state * decay.float().exp()[..., None, None]
    predicted = torch.einsum("bhkv,bhk->bhv", kept, k)
    correction = (value.float() - predicted) * beta.float()[..., None]
    return kept + k[..., None] * correction[..., None, :]


def _kernel_runs(state: torch.Tensor) -> bool:
    width, columns = state.shape[2], state.shape[3]
    return (
        state.is_cuda
        and state.dtype == torch.float32
        and state.is_contiguous()
        and width & (width - 1) == 0
        and columns & (columns - 1) == 0
    )


def _operands(
    key: torch.Tensor, value: torch.Tensor, decay: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    return (
        key.contiguous(),
        value.contiguous(),
        decay.float().contiguous(),
        beta.float().contiguous(),
    )


def gated_delta_update(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """`std.nn.recurrent::gated_delta_update` into a new state: one pass
    over the old state on CUDA (`kernels.delta_update`)."""
    if _kernel_runs(state):
        from .kernels import delta_update

        return delta_update(state, *_operands(key, value, decay, beta))
    return _update_math(state, key, value, decay, beta)


def gated_delta_update_(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """`gated_delta_update` written into `state` itself, which it returns:
    where the old state has no other reader (the generated code knows), a
    decoding step reads and writes it once rather than copying it."""
    if _kernel_runs(state):
        from .kernels import delta_update_

        delta_update_(state, *_operands(key, value, decay, beta))
    else:
        state.copy_(_update_math(state, key, value, decay, beta))
    return state


def causal_conv(x: torch.Tensor, history: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`std.nn.recurrent::causal_conv`: `x` `[B, S, C]` after `history`
    `[B, W - 1, C]`, each channel convolved with its row of `weight` `[C, W]`,
    as one grouped convolution; `[B, S, C]` in f32."""
    signal = torch.cat([history, x], dim=1).transpose(1, 2)
    out = functional.conv1d(signal, weight[:, None, :], groups=weight.shape[0])
    return out.transpose(1, 2).float()


__all__ = ["causal_conv", "gated_delta_rule", "gated_delta_update", "gated_delta_update_"]
