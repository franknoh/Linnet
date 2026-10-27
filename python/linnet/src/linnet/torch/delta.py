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
        and state.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and state.is_contiguous()
        and width & (width - 1) == 0
        and columns & (columns - 1) == 0
    )


def _operands(
    key: torch.Tensor, value: torch.Tensor, decay: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _decode_math(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """`gated_delta_decode` in PyTorch: `[B, 1 + K, Hv, V]` in `state`'s
    dtype, the step in f32."""
    heads = value.shape[1]
    updated = _update_math(state[:, 1:].permute(0, 2, 1, 3).float(), key, value, decay, beta)
    q = query.float().repeat_interleave(heads // query.shape[1], dim=1) * scale
    read = torch.einsum("bhkv,bhk->bhv", updated, q)
    return torch.cat([read[:, None], updated.permute(0, 2, 1, 3)], dim=1).to(state.dtype)


def _decode_runs(state: torch.Tensor) -> bool:
    width, columns = state.shape[1] - 1, state.shape[3]
    return (
        state.is_cuda
        and state.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and state.is_contiguous()
        and width & (width - 1) == 0
        and columns & (columns - 1) == 0
    )


def gated_delta_decode(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """`std.nn.recurrent::gated_delta_decode` into a new state: one pass over
    the old state on CUDA (`kernels.delta_decode`), the output in row 0."""
    if _decode_runs(state):
        from .kernels import delta_decode

        return delta_decode(state, query.contiguous(), *_operands(key, value, decay, beta), scale)
    return _decode_math(state, query, key, value, decay, beta, scale)


def gated_delta_decode_(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """`gated_delta_decode` written into `state` itself, which it returns:
    a decoding step reads the state once, writes it once, and reads its
    output from row 0."""
    if _decode_runs(state):
        from .kernels import delta_decode_

        delta_decode_(state, query.contiguous(), *_operands(key, value, decay, beta), scale)
    else:
        state.copy_(_decode_math(state, query, key, value, decay, beta, scale))
    return state


def causal_conv(x: torch.Tensor, history: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`std.nn.recurrent::causal_conv`: `x` `[B, S, C]` after `history`
    `[B, W - 1, C]`, each channel convolved with its row of `weight` `[C, W]`,
    as one grouped convolution; `[B, S, C]` in f32."""
    signal = torch.cat([history, x], dim=1).transpose(1, 2)
    out = functional.conv1d(signal, weight[:, None, :], groups=weight.shape[0])
    return out.transpose(1, 2).float()


def causal_conv_packed(
    x: torch.Tensor, segments: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """`std.nn.recurrent::causal_conv_packed`: `x` `[P, C]` of prompts packed
    together, each tap reading only its own prompt's inputs; `[P, C]` in f32."""
    width = weight.shape[1]
    taps = weight.float()
    out = x.float() * taps[:, width - 1]
    for back in range(1, min(width, x.shape[0])):
        same = (segments[back:] == segments[:-back])[:, None]
        out[back:] += x[:-back].float() * taps[:, width - 1 - back] * same
    return out


def causal_history_packed(
    x: torch.Tensor, segments: torch.Tensor, rows: torch.Tensor, history: torch.Tensor
) -> torch.Tensor:
    """`std.nn.recurrent::causal_history_packed`: each packed prompt's last
    inputs into its row of `history` `[R, L, C]`, found on the device."""
    total, count, width = x.shape[0], history.shape[0], history.shape[1]
    after = torch.cat([segments[1:], segments.new_full((1,), -1)])
    ends = (segments >= 0) & (after != segments)
    # Each row's prompt's last token; the tokens that end no prompt aim past
    # the rows, at an entry dropped after.
    target = torch.where(ends, rows.long(), count)
    steps = torch.arange(total, device=x.device)
    last = torch.full((count + 1,), -1, dtype=torch.long, device=x.device)
    last = last.scatter_reduce(0, target, steps, "amax")[:count]
    back = last[:, None] + torch.arange(width, device=x.device) - (width - 1)
    at = back.clamp(min=0)
    same = (back >= 0) & (segments[at] == segments[last.clamp(min=0)][:, None])
    window = x[at] * same[..., None]
    return torch.where((last >= 0)[:, None, None], window.to(history.dtype), history)


def _owners(
    segments: torch.Tensor, rows: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each prompt's length (`[count]`, by segment) and each row's prompt
    (`[count]`, `-1` for the rows no prompt fills), without a host sync."""
    live = segments >= 0
    index = torch.where(live, segments.long(), count)
    lengths = torch.zeros(count + 1, dtype=torch.long, device=segments.device)
    lengths = lengths.scatter_add(0, index, live.long())[:count]
    owner = torch.full((count + 1,), -1, dtype=torch.long, device=segments.device)
    owner = owner.scatter_reduce(0, torch.where(live, rows.long(), count), index, "amax")
    return lengths, owner[:count]


def _fla_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`fla`'s chunked kernel over the prompts as sequences of their lengths:
    the outputs `[1, P, Hv, V]` and each prompt's final state. Its sequence
    and block tables are made on the device at sizes fixed by the shapes
    (unused prompts empty, the blocks past the last aimed at an empty one), so
    a CUDA graph replays the pass whatever the prompts' lengths."""
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

    total, heads, width = value.shape[1], value.shape[2], value.shape[3]
    count = lengths.shape[0]
    ends = functional.pad(lengths.cumsum(0), (1, 0))
    # One more, empty, sequence: where the unused blocks point.
    bounds = torch.cat([ends, ends[-1:]])
    blocks = (lengths + _CHUNK - 1) // _CHUNK
    firsts = functional.pad(blocks.cumsum(0), (1, 0))
    slots = -(-total // _CHUNK) + min(count, total)
    step = torch.arange(slots, device=value.device)
    sequence = torch.searchsorted(firsts[1:], step, right=True)
    within = torch.where(sequence < count, step - firsts[sequence.clamp(max=count)], 0)
    initial = value.new_zeros(count + 1, heads, k.shape[3], width, dtype=torch.float32)
    # The kernels index their inputs as laid out contiguously, which the
    # public entry point (not this one) makes sure of.
    _, out, _, final, *_ = chunk_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=value.contiguous(),
        g=decay.float().contiguous(),
        beta=beta.to(value.dtype).contiguous(),
        scale=scale,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=bounds,
        chunk_indices=torch.stack([sequence, within], 1),
    )
    return cast("torch.Tensor", out), cast("torch.Tensor", final)[:count]


def _host(lengths: torch.Tensor) -> list[int]:
    return [int(n) for n in lengths.cpu()]


def _chunked_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`_fla_packed` with the PyTorch chunked algorithm, prompt by prompt."""
    heads, keys, width = value.shape[2], k.shape[3], value.shape[3]
    out = value.new_zeros(value.shape, dtype=torch.float32)
    finals = value.new_zeros(lengths.shape[0], heads, keys, width, dtype=torch.float32)
    start = 0
    for n, length in enumerate(_host(lengths)):
        if length:
            end = start + length
            zero = value.new_zeros(1, heads, keys, width, dtype=torch.float32)
            o, state = _chunked(
                q[:, start:end],
                k[:, start:end],
                value[:, start:end],
                decay[:, start:end],
                beta[:, start:end],
                zero,
                scale,
            )
            out[:, start:end] = o
            finals[n] = state[0]
            start = end
    return out, finals


@torch.library.custom_op("linnet::gated_delta_rule_packed", mutates_args=())
def gated_delta_rule_packed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    segments: torch.Tensor,
    rows: torch.Tensor,
    current: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """`std.nn.recurrent::gated_delta_rule_packed`: `[P * Hv + R * Hv * K,
    V]` f32, each token's output then the rows' states. The prompts are
    numbered from 0 (`segments`, as `linnet.serve` packs them), fewer than
    the rows."""
    total, heads, width = value.shape[1], value.shape[2], value.shape[3]
    q, k = _repeat(query, heads), _repeat(key, heads)
    lengths, owner = _owners(segments, rows, current.shape[0])
    if _has_fla() and query.is_cuda and query.dtype in (torch.bfloat16, torch.float16):
        out, finals = _fla_packed(q, k, value, decay, beta, lengths, scale)
    else:
        out, finals = _chunked_packed(q, k, value, decay, beta, lengths, scale)
    # Selected, not multiplied: `fla` leaves the padding's outputs unwritten,
    # and what memory held there may not be a number.
    out = torch.where((segments >= 0)[:, None, None], out[0].float(), 0.0)
    state = torch.where(
        (owner >= 0)[:, None, None, None], finals[owner.clamp(min=0)], current.float()
    )
    return torch.cat([out.reshape(total * heads, width), state.reshape(-1, width)])


@gated_delta_rule_packed.register_fake  # pyright: ignore[reportUnknownMemberType]
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    segments: torch.Tensor,
    rows: torch.Tensor,
    current: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    total, heads, width = value.shape[1], value.shape[2], value.shape[3]
    states = current.shape[0] * heads * key.shape[3]
    return value.new_empty(total * heads + states, width, dtype=torch.float32)


__all__ = [
    "causal_conv",
    "causal_conv_packed",
    "causal_history_packed",
    "gated_delta_decode",
    "gated_delta_decode_",
    "gated_delta_rule",
    "gated_delta_rule_packed",
    "gated_delta_update",
    "gated_delta_update_",
]
