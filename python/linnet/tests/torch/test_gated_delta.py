"""`std.nn.recurrent`: the gated delta rule (Gated DeltaNet) and the causal
convolution beside it, against the recurrence written out token by token in
f64. The canonical body, the chunked PyTorch kernel (`linnet.torch.delta`),
and on CUDA `fla`'s where it is installed."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from linnet.torch import load_function
from linnet.torch.delta import gated_delta_rule
from tests.helpers import STDLIB, requires_cuda

SOURCE = """\
module tests.delta

use std.nn.recurrent::{gated_delta_rule, gated_delta_step, causal_conv}

pub entry rule<B: Dim, S: Dim, Hk: Dim, Hv: Dim, K: Dim, V: Dim, T: Float>(
    query: Tensor[B, S, Hk, K; T],
    key: Tensor[B, S, Hk, K; T],
    value: Tensor[B, S, Hv, V; T],
    decay: Tensor[B, S, Hv; f32],
    beta: Tensor[B, S, Hv; f32],
    initial: Tensor[B, Hv, K, V; f32],
) -> Tensor[B, S + K, Hv, V; f32]
where
    Hk > 0,
    Hv % Hk == 0
{
    return gated_delta_rule(query, key, value, decay, beta, initial, 0.25)
}

pub entry step<B: Dim, Hk: Dim, Hv: Dim, K: Dim, V: Dim>(
    query: Tensor[B, Hk, K; f32],
    key: Tensor[B, Hk, K; f32],
    value: Tensor[B, Hv, V; f32],
    decay: Tensor[B, Hv; f32],
    beta: Tensor[B, Hv; f32],
    initial: Tensor[B, Hv, K, V; f32],
) -> Tensor[B, 1 + K, Hv, V; f32]
where
    Hk > 0,
    Hv % Hk == 0
{
    return gated_delta_step(query, key, value, decay, beta, initial, 0.25)
}

pub entry conv<B: Dim, S: Dim, C: Dim>(
    x: Tensor[B, S, C; f32],
    history: Tensor[B, 3, C; f32],
    weight: Tensor[C, 4; f32],
) -> Tensor[B, S, C; f32] {
    return causal_conv(x, history, weight)
}
"""


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Token by token in f64: `[B, S + K, Hv, V]`, the outputs then the state."""
    heads = value.shape[2]
    group = heads // query.shape[2]
    q = query.double().repeat_interleave(group, dim=2) * 0.25
    k = key.double().repeat_interleave(group, dim=2)
    v, g, b = value.double(), decay.double(), beta.double()
    state = initial.double().clone()
    outputs: list[torch.Tensor] = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        predicted = torch.einsum("bhkv,bhk->bhv", state, k[:, t])
        correction = (v[:, t] - predicted) * b[:, t][..., None]
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], correction)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, t]))
    return torch.cat([torch.stack(outputs, dim=1), state.permute(0, 2, 1, 3)], dim=1)


def _inputs(
    device: str, length: int, dtype: torch.dtype = torch.float32, batch: int = 2
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(0)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=device)

    # Keys and queries of unit length, as the models normalize them, so the
    # state stays bounded however long the sequence.
    query = torch.nn.functional.normalize(normal(batch, length, 2, 16), dim=-1).to(dtype)
    key = torch.nn.functional.normalize(normal(batch, length, 2, 16), dim=-1).to(dtype)
    value = normal(batch, length, 4, 8).to(dtype)
    decay = -torch.nn.functional.softplus(normal(batch, length, 4))
    beta = torch.sigmoid(normal(batch, length, 4))
    initial = normal(batch, 4, 16, 8) * 0.1
    return query, key, value, decay, beta, initial


@pytest.mark.parametrize(("numerics", "length"), [("exact", 5), ("fast", 5), ("fast", 150)])
def test_the_rule_matches_the_recurrence(tmp_path: Path, numerics: str, length: int) -> None:
    """The body and the chunked kernel (blocks of 64, here a partial last
    one), from a state already holding something."""
    source = tmp_path / "delta.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    rule = load_function(source, "rule", std_root=STDLIB, numerics=numerics)
    inputs = _inputs("cpu", length)
    torch.testing.assert_close(rule(*inputs).double(), _reference(*inputs), atol=1e-4, rtol=1e-4)


def test_padding_leaves_the_state(tmp_path: Path) -> None:
    """Tokens with no decay and no write change nothing: a padded prompt's
    final state is the unpadded one's."""
    inputs = list(_inputs("cpu", 9))
    inputs[3][:, 6:] = 0.0
    inputs[4][:, 6:] = 0.0
    full = gated_delta_rule(*inputs, 0.25)
    short = gated_delta_rule(*(x[:, :6] if i < 5 else x for i, x in enumerate(inputs)), 0.25)
    torch.testing.assert_close(full[:, 9:], short[:, 6:], atol=1e-5, rtol=1e-5)


def test_a_step_is_one_token_of_the_rule(tmp_path: Path) -> None:
    source = tmp_path / "delta.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    step = load_function(source, "step", std_root=STDLIB)
    inputs = _inputs("cpu", 1)
    got = step(*(x[:, 0] if i < 5 else x for i, x in enumerate(inputs)))
    torch.testing.assert_close(got.double(), _reference(*inputs), atol=1e-5, rtol=1e-5)


def test_the_convolution_reads_its_history(tmp_path: Path) -> None:
    source = tmp_path / "delta.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    conv = load_function(source, "conv", std_root=STDLIB)
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(2, 7, 5, generator=generator)
    history = torch.randn(2, 3, 5, generator=generator)
    weight = torch.randn(5, 4, generator=generator)
    expected = torch.nn.functional.conv1d(
        torch.cat([history, x], dim=1).transpose(1, 2), weight[:, None], groups=5
    ).transpose(1, 2)
    torch.testing.assert_close(conv(x, history, weight), expected, atol=1e-5, rtol=1e-5)


@requires_cuda
@pytest.mark.parametrize("length", [1, 64, 300])
def test_the_kernel_on_cuda(tmp_path: Path, length: int) -> None:
    """bf16 on CUDA under `torch.compile`: `fla`'s kernel where it is
    installed, the chunked PyTorch otherwise."""
    source = tmp_path / "delta.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    rule = load_function(source, "rule", std_root=STDLIB, numerics="fast", compile="inductor")
    inputs = _inputs("cuda", length, torch.bfloat16)
    expected = _reference(*inputs)
    torch.testing.assert_close(rule(*inputs).double(), expected, atol=3e-2, rtol=3e-2)
