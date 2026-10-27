"""`std.nn.recurrent`: the gated delta rule (Gated DeltaNet) and the causal
convolution beside it, against the recurrence written out token by token in
f64. The canonical body, the chunked PyTorch kernel (`linnet.torch.delta`),
and on CUDA `fla`'s where it is installed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from linnet.compiler import find_compiler
from linnet.torch import load_function
from linnet.torch.delta import (
    gated_delta_decode,
    gated_delta_decode_,
    gated_delta_rule,
    gated_delta_rule_packed,
    gated_delta_update,
    gated_delta_update_,
)
from tests.helpers import STDLIB, requires_cuda

SOURCE = """\
module tests.delta

use std.nn.recurrent::{gated_delta_rule, gated_delta_step, gated_delta_update, causal_conv}
use std.nn.recurrent::{gated_delta_rule_packed, causal_conv_packed, causal_history_packed}
use std.nn.recurrent::{gated_delta_decode}

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

pub entry update<B: Dim, Hk: Dim, Hv: Dim, K: Dim, V: Dim>(
    key: Tensor[B, Hk, K; f32],
    value: Tensor[B, Hv, V; f32],
    decay: Tensor[B, Hv; f32],
    beta: Tensor[B, Hv; f32],
    initial: Tensor[B, Hv, K, V; f32],
) -> Tensor[B, Hv, K, V; f32]
where
    Hk > 0,
    Hv % Hk == 0
{
    return gated_delta_update(key, value, decay, beta, initial)
}

pub entry decode<B: Dim, Hk: Dim, Hv: Dim, K: Dim, V: Dim>(
    query: Tensor[B, Hk, K; f32],
    key: Tensor[B, Hk, K; f32],
    value: Tensor[B, Hv, V; f32],
    decay: Tensor[B, Hv; f32],
    beta: Tensor[B, Hv; f32],
    current: Tensor[B, 1 + K, Hv, V; f32],
) -> Tensor[B, 1 + K, Hv, V; f32]
where
    Hk > 0,
    Hv % Hk == 0
{
    return gated_delta_decode(query, key, value, decay, beta, current, 0.25)
}

pub entry conv<B: Dim, S: Dim, C: Dim>(
    x: Tensor[B, S, C; f32],
    history: Tensor[B, 3, C; f32],
    weight: Tensor[C, 4; f32],
) -> Tensor[B, S, C; f32] {
    return causal_conv(x, history, weight)
}

pub entry packed<P: Dim, R: Dim, Hk: Dim, Hv: Dim, K: Dim, V: Dim, T: Float>(
    query: Tensor[1, P, Hk, K; T],
    key: Tensor[1, P, Hk, K; T],
    value: Tensor[1, P, Hv, V; T],
    decay: Tensor[1, P, Hv; f32],
    beta: Tensor[1, P, Hv; f32],
    segments: Tensor[P; i32],
    rows: Tensor[P; i32],
    current: Tensor[R, Hv, K, V; f32],
) -> Tensor[P * Hv + R * Hv * K, V; f32]
where
    Hk > 0,
    Hv % Hk == 0
{
    return gated_delta_rule_packed(query, key, value, decay, beta, segments, rows, current, 0.25)
}

pub entry conv_packed<P: Dim, C: Dim>(
    x: Tensor[P, C; f32],
    segments: Tensor[P; i32],
    weight: Tensor[C, 4; f32],
) -> Tensor[P, C; f32] {
    return causal_conv_packed(x, segments, weight)
}

pub entry history_packed<P: Dim, R: Dim, C: Dim>(
    x: Tensor[P, C; f32],
    segments: Tensor[P; i32],
    rows: Tensor[P; i32],
    history: Tensor[R, 3, C; f32],
) -> Tensor[R, 3, C; f32] {
    return causal_history_packed(x, segments, rows, history)
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


def test_an_update_is_one_token_of_the_rule(tmp_path: Path) -> None:
    """The new state alone, through the body and through the kernel's
    PyTorch fallback."""
    source = tmp_path / "delta.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    inputs = _inputs("cpu", 1)
    expected = _reference(*inputs)[:, 1:].permute(0, 2, 1, 3)
    _, key, value, decay, beta, initial = inputs
    arguments = [key[:, 0], value[:, 0], decay[:, 0], beta[:, 0], initial]
    for numerics in ("exact", "fast"):
        update = load_function(source, "update", std_root=STDLIB, numerics=numerics)
        torch.testing.assert_close(update(*arguments).double(), expected, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_the_update_kernel_writes_in_place() -> None:
    """The Triton kernel, into a new state and into the old one itself."""
    _, key, value, decay, beta, initial = _inputs("cuda", 1, torch.bfloat16)
    arguments = (key[:, 0], value[:, 0], decay[:, 0], beta[:, 0])
    expected = gated_delta_update(initial.clone().cpu(), *(a.cpu() for a in arguments))
    torch.testing.assert_close(
        gated_delta_update(initial, *arguments).cpu(), expected, atol=1e-5, rtol=1e-5
    )
    state = initial.clone()
    same = gated_delta_update_(state, *arguments)
    assert same.data_ptr() == state.data_ptr()
    torch.testing.assert_close(state.cpu(), expected, atol=1e-5, rtol=1e-5)


STATEFUL = """\
module tests.delta_state

use std.nn.recurrent::{gated_delta_update}

pub block Rows<B: Dim, H: Dim, K: Dim, V: Dim>
where H > 0 {
    state memory: Tensor[B, H, K, V; f32]

    pub entry step(
        key: Tensor[B, H, K; f32],
        value: Tensor[B, H, V; f32],
        decay: Tensor[B, H; f32],
        beta: Tensor[B, H; f32],
    ) -> Tensor[B, H, V; f32] {
        memory = gated_delta_update(key, value, decay, beta, memory)
        let read[b, h, v] = sum[k] memory[b, h, k, v] * key[b, h, k]
        return read
    }
}
"""


def test_a_state_only_the_update_reads_is_updated_in_place(tmp_path: Path) -> None:
    """The generated step writes the state where it lies (`IN_PLACE`) when
    the update is the only reader of its old value."""
    source = tmp_path / "state.linnet"
    source.write_text(STATEFUL, encoding="utf-8")
    command = [find_compiler(), "torch", "--root", "Rows", "--entry", "step", "--numerics", "fast"]
    command += ["--std", str(STDLIB), "--bind", "B=2", "--bind", "H=2", "--bind", "K=8"]
    command += ["--bind", "V=4", str(source)]
    text = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    assert "_gated_delta_update_(" in text
    assert 'IN_PLACE = ["memory"]' in text


# Three prompts packed as `linnet.serve` packs them, into rows 2, 0 and 3 of
# four, then two tokens of padding; row 1 keeps its state.
LENGTHS, ROWS, PAD = [3, 1, 70], [2, 0, 3], 2


def _layout(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    segments = [m for m, n in enumerate(LENGTHS) for _ in range(n)] + [-1] * PAD
    rows = [r for r, n in zip(ROWS, LENGTHS, strict=True) for _ in range(n)] + [ROWS[0]] * PAD
    return (
        torch.tensor(segments, dtype=torch.int32, device=device),
        torch.tensor(rows, dtype=torch.int32, device=device),
    )


def _packed_reference(inputs: tuple[torch.Tensor, ...], current: torch.Tensor) -> torch.Tensor:
    """Each prompt alone from zero (`_reference`), laid out as the packed op's
    result: the outputs (zero for padding), then the rows' states."""
    key, value = inputs[1], inputs[2]
    heads, width = value.shape[2], value.shape[3]
    out = torch.zeros(value.shape[1], heads, width, dtype=torch.float64)
    state = current.double().cpu().clone()
    start = 0
    for row, length in zip(ROWS, LENGTHS, strict=True):
        end = start + length
        zero = torch.zeros(1, heads, key.shape[3], width)
        query, key_part, value_part, decay, beta = (x[:, start:end].cpu() for x in inputs)
        alone = _reference(query, key_part, value_part, decay, beta, zero)
        out[start:end] = alone[0, :length]
        state[row] = alone[0, length:].permute(1, 0, 2)
        start = end
    return torch.cat([out.reshape(-1, width), state.reshape(-1, width)])


@pytest.mark.parametrize("numerics", ["exact", "fast"])
def test_packed_prompts_are_each_their_own(tmp_path: Path, numerics: str) -> None:
    """The body and the PyTorch kernel: every prompt from zero, its final
    state in its row, the padding and the other rows untouched."""
    source = tmp_path / "delta.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    packed = load_function(source, "packed", std_root=STDLIB, numerics=numerics)
    query, key, value, decay, beta, _ = _inputs("cpu", sum(LENGTHS) + PAD, batch=1)
    current = torch.randn(4, 4, 16, 8, generator=torch.Generator().manual_seed(2))
    segments, rows = _layout("cpu")
    inputs = (query, key, value, decay, beta)
    got = packed(*inputs, segments, rows, current)
    expected = _packed_reference(inputs, current)
    torch.testing.assert_close(got.double(), expected, atol=1e-4, rtol=1e-4)


@requires_cuda
def test_packed_prompts_on_cuda_replay_as_a_graph() -> None:
    """bf16 on CUDA (`fla`'s kernel where it is installed) captured as a CUDA
    graph once, then replayed for another packing of the same size."""
    total = sum(LENGTHS) + PAD
    query, key, value, decay, beta, _ = _inputs("cuda", total, torch.bfloat16, batch=1)
    # Slices of one projection, as a model's heads are: not contiguous.
    value = torch.cat([value, torch.zeros_like(value)], dim=-1)[..., :8]
    current = torch.randn(4, 4, 16, 8, device="cuda") * 0.1
    segments, rows = _layout("cuda")
    inputs = (query, key, value, decay, beta)
    first = gated_delta_rule_packed(*inputs, segments, rows, current, 0.25)
    assert not first.isnan().any()
    torch.testing.assert_close(
        first.double().cpu(), _packed_reference(inputs, current), atol=3e-2, rtol=3e-2
    )
    # Captured with every token one prompt in row 1, replayed with the
    # layout above.
    static = [segments.clone().zero_(), rows.clone().fill_(1)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gated_delta_rule_packed(*inputs, *static, current, 0.25)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = gated_delta_rule_packed(*inputs, *static, current, 0.25)
    static[0].copy_(segments)
    static[1].copy_(rows)
    graph.replay()
    torch.testing.assert_close(replayed, first, atol=1e-5, rtol=1e-5)


def test_packed_convolution_stays_in_each_prompt(tmp_path: Path) -> None:
    """Each prompt convolved from a zero history; each row's new history its
    prompt's last three inputs, zero-filled before a short prompt's start."""
    source = tmp_path / "delta.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generator = torch.Generator().manual_seed(3)
    total = sum(LENGTHS) + PAD
    x = torch.randn(total, 5, generator=generator)
    weight = torch.randn(5, 4, generator=generator)
    history = torch.randn(4, 3, 5, generator=generator)
    segments, rows = _layout("cpu")
    expected = torch.zeros(total, 5)
    windows = history.clone()
    start = 0
    for row, length in zip(ROWS, LENGTHS, strict=True):
        end = start + length
        signal = torch.cat([torch.zeros(3, 5), x[start:end]])
        expected[start:end] = torch.nn.functional.conv1d(signal.T[None], weight[:, None], groups=5)[
            0
        ].T
        windows[row] = signal[-3:]
        start = end
    for numerics in ("exact", "fast"):
        conv = load_function(source, "conv_packed", std_root=STDLIB, numerics=numerics)
        ends = load_function(source, "history_packed", std_root=STDLIB, numerics=numerics)
        got = conv(x, segments, weight)
        torch.testing.assert_close(got[:start], expected[:start], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ends(x, segments, rows, history), windows)


def _decoder_state(initial: torch.Tensor) -> torch.Tensor:
    """`initial` `[B, Hv, K, V]` as a decoder keeps it, `[B, 1 + K, Hv, V]`,
    row 0 something the step does not read."""
    junk = torch.full_like(initial[:, :, :1], 7.0).permute(0, 2, 1, 3)
    return torch.cat([junk, initial.permute(0, 2, 1, 3)], dim=1).contiguous()


def test_a_decode_is_one_token_of_the_rule(tmp_path: Path) -> None:
    """The output and the new state in the rule's own layout, through the
    body and through the PyTorch fallback."""
    source = tmp_path / "delta.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    inputs = _inputs("cpu", 1)
    expected = _reference(*inputs)
    query, key, value, decay, beta, initial = inputs
    arguments = [query[:, 0], key[:, 0], value[:, 0], decay[:, 0], beta[:, 0]]
    for numerics in ("exact", "fast"):
        decode = load_function(source, "decode", std_root=STDLIB, numerics=numerics)
        got = decode(*arguments, _decoder_state(initial))
        torch.testing.assert_close(got.double(), expected, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_the_decode_kernel_writes_in_place() -> None:
    """The Triton kernel, into a new state and into the old one itself."""
    query, key, value, decay, beta, initial = _inputs("cuda", 1, torch.bfloat16)
    arguments = (query[:, 0], key[:, 0], value[:, 0], decay[:, 0], beta[:, 0])
    state = _decoder_state(initial)
    q, k, v, g, b = (a.cpu() for a in arguments)
    expected = gated_delta_decode(state.cpu(), q, k, v, g, b, 0.25)
    torch.testing.assert_close(
        gated_delta_decode(state, *arguments, 0.25).cpu(), expected, atol=1e-5, rtol=1e-5
    )
    same = gated_delta_decode_(state, *arguments, 0.25)
    assert same.data_ptr() == state.data_ptr()
    torch.testing.assert_close(state.cpu(), expected, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_a_bf16_decoding_state_steps_in_f32() -> None:
    """A state kept in bf16: the kernel steps it in f32 and rounds once on
    store, as the f32 step rounded to bf16 does."""
    query, key, value, decay, beta, initial = _inputs("cuda", 1, torch.bfloat16)
    arguments = (query[:, 0], key[:, 0], value[:, 0], decay[:, 0], beta[:, 0])
    state = _decoder_state(initial).bfloat16()
    expected = gated_delta_decode(state.float(), *arguments, 0.25).bfloat16()
    torch.testing.assert_close(gated_delta_decode(state, *arguments, 0.25), expected)
    gated_delta_decode_(state, *arguments, 0.25)
    assert state.dtype == torch.bfloat16
    torch.testing.assert_close(state, expected)


DECODING = """\
module tests.delta_decoding

use std.nn.recurrent::{gated_delta_decode}

pub block Rows<B: Dim, H: Dim, K: Dim, V: Dim>
where H > 0 {
    state memory: Tensor[B, 1 + K, H, V; f32]

    pub entry step(
        query: Tensor[B, H, K; f32],
        key: Tensor[B, H, K; f32],
        value: Tensor[B, H, V; f32],
        decay: Tensor[B, H; f32],
        beta: Tensor[B, H; f32],
    ) -> Tensor[B, H, V; f32] {
        memory = gated_delta_decode(query, key, value, decay, beta, memory, 0.5)
        return reshape(memory[:, 0:1, :, :], [B, H, V])
    }
}
"""


def test_a_decoding_state_is_updated_in_place(tmp_path: Path) -> None:
    """The step writes the state and its output where they lie, then reads
    the output from row 0."""
    source = tmp_path / "decoding.linnet"
    source.write_text(DECODING, encoding="utf-8")
    command = [find_compiler(), "torch", "--root", "Rows", "--entry", "step", "--numerics", "fast"]
    command += ["--std", str(STDLIB), "--bind", "B=2", "--bind", "H=2", "--bind", "K=8"]
    command += ["--bind", "V=4", str(source)]
    text = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    assert "_gated_delta_decode_(" in text
    assert 'IN_PLACE = ["memory"]' in text
