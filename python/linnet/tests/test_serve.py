"""Continuous batching (`linnet.serve`): requests decoded together, each at
its own length and joining as rows free up, must each get exactly the tokens
greedy decoding over the whole sequence gives them -- on both backends."""

from __future__ import annotations

import json
import os
import random
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.serve import Completion, Engine, Request
from linnet.torch import LinnetModule, load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.serve

use std.nn.attention::{causal_mask, grouped_attention, grouped_attention_rows}
use std.nn.cache::{write_rows, write_slots, write_tokens}
use std.nn.embedding::{Embedding}
use std.nn.linear::{Linear}

pub block Model<Vocab: Dim, H: Dim, Heads: Dim, Batch: Dim, MaxSeq: Dim, T: Float = f32>
where
    Heads > 0,
    H % Heads == 0,
    MaxSeq > 0
{
    sub embedding: Embedding<Vocab, H, T>
    sub positions: Embedding<MaxSeq, H, T>
    sub q_proj: Linear<H, H, T>
    sub k_proj: Linear<H, H, T>
    sub v_proj: Linear<H, H, T>
    sub head: Linear<H, Vocab, T>

    state cache_k: Tensor[Batch, Heads, MaxSeq, H / Heads; T]
    state cache_v: Tensor[Batch, Heads, MaxSeq, H / Heads; T]

    pub entry forward<S: Dim>(tokens: Tensor[1, S; i32]) -> Tensor[1, S, Vocab; T]
    where S <= MaxSeq {
        let x = embedding.forward(tokens) + positions.forward(iota<i32>(S))
        let mixed = grouped_attention(
            heads<1, S>(q_proj.forward(x)),
            heads<1, S>(k_proj.forward(x)),
            heads<1, S>(v_proj.forward(x)),
            rsqrt(cast<f32>(H / Heads)),
            some(causal_mask<S, S>()),
        )
        return head.forward(x + merge<1, S>(mixed))
    }

    pub entry prefill_slots<M: Dim, S: Dim>(
        tokens: Tensor[M, S; i32],
        slots: Tensor[M; i32],
        lengths: Tensor[M; i32],
    ) -> Tensor[M, Vocab; T]
    where
        M > 0,
        S > 0,
        S <= MaxSeq
    {
        let x = embedding.forward(tokens) + positions.forward(iota<i32>(S))
        let k = heads<M, S>(k_proj.forward(x))
        let v = heads<M, S>(v_proj.forward(x))
        cache_k = write_slots(cache_k, k, slots, 0)
        cache_v = write_slots(cache_v, v, slots, 0)
        let mixed = grouped_attention(
            heads<M, S>(q_proj.forward(x)),
            k,
            v,
            rsqrt(cast<f32>(H / Heads)),
            some(causal_mask<S, S>()),
        )
        let out = x + merge<M, S>(mixed)
        let last[b, h] = out[b, cast<i64>(lengths[b]) - 1, h]
        return head.forward(last)
    }

    pub entry prefill_packed<P: Dim>(
        tokens: Tensor[P; i32],
        rows: Tensor[P; i32],
        at: Tensor[P; i32],
        segments: Tensor[P; i32],
        last: Tensor[Batch; i32],
    ) -> Tensor[Batch, Vocab; T]
    where P > 0 {
        let placed = reshape(positions.forward(at), [1, P, H])
        let x = embedding.forward(reshape(tokens, [1, P])) + placed
        let k = heads<1, P>(k_proj.forward(x))
        let v = heads<1, P>(v_proj.forward(x))
        cache_k = write_tokens(cache_k, k, rows, at)
        cache_v = write_tokens(cache_v, v, rows, at)
        let own[i, j] = segments[i] == segments[j] && at[j] <= at[i]
        let mixed = grouped_attention(
            heads<1, P>(q_proj.forward(x)),
            k,
            v,
            rsqrt(cast<f32>(H / Heads)),
            some(own),
        )
        let out = x + merge<1, P>(mixed)
        let ends[m, h] = out[0, cast<i64>(last[m]), h]
        return head.forward(ends)
    }

    // `prefill_packed`'s prompts and `decode_rows`'s step for every row in one
    // pass: the projections run once over all `P + Batch` tokens, each
    // token's key and value go to its row at its position, and the prompts
    // attend among themselves while each row's step attends over its row.
    // Logits after each prompt's last token, then each row's step.
    pub entry step_packed<P: Dim>(
        tokens: Tensor[P; i32],
        rows: Tensor[P; i32],
        at: Tensor[P; i32],
        segments: Tensor[P; i32],
        last: Tensor[Batch; i32],
        step_tokens: Tensor[Batch, 1; i32],
        step_at: Tensor[Batch; i32],
    ) -> Tensor[2 * Batch, Vocab; T]
    where P > 0 {
        let every = concat(tokens, reshape(step_tokens, [Batch]), axis = 0)
        let every_at = concat(at, step_at, axis = 0)
        let x =
            embedding.forward(reshape(every, [1, P + Batch])) +
            reshape(positions.forward(every_at), [1, P + Batch, H])
        let k = heads<1, P + Batch>(k_proj.forward(x))
        let v = heads<1, P + Batch>(v_proj.forward(x))
        let every_row = concat(rows, iota<i32>(Batch), axis = 0)
        cache_k = write_tokens(cache_k, k, every_row, every_at)
        cache_v = write_tokens(cache_v, v, every_row, every_at)
        let q = heads<1, P + Batch>(q_proj.forward(x))
        let own[i, j] = segments[i] == segments[j] && at[j] <= at[i]
        let prompts = grouped_attention(
            q[:, :, 0:P, :],
            k[:, :, 0:P, :],
            v[:, :, 0:P, :],
            rsqrt(cast<f32>(H / Heads)),
            some(own),
        )
        let slots = iota<i32>(MaxSeq)
        let seen[b, s] = slots[s] <= step_at[b]
        let steps = grouped_attention_rows(
            permute(q[:, :, P:P + Batch, :], [2, 1, 0, 3]),
            cache_k,
            cache_v,
            rsqrt(cast<f32>(H / Heads)),
            reshape(seen, [Batch, 1, MaxSeq]),
        )
        let mixed = concat(
            merge<1, P>(prompts),
            permute(merge<Batch, 1>(steps), [1, 0, 2]),
            axis = 1,
        )
        let out = x + mixed
        let ends[m, h] = out[0, cast<i64>(last[m]), h]
        return head.forward(concat(ends, out[0, P:P + Batch, :], axis = 0))
    }

    pub entry decode_rows(
        tokens: Tensor[Batch, 1; i32],
        at: Tensor[Batch; i32],
    ) -> Tensor[Batch, Vocab; T] {
        let x = embedding.forward(tokens) + reshape(positions.forward(at), [Batch, 1, H])
        cache_k = write_rows(cache_k, heads<Batch, 1>(k_proj.forward(x)), at)
        cache_v = write_rows(cache_v, heads<Batch, 1>(v_proj.forward(x)), at)
        let slots = iota<i32>(MaxSeq)
        let seen[b, s] = slots[s] <= at[b]
        let mixed = grouped_attention_rows(
            heads<Batch, 1>(q_proj.forward(x)),
            cache_k,
            cache_v,
            rsqrt(cast<f32>(H / Heads)),
            reshape(seen, [Batch, 1, MaxSeq]),
        )
        return head.forward((x + merge<Batch, 1>(mixed))[:, 0, :])
    }

    fn heads<B: Dim, S: Dim>(x: Tensor[B, S, H; T]) -> Tensor[B, Heads, S, H / Heads; T] {
        return permute(reshape(x, [B, S, Heads, H / Heads]), [0, 2, 1, 3])
    }

    fn merge<B: Dim, S: Dim>(x: Tensor[B, Heads, S, H / Heads; T]) -> Tensor[B, S, H; T] {
        return reshape(permute(x, [0, 2, 1, 3]), [B, S, H])
    }
}
"""

GENERICS: dict[str, int | str] = {"Vocab": 64, "H": 32, "Heads": 4, "Batch": 3, "MaxSeq": 48}


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.fixture
def model_files(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "serve.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generator = torch.Generator().manual_seed(0)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    tensors = {"embedding.weight": normal(64, 32), "positions.weight": normal(48, 32)}
    for name in ("q_proj", "k_proj", "v_proj"):
        tensors[f"{name}.weight"] = normal(32, 32) * 0.3
    tensors["head.weight"] = normal(64, 32)
    weights = tmp_path / "model.safetensors"
    save_file(tensors, str(weights))
    return source, weights


def _requests() -> list[Request]:
    rng = random.Random(0)
    return [
        Request(
            prompt=[rng.randrange(64) for _ in range(rng.randrange(1, 20))],
            max_new_tokens=rng.randrange(1, 12),
            id=i,
        )
        for i in range(8)
    ]


def _greedy(model: torch.nn.Module, request: Request) -> list[int]:
    """Greedy decoding without a cache: the whole sequence every token."""
    ids = list(request.prompt)
    out: list[int] = []
    device = next(iter(model.parameters())).device
    for _ in range(request.max_new_tokens):
        tokens = torch.tensor([ids], dtype=torch.int32, device=device)
        token = int(model.run_entry("forward", [tokens])[0, -1].argmax())  # type: ignore[operator]
        out.append(token)
        ids.append(token)
    return out


def test_load_weights_serves_the_new_weights(model_files: tuple[Path, Path]) -> None:
    """A model trained elsewhere copied into a running engine: its compiled
    passes then decode with the new weights, the old never seen again."""
    source, weights = model_files
    served = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    engine = Engine(served, graphs=False, buckets=[8, 16, 32])
    requests = _requests()
    engine.run(requests)
    trained = load(source, generics=GENERICS, std_root=STDLIB, weights=weights)
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.mul_(1.5).add_(0.1)
    engine.load_weights(trained)
    done, _ = engine.run(requests)
    for completion in done:
        assert completion.tokens == _greedy(trained, completion.request)


def test_identical_prompts_share_one_pass(model_files: tuple[Path, Path]) -> None:
    """Four samples each of two prompts: each prompt passes once, the rows
    sharing it copy its cache, and every row draws what it would alone."""
    source, weights = model_files
    requests = [
        Request(prompt=[5 + i, 9, 2, 7], max_new_tokens=6, temperature=1.0, seed=100 * i + k)
        for i in range(2)
        for k in range(4)
    ]

    def run(share: bool) -> tuple[list[list[int]], list[int]]:
        model = load(
            source,
            generics={**GENERICS, "Batch": 8},
            std_root=STDLIB,
            weights=weights,
            compile=True,
        )
        engine = Engine(model, graphs=False, pack=48, share=share)
        passed: list[int] = []
        prefill = engine.backend.prefill_packed

        def counting(prompts: list[list[int]], *args: Any, **kwargs: Any) -> Any:
            passed.append(sum(len(p) for p in prompts))
            return prefill(prompts, *args, **kwargs)

        engine.backend.prefill_packed = counting  # type: ignore[method-assign]
        done, _ = engine.run(requests)
        return [c.tokens for c in done], passed

    shared, shared_passed = run(True)
    alone, alone_passed = run(False)
    assert shared == alone
    assert sum(shared_passed) == 8 and sum(alone_passed) == 32
    assert len({tuple(tokens) for tokens in shared[:4]}) > 1  # each row its own draws


@pytest.mark.parametrize("pack", [48, 0])
def test_torch(model_files: tuple[Path, Path], pack: int) -> None:
    """Prompts packed end to end into passes (`prefill_packed`), or grouped
    and padded (`prefill_slots`, `pack=0`): the same tokens either way."""
    source, weights = model_files
    model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    requests = _requests()
    engine = Engine(model, graphs=False, buckets=[8, 16, 32], pack=pack)
    assert bool(engine.pack) == (pack > 0)
    done, stats = engine.run(requests)
    # Eight requests through three rows: some had to wait for a row, and the
    # first three prompts went through together.
    assert stats.prefills < len(requests) and max(c.admitted for c in done) > 0
    for completion in done:
        assert completion.tokens == _greedy(model, completion.request)
        assert completion.reason == "length"
    assert stats.generated_tokens == sum(r.max_new_tokens for r in requests)


def test_packs_fill_passes_longest_first(model_files: tuple[Path, Path]) -> None:
    """First fit by decreasing length, at most `pack` tokens and `Batch`
    prompts a pass, each pass padded to the smallest compiled size."""
    source, weights = model_files
    model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    engine = Engine(model, graphs=False, pack=48)
    assert engine.pack_sizes == [16, 32, 48, 64, 96, 128]
    placed = [
        (slot, Completion(Request(prompt=[1] * n, max_new_tokens=1)))
        for slot, n in enumerate([30, 100, 40])
    ]
    passes = engine._packed(placed)  # pyright: ignore[reportPrivateUsage]
    assert [[len(c.request.prompt) for _, c in p] for p in passes] == [[100], [40, 30]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_torch_cuda_graphs(model_files: tuple[Path, Path]) -> None:
    """On CUDA the step replays as a CUDA graph, its tokens stay on the
    device, and each step is queued before the last one is read: rows that
    finish are seen a step late and their extra token is dropped."""
    source, weights = model_files
    model = load(
        source, generics=GENERICS, std_root=STDLIB, weights=weights, device="cuda", compile=True
    )
    requests = _requests()
    done, stats = Engine(model, buckets=[8, 16, 32]).run(requests)
    for completion in done:
        assert completion.tokens == _greedy(model, completion.request)
    assert stats.generated_tokens == sum(r.max_new_tokens for r in requests)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mixed_passes_on_flex_attention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Heads 16 wide and passes of 128 tokens and more, so the prompts and
    the rows' steps both run on FlexAttention, replayed as CUDA graphs: the
    passes that take the rows' steps along give greedy decoding's tokens."""
    source = tmp_path / "serve.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generics: dict[str, int | str] = {"Vocab": 64, "H": 64, "Heads": 4, "Batch": 4, "MaxSeq": 256}
    model = load(source, generics=generics, std_root=STDLIB, device="cuda", compile=True)
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, device="cuda", generator=generator) * 0.02)
    rng = random.Random(0)
    requests = [
        Request(
            prompt=[rng.randrange(64) for _ in range(rng.randrange(20, 120))],
            max_new_tokens=rng.randrange(4, 40),
        )
        for _ in range(16)
    ]
    engine = Engine(model, pack=1024)
    assert engine.mix and engine.pack_sizes[0] == 128
    passes: list[int] = []
    step_packed = engine.backend.step_packed

    def counted(*args: Any, **kwargs: Any) -> Any:
        passes.append(len(args[0]))
        return step_packed(*args, **kwargs)

    monkeypatch.setattr(engine.backend, "step_packed", counted)
    done, _ = engine.run(requests)
    assert passes
    for completion in done:
        assert completion.tokens == _greedy(model, completion.request)


def test_stops_at_eos_and_at_the_end_of_the_cache(model_files: tuple[Path, Path]) -> None:
    source, weights = model_files
    model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    engine = Engine(model, graphs=False, buckets=[8, 16, 32, 48])
    first = _greedy(model, Request(prompt=[5, 6, 7], max_new_tokens=4))
    done, _ = engine.run([Request(prompt=[5, 6, 7], max_new_tokens=4, eos=frozenset({first[1]}))])
    assert done[0].tokens == first[:2] and done[0].reason == "eos"
    done, _ = engine.run([Request(prompt=list(range(40)), max_new_tokens=100)])
    # Positions 40..47 hold the prompt's successors: the last token is the
    # one whose position would be the cache's 49th.
    assert done[0].reason == "cache" and len(done[0].tokens) == 48 - 40


def test_jax(model_files: tuple[Path, Path]) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load_model

    source, weights = model_files
    torch_model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    model = load_model(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    requests = _requests()
    done, _ = Engine(model, buckets=[8, 16, 32]).run(requests)
    for completion in done:
        assert completion.tokens == _greedy(torch_model, completion.request)
    # One copy of each weight, shared by every entry the engine ran.
    first = model._function("prefill_slots")  # pyright: ignore[reportPrivateUsage]
    second = model._function("decode_rows")  # pyright: ignore[reportPrivateUsage]
    shared = [
        path
        for path in first.weights
        if first.weights[path] is second.weights[path]
        and not isinstance(first.weights[path], np.ndarray)
    ]
    assert len(shared) == len(first.weights)


def test_jax_load_weights_serves_the_new_weights(model_files: tuple[Path, Path]) -> None:
    """New weights copied into a JAX model an engine already ran: its
    compiled passes then decode with them."""
    pytest.importorskip("jax")
    from linnet.jax import load_model
    from linnet.jax.source import SourceFunction

    source, weights = model_files
    model = load_model(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    engine = Engine(model, buckets=[8, 16, 32])
    requests = _requests()
    engine.run(requests)
    trained = load(source, generics=GENERICS, std_root=STDLIB, weights=weights)
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.mul_(1.5).add_(0.1)
    engine.load_weights(
        {n.removeprefix("root."): p.detach().numpy() for n, p in trained.named_parameters()}
    )
    done, _ = engine.run(requests)
    for completion in done:
        assert completion.tokens == _greedy(trained, completion.request)
    # Nothing holds the weights before: every function's loaded arrays are
    # the new ones. Arrays already in place are taken, not copied.
    for function in model._functions.values():  # pyright: ignore[reportPrivateUsage]
        assert isinstance(function, SourceFunction)
        loaded = function.parameters
        assert loaded and all(loaded[path] is model.weights[path] for path in loaded)
    placed = {path: value * 2 for path, value in model.weights.items()}
    engine.load_weights(placed)
    assert all(model.weights[path] is placed[path] for path in placed)


def test_onnx(model_files: tuple[Path, Path]) -> None:
    """ONNX Runtime serves the same tokens, over one copy of each weight
    bound to every entry's session, with the caches kept between calls."""
    pytest.importorskip("onnxruntime")
    from linnet.onnx import load_model as load_onnx

    source, weights = model_files
    torch_model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    model = load_onnx(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    requests = _requests()
    done, _ = Engine(model, buckets=[8, 16, 32]).run(requests)
    for completion in done:
        assert completion.tokens == _greedy(torch_model, completion.request)
    # prefill_slots at several shapes and decode_rows, all over six weights.
    assert len(model._sessions) > 2  # pyright: ignore[reportPrivateUsage]
    assert len(model._weights) == 6  # pyright: ignore[reportPrivateUsage]


def test_onnx_step_replays_as_a_cuda_graph(model_files: tuple[Path, Path]) -> None:
    """On a GPU the step is captured once and replayed: the caches are
    scattered into where they lie, so their buffers never move, and the
    tokens are greedy decoding's."""
    onnxruntime = pytest.importorskip("onnxruntime")
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        pytest.skip("CUDA graphs need ONNX Runtime's CUDA provider")
    from linnet.onnx import load_model as load_onnx

    source, weights = model_files
    torch_model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    model = load_onnx(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    done, _ = Engine(model, buckets=[8, 16, 32]).run(_requests())
    for completion in done:
        assert completion.tokens == _greedy(torch_model, completion.request)
    sessions = model._sessions.values()  # pyright: ignore[reportPrivateUsage]
    graphed = [session for session in sessions if session.graphed]
    assert len(graphed) == 1 and graphed[0].fixed is not None
    assert graphed[0].fixed.current(model)


def test_onnx_reads_the_checkpoint_once(
    model_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every prompt length is a session of its own; each tensor is read for
    the first that needs it and only checked against the header after."""
    pytest.importorskip("onnxruntime")
    from linnet import weights as weights_module
    from linnet.onnx import load_model as load_onnx

    source, weights = model_files
    model = load_onnx(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    reads: list[str] = []
    original = weights_module.TensorLocation.read

    def counted(location: weights_module.TensorLocation) -> bytes:
        reads.append(f"{location.file}:{location.start}")
        return original(location)

    monkeypatch.setattr(weights_module.TensorLocation, "read", counted)
    Engine(model, buckets=[8, 16, 32]).warmup([4, 12, 20])
    assert len(model._sessions) > 3  # pyright: ignore[reportPrivateUsage]
    assert sorted(reads) == sorted(set(reads)) and len(reads) == 6


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_onnx_in_sixteen_bits(model_files: tuple[Path, Path], dtype: str) -> None:
    """An f32 checkpoint exported as a 16-bit model (`cast_dtype=True`):
    the same logits within the narrower type's rounding."""
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import load_model as load_onnx

    if dtype == "bf16" and "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        pytest.skip("ONNX Runtime's CPU kernels have no bf16 arithmetic")
    source, weights = model_files
    wide = load_onnx(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    narrow = load_onnx(
        source, generics={**GENERICS, "T": dtype}, weights=weights, std_root=STDLIB, cast_dtype=True
    )
    tokens = np.array([[3, 1, 4, 1, 5, 9, 2, 6]], dtype=np.int32)
    expected = wide.run_entry("forward", [tokens])
    got = np.asarray(narrow.run_entry("forward", [tokens]))
    # Two steps of the type at the logits' own scale: bf16's step at 32 is
    # already 0.25.
    step = {"f16": 2.0**-10, "bf16": 2.0**-7}[dtype] * float(np.abs(expected).max())
    np.testing.assert_allclose(
        got.astype(np.float32), expected, atol=max(0.15, 2 * step), rtol=0.05
    )


def test_onnx_bf16_values_cross_as_bits() -> None:
    """ONNX Runtime has no NumPy type for bf16: inputs go in as rounded bits
    and results come back widened to f32, never truncated like integers."""
    from linnet.onnx.runtime import (
        _decode,  # pyright: ignore[reportPrivateUsage]
        _encode,  # pyright: ignore[reportPrivateUsage]
    )

    values = np.array([0.1, -2.5, 3.14159, 1e-3], dtype=np.float32)
    bits = _encode(values, 16)
    assert bits.dtype == np.uint16
    back = _decode(bits, 16)
    np.testing.assert_allclose(back, values, rtol=4e-3)


def test_steps_ride_along_with_prompts(
    model_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step that admits prompts while other rows decode takes their step
    in the same pass (`step_packed`): the same tokens as passes apart."""
    source, weights = model_files
    model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    requests = _requests()
    apart, _ = Engine(model, graphs=False, pack=48, mix=False).run(requests)
    engine = Engine(model, graphs=False, pack=48)
    assert engine.mix
    passes: list[int] = []
    step_packed = engine.backend.step_packed

    def counted(*args: Any, **kwargs: Any) -> Any:
        passes.append(len(args[0]))
        return step_packed(*args, **kwargs)

    monkeypatch.setattr(engine.backend, "step_packed", counted)
    together, _ = engine.run(requests)
    # Rows freed one by one while others decoded, so prompts went in with
    # their steps.
    assert passes
    assert [c.tokens for c in together] == [c.tokens for c in apart]
    for completion in together:
        assert completion.tokens == _greedy(model, completion.request)


def test_requests_arriving_while_others_run(model_files: tuple[Path, Path]) -> None:
    """`submit` and `step`, as a server uses them: a request that arrives
    mid-decode joins a free row and gets the same tokens as on its own."""
    source, weights = model_files
    model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    engine = Engine(model, graphs=False, buckets=[8, 16, 32])
    early = [engine.submit(r) for r in _requests()[:2]]
    engine.step()
    engine.step()
    late = engine.submit(Request(prompt=[9, 8, 7], max_new_tokens=5))
    finished: list[Completion] = []
    while engine.busy:
        finished += engine.step()
    # `step` reported every request once, when it finished.
    assert sorted(id(c) for c in finished) == sorted(id(c) for c in [*early, late])
    for completion in [*early, late]:
        assert completion.tokens == _greedy(model, completion.request)
    assert late.admitted > 0 and late.reason == "length"


def _sampled() -> list[Request]:
    """`_requests`, most of them sampled, each with a seed of its own."""
    kinds = [(0.9, 0, 1.0), (1.2, 8, 1.0), (0.8, 0, 0.9), (0.0, 0, 1.0)]
    return [
        Request(
            prompt=r.prompt,
            max_new_tokens=r.max_new_tokens + 4,
            id=r.id,
            temperature=kinds[r.id % 4][0],
            top_k=kinds[r.id % 4][1],
            top_p=kinds[r.id % 4][2],
            seed=100 + r.id,
        )
        for r in _requests()
    ]


def _alone(engine: Engine, requests: list[Request]) -> list[list[int]]:
    return [engine.run([request])[0][0].tokens for request in requests]


def test_sampled_requests_draw_alike_in_any_batch(model_files: tuple[Path, Path]) -> None:
    """A seeded request draws the same tokens alone as in a batch with
    others, greedy or not, whichever row it lands in."""
    source, weights = model_files
    model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    engine = Engine(model, graphs=False, buckets=[8, 16, 32])
    requests = _sampled()
    together, _ = engine.run(requests)
    backwards, _ = engine.run(requests[::-1])
    alone = _alone(engine, requests)
    for completion, reversed_, single in zip(together, backwards[::-1], alone, strict=True):
        assert completion.tokens == reversed_.tokens == single
        assert completion.sampling.seed == completion.request.seed
    greedy = [c for c in together if c.request.temperature == 0]
    assert greedy and all(c.tokens == _greedy(model, c.request) for c in greedy)
    drawn = [c for c in together if c.request.temperature > 0]
    assert any(c.tokens != _greedy(model, c.request) for c in drawn)
    # Without a seed, the engine picks one per request.
    unseeded = Request(prompt=[1, 2, 3], max_new_tokens=12, temperature=1.0)
    first, second = engine.run([unseeded, unseeded])[0]
    assert first.sampling.seed != second.sampling.seed


@pytest.mark.parametrize("backend", ["jax", "onnx"])
def test_backends_sample_as_torch_does(model_files: tuple[Path, Path], backend: str) -> None:
    source, weights = model_files
    torch_model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    if backend == "jax":
        pytest.importorskip("jax")
        from linnet.jax import load_model

        model = load_model(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    else:
        pytest.importorskip("onnxruntime")
        from linnet.onnx import load_model as load_onnx

        model = load_onnx(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    requests = _sampled()
    expected, _ = Engine(torch_model, graphs=False, buckets=[8, 16, 32]).run(requests)
    engine = Engine(model, buckets=[8, 16, 32])
    done, _ = engine.run(requests)
    assert [c.tokens for c in done] == [c.tokens for c in expected]
    assert _alone(engine, requests[:3]) == [c.tokens for c in expected[:3]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_torch_cuda_graphs_sample(model_files: tuple[Path, Path]) -> None:
    source, weights = model_files
    reference = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    model = load(
        source, generics=GENERICS, std_root=STDLIB, weights=weights, device="cuda", compile=True
    )
    requests = _sampled()
    expected, _ = Engine(reference, graphs=False, buckets=[8, 16, 32]).run(requests)
    done, _ = Engine(model, buckets=[8, 16, 32]).run(requests)
    assert [c.tokens for c in done] == [c.tokens for c in expected]


def _reference_logprobs(
    model: LinnetModule, request: Request, tokens: list[int], top: int
) -> tuple[list[float], list[list[tuple[int, float]]]]:
    """Each token's log-probability, and the `top` most likely there, from the
    whole sequence run again for every token."""
    ids = list(request.prompt)
    chosen: list[float] = []
    alternatives: list[list[tuple[int, float]]] = []
    for token in tokens:
        logits: torch.Tensor = model.run_entry("forward", [torch.tensor([ids], dtype=torch.int32)])
        scores = logits[0, -1].float().log_softmax(-1)
        chosen.append(float(scores[token]))
        values, best = scores.topk(top)
        alternatives.append([(int(t), float(v)) for t, v in zip(best, values, strict=True)])
        ids.append(token)
    return chosen, alternatives


@pytest.mark.parametrize(
    ("backend", "pack"), [("torch", 48), ("torch", 0), ("jax", 0), ("onnx", 0)]
)
def test_logprobs_are_the_models(model_files: tuple[Path, Path], backend: str, pack: int) -> None:
    """A request with `logprobs` gets its tokens' log-probabilities and the
    most likely tokens with theirs, as the model gives them over the whole
    sequence -- greedy or drawn, beside a request that asked for none."""
    source, weights = model_files
    reference = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    model: Any = reference
    if backend == "jax":
        pytest.importorskip("jax")
        from linnet.jax import load_model

        model = load_model(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    elif backend == "onnx":
        pytest.importorskip("onnxruntime")
        from linnet.onnx import load_model as load_onnx

        model = load_onnx(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    requests = [
        Request(prompt=[3, 9, 4], max_new_tokens=5, id=0, logprobs=3),
        Request(prompt=[7, 1], max_new_tokens=4, id=1),
        Request(prompt=[2, 2, 8, 5], max_new_tokens=6, id=2, logprobs=0, temperature=1.0, seed=3),
    ]
    options: dict[str, Any] = {"graphs": False, "pack": pack} if backend == "torch" else {}
    done, _ = Engine(model, buckets=[8, 16, 32], **options).run(requests)
    assert done[1].logprobs == [] and done[1].top_logprobs == []
    # JAX and ONNX Runtime on a GPU multiply f32 in TF32: about 1e-3 from the
    # CPU reference, relative to the log-probability.
    tf32 = backend != "torch" and torch.cuda.is_available()
    rtol, atol = (2e-3, 5e-3) if tf32 else (1e-4, 1e-4)
    for completion in (done[0], done[2]):
        wanted = completion.request.logprobs
        assert wanted is not None
        chosen, alternatives = _reference_logprobs(
            reference, completion.request, completion.tokens, 3
        )
        np.testing.assert_allclose(completion.logprobs, chosen, rtol=rtol, atol=atol)
        assert [len(top) for top in completion.top_logprobs] == [wanted] * len(completion.tokens)
        for got, expected in zip(completion.top_logprobs, alternatives, strict=True):
            assert [t for t, _ in got] == [t for t, _ in expected[:wanted]]
            np.testing.assert_allclose(
                [v for _, v in got], [v for _, v in expected[:wanted]], rtol=rtol, atol=atol
            )
    # Greedy: each token is the most likely one.
    assert [top[0][0] for top in done[0].top_logprobs] == done[0].tokens


def test_tokens_stream_and_requests_cancel(model_files: tuple[Path, Path]) -> None:
    """`on_token` sees every token as it is read, `reason` set on the last;
    `cancel` ends a request mid-decode and frees its row for the next."""
    source, weights = model_files
    model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    engine = Engine(model, graphs=False, buckets=[8, 16, 32])
    seen: dict[int, list[tuple[int, str]]] = {0: [], 1: [], 2: []}

    def record(completion: Completion) -> None:
        seen[completion.request.id].append((completion.tokens[-1], completion.reason))

    requests = [
        Request(prompt=[3, 4, 5], max_new_tokens=20, id=0, on_token=record),
        Request(prompt=[6, 7], max_new_tokens=6, id=1, on_token=record),
        Request(prompt=[8, 9, 10, 11], max_new_tokens=5, id=2, on_token=record),
    ]
    running = [engine.submit(r) for r in requests]
    waiting = engine.submit(Request(prompt=[1, 2], max_new_tokens=4, id=3))
    while len(running[0].tokens) < 3:
        engine.step()
    engine.cancel(running[0])
    kept = len(running[0].tokens)
    while engine.busy:
        engine.step()
    assert running[0].reason == "cancelled" and len(running[0].tokens) == kept
    assert [t for t, _ in seen[0]] == running[0].tokens and not any(r for _, r in seen[0])
    for completion in [*running[1:], waiting]:
        assert completion.tokens == _greedy(model, completion.request)
    for completion in running[1:]:
        tokens = [t for t, _ in seen[completion.request.id]]
        reasons = [r for _, r in seen[completion.request.id]]
        assert tokens == completion.tokens and reasons[-1] == "length" and not any(reasons[:-1])
    engine.cancel(waiting)  # finished already: left as it is
    assert waiting.reason == "length"


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ."


class _Letters:
    """A tokenizer of one letter per token, over the test model's 64."""

    eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ALPHABET.index(c) for c in text]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(ALPHABET[i] for i in token_ids)

    def apply_chat_template(self, conversation: Any, **options: Any) -> str:
        turns = "".join(f"{m['role'][0].upper()} {m['content']}." for m in conversation)
        return turns + "A "


def _post(url: str, body: dict[str, Any]) -> tuple[int, str]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def _events(text: str) -> list[Any]:
    lines = [part.removeprefix("data: ") for part in text.split("\n\n") if part]
    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


def test_server_speaks_openai(model_files: tuple[Path, Path]) -> None:
    from linnet.serve.server import Server

    source, weights = model_files
    model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    letters = _Letters()
    server = Server(
        Engine(model, graphs=False, buckets=[8, 16, 32, 48]), letters, name="tiny", port=0
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.address
    base = f"http://{host}:{port}/v1"
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=60) as response:
            assert json.loads(response.read())["data"][0]["id"] == "tiny"

        prompt = "Hello"
        expected = letters.decode(
            _greedy(model, Request(prompt=letters.encode(prompt), max_new_tokens=8))
        )
        expected_completion = expected
        greedy = {"prompt": prompt, "max_tokens": 8, "temperature": 0}
        status, text = _post(f"{base}/completions", greedy)
        answer = json.loads(text)
        assert status == 200 and answer["object"] == "text_completion"
        assert answer["choices"][0]["text"] == expected
        assert answer["choices"][0]["finish_reason"] == "length"
        assert answer["usage"] == {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13}

        status, text = _post(
            f"{base}/completions",
            {**greedy, "stream": True, "stream_options": {"include_usage": True}},
        )
        events = _events(text)
        assert "".join(e["choices"][0]["text"] for e in events if e["choices"]) == expected
        assert events[-2]["choices"][0]["finish_reason"] == "length"
        assert events[-1]["usage"]["completion_tokens"] == 8

        # Stopped where a stop string begins, which is not sent.
        stop = expected[3:5]
        status, text = _post(f"{base}/completions", {**greedy, "stop": [stop, "never"]})
        choice = json.loads(text)["choices"][0]
        assert choice["text"] == expected[: expected.index(stop)]
        assert choice["finish_reason"] == "stop"
        status, text = _post(f"{base}/completions", {**greedy, "stop": stop, "stream": True})
        pieces = [e["choices"][0]["text"] for e in _events(text)]
        assert "".join(pieces) == expected[: expected.index(stop)]

        messages = [{"role": "user", "content": "Hi"}]
        chat_prompt = letters.encode(letters.apply_chat_template(messages))
        expected = letters.decode(_greedy(model, Request(prompt=chat_prompt, max_new_tokens=6)))
        chat = {"messages": messages, "max_completion_tokens": 6, "temperature": 0}
        status, text = _post(f"{base}/chat/completions", chat)
        answer = json.loads(text)
        assert answer["object"] == "chat.completion"
        assert answer["choices"][0]["message"] == {"role": "assistant", "content": expected}
        status, text = _post(f"{base}/chat/completions", {**chat, "stream": True})
        events = _events(text)
        assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        assert content == expected and events[-1]["choices"][0]["finish_reason"] == "length"

        # A seeded draw is the same every time.
        drawn = {"prompt": prompt, "max_tokens": 8, "temperature": 1.0, "top_k": 10, "seed": 5}
        assert (
            _post(f"{base}/completions", drawn)[1].split('"text"')[1]
            == _post(f"{base}/completions", drawn)[1].split('"text"')[1]
        )

        # Several choices: each its own request, the same when greedy, and
        # seeded draws continuing from `seed`.
        status, text = _post(f"{base}/completions", {**greedy, "n": 2})
        answer = json.loads(text)
        assert [c["text"] for c in answer["choices"]] == [expected_completion] * 2
        assert [c["index"] for c in answer["choices"]] == [0, 1]
        assert answer["usage"]["completion_tokens"] == 16
        pair = json.loads(_post(f"{base}/completions", {**drawn, "n": 2})[1])["choices"]
        next_seed = json.loads(_post(f"{base}/completions", {**drawn, "seed": 6})[1])["choices"]
        assert pair[1]["text"] == next_seed[0]["text"]
        status, text = _post(f"{base}/completions", {**greedy, "n": 2, "stream": True})
        events = _events(text)
        for index in (0, 1):
            pieces = [e["choices"][0] for e in events if e["choices"][0]["index"] == index]
            assert "".join(p["text"] for p in pieces) == expected_completion
            assert pieces[-1]["finish_reason"] == "length"

        # Log probabilities: the chosen token's, and the most likely tokens'.
        status, text = _post(f"{base}/completions", {**greedy, "logprobs": 2})
        logprobs = json.loads(text)["choices"][0]["logprobs"]
        assert logprobs["tokens"] == list(expected_completion)
        assert logprobs["text_offset"] == list(range(8))
        for value, top in zip(logprobs["token_logprobs"], logprobs["top_logprobs"], strict=True):
            assert len(top) == 2 and value <= 0 and max(top.values()) == value
        status, text = _post(
            f"{base}/chat/completions", {**chat, "logprobs": True, "top_logprobs": 3}
        )
        content = json.loads(text)["choices"][0]["logprobs"]["content"]
        assert "".join(entry["token"] for entry in content) == expected
        assert all(len(entry["top_logprobs"]) == 3 for entry in content)
        assert content[0]["bytes"] == list(content[0]["token"].encode())
        status, text = _post(f"{base}/chat/completions", {**chat, "logprobs": True, "stream": True})
        streamed = [
            entry
            for e in _events(text)
            if e["choices"] and e["choices"][0]["logprobs"]
            for entry in e["choices"][0]["logprobs"]["content"]
        ]
        assert [entry["token"] for entry in streamed] == [entry["token"] for entry in content]

        # `echo` puts the prompt first.
        status, text = _post(f"{base}/completions", {**greedy, "echo": True})
        assert json.loads(text)["choices"][0]["text"] == prompt + expected_completion

        assert _post(f"{base}/completions", {**greedy, "n": 0})[0] == 400
        assert _post(f"{base}/completions", {**greedy, "echo": True, "logprobs": 1})[0] == 400
        assert _post(f"{base}/completions", {**greedy, "logprobs": 21})[0] == 400
        assert _post(f"{base}/chat/completions", {**chat, "top_logprobs": 2})[0] == 400
        assert _post(f"{base}/completions", {**greedy, "top_p": 0})[0] == 400
        assert _post(f"{base}/completions", {"prompt": "x" * 60})[0] == 400
        assert _post(f"{base}/embeddings", {})[0] == 404
    finally:
        server.shutdown()
        thread.join(timeout=10)
