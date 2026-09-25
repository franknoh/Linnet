"""Continuous batching (`linnet.serve`): requests decoded together, each at
its own length and joining as rows free up, must each get exactly the tokens
greedy decoding over the whole sequence gives them -- on both backends."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.serve import Engine, Request
from linnet.torch import load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.serve

use std.nn.attention::{causal_mask, grouped_attention, grouped_attention_rows}
use std.nn.cache::{write_rows, write_slot}
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

    pub entry prefill_slot<S: Dim>(
        tokens: Tensor[1, S; i32],
        slot: i32,
        length: i32,
    ) -> Tensor[1, Vocab; T]
    where
        S > 0,
        S <= MaxSeq
    {
        let x = embedding.forward(tokens) + positions.forward(iota<i32>(S))
        let k = heads<1, S>(k_proj.forward(x))
        let v = heads<1, S>(v_proj.forward(x))
        cache_k = write_slot(cache_k, k, slot, 0)
        cache_v = write_slot(cache_v, v, slot, 0)
        let mixed = grouped_attention(
            heads<1, S>(q_proj.forward(x)),
            k,
            v,
            rsqrt(cast<f32>(H / Heads)),
            some(causal_mask<S, S>()),
        )
        let out = x + merge<1, S>(mixed)
        let last[b, h] = out[b, cast<i64>(length) - 1, h]
        return head.forward(last)
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
    for _ in range(request.max_new_tokens):
        tokens = torch.tensor([ids], dtype=torch.int32)
        token = int(model.run_entry("forward", [tokens])[0, -1].argmax())  # type: ignore[operator]
        out.append(token)
        ids.append(token)
    return out


def test_torch(model_files: tuple[Path, Path]) -> None:
    source, weights = model_files
    model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    requests = _requests()
    done, stats = Engine(model, graphs=False, buckets=[8, 16, 32]).run(requests)
    # Eight requests through three rows: some had to wait for a row.
    assert stats.prefills == len(requests) and max(c.admitted for c in done) > 0
    for completion in done:
        assert completion.tokens == _greedy(model, completion.request)
        assert completion.reason == "length"
    assert stats.generated_tokens == sum(r.max_new_tokens for r in requests)


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
    first = model._function("prefill_slot")  # pyright: ignore[reportPrivateUsage]
    second = model._function("decode_rows")  # pyright: ignore[reportPrivateUsage]
    shared = [
        path
        for path in first.weights
        if first.weights[path] is second.weights[path]
        and not isinstance(first.weights[path], np.ndarray)
    ]
    assert len(shared) == len(first.weights)
