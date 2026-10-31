"""Embeddings served (`linnet.serve.embeddings`): inputs of many lengths,
from many requests at once, padded into shared passes, must each get the
vector the model gives that input alone -- and the server must hand them
out as OpenAI's embeddings API does."""

from __future__ import annotations

import base64
import json
import random
import struct
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import pytest
import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

from linnet.serve.embeddings import Embedder, EmbeddingError
from linnet.torch import LinnetModule, load
from tests.helpers import STDLIB

SOURCE = """\
module tests.embeddings

use std.nn.attention::{attention, attention_rows}
use std.nn.embedding::{Embedding}
use std.nn.linear::{Linear}

pub block Encoder<Vocab: Dim, H: Dim, MaxPositions: Dim, T: Float = f32> {
    sub embedding: Embedding<Vocab, H, T>
    sub positions: Embedding<MaxPositions, H, T>
    sub query: Linear<H, H, T>
    sub key: Linear<H, H, T>
    sub value: Linear<H, H, T>

    // Every position pooled.
    pub entry embed<B: Dim, S: Dim>(tokens: Tensor[B, S; i32]) -> Tensor[B, H; f32]
    where S <= MaxPositions {
        let x = mixed(tokens, none)
        let pooled[b, h] = sum<f32>[s] x[b, s, h] / cast<f32>(S)
        return unit(pooled)
    }

    // Sequence `b` is its first `lengths[b]` tokens.
    pub entry embed_padded<B: Dim, S: Dim>(
        tokens: Tensor[B, S; i32],
        lengths: Tensor[B; i32],
    ) -> Tensor[B, H; f32]
    where S <= MaxPositions {
        let places = iota<i32>(S)
        let keep[b, s] = places[s] < lengths[b]
        let x = mixed(tokens, some(broadcast_to(reshape(keep, [B, 1, S]), [B, S, S])))
        let summed[b, h] = sum<f32>[s] select(keep[b, s], x[b, s, h], 0.0)
        let pooled[b, h] = summed[b, h] / cast<f32>(lengths[b])
        return unit(pooled)
    }

    fn mixed<B: Dim, S: Dim>(
        tokens: Tensor[B, S; i32],
        mask: Tensor[B, S, S; bool]?,
    ) -> Tensor[B, S, H; f32] {
        let x = embedding.forward(tokens) + positions.forward(iota<i32>(S))
        let q = reshape(query.forward(x), [B, 1, S, H])
        let k = reshape(key.forward(x), [B, 1, S, H])
        let v = reshape(value.forward(x), [B, 1, S, H])
        let scale = rsqrt(cast<f32>(H))
        let attended = match mask {
            some(m) => attention_rows(q, k, v, scale, m)
            none => attention(q, k, v, scale, none)
        }
        return cast<f32>(x + reshape(attended, [B, S, H]))
    }

    fn unit<B: Dim>(pooled: Tensor[B, H; f32]) -> Tensor[B, H; f32] {
        let squared[b] = sum<f32>[h] pooled[b, h] * pooled[b, h]
        let unit[b, h] = pooled[b, h] * rsqrt(squared[b])
        return unit
    }
}
"""

GENERICS: dict[str, int | str] = {"Vocab": 64, "H": 16, "MaxPositions": 32}

# The same encoder with only `embed`: inputs then share a pass only with
# others of their length.
EXACT = SOURCE[: SOURCE.index("    // Sequence `b`")] + SOURCE[SOURCE.index("    fn mixed") :]


def _model(tmp_path: Path, source: str = SOURCE) -> LinnetModule:
    path = tmp_path / "encoder.linnet"
    path.write_text(source, encoding="utf-8")
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "embedding.weight": torch.randn(64, 16, generator=generator),
        "positions.weight": torch.randn(32, 16, generator=generator),
    }
    for name in ("query", "key", "value"):
        tensors[f"{name}.weight"] = torch.randn(16, 16, generator=generator) * 0.3
    weights = tmp_path / "model.safetensors"
    save_file(tensors, str(weights))
    model = load(path, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    assert isinstance(model, LinnetModule)
    return model


def _alone(model: LinnetModule, ids: Sequence[int]) -> torch.Tensor:
    vector = model.run_entry("embed", [torch.tensor([list(ids)], dtype=torch.int32)])
    assert isinstance(vector, torch.Tensor)
    return vector[0]


def _inputs(count: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(64) for _ in range(rng.randrange(1, 33))] for _ in range(count)]


@pytest.mark.parametrize("exact", [False, True])
def test_inputs_get_what_they_get_alone(tmp_path: Path, exact: bool) -> None:
    model = _model(tmp_path, EXACT if exact else SOURCE)
    embedder = Embedder(model, max_len=32, batch=4)
    assert embedder.entry == ("embed" if exact else "embed_padded")
    embedder.start()
    try:
        inputs = _inputs(11)
        # Several requests at once, sharing passes.
        results: dict[int, list[list[float]]] = {}

        def request(index: int) -> None:
            results[index] = embedder.embed(inputs[index::3])

        threads = [threading.Thread(target=request, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        for index in range(3):
            for ids, vector in zip(inputs[index::3], results[index], strict=True):
                expected = _alone(model, ids)
                torch.testing.assert_close(torch.tensor(vector), expected, atol=1e-5, rtol=1e-5)
    finally:
        embedder.stop()


def test_padded_shapes_are_few(tmp_path: Path) -> None:
    embedder = Embedder(_model(tmp_path), max_len=32, batch=6)
    assert embedder.lengths == [8, 16, 32]
    assert embedder.sizes == [1, 2, 4, 6]
    embedder.warmup()


def test_inputs_are_checked(tmp_path: Path) -> None:
    embedder = Embedder(_model(tmp_path), max_len=32)
    with pytest.raises(EmbeddingError, match="more than"):
        embedder.check([[1] * 33])
    with pytest.raises(EmbeddingError, match="empty"):
        embedder.check([[1], []])
    with pytest.raises(EmbeddingError, match="empty"):
        embedder.check([])


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ."


class _Letters:
    """A tokenizer of one letter per token, over the test model's 64."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ALPHABET.index(c) for c in text]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(ALPHABET[i] for i in token_ids)

    def apply_chat_template(self, conversation: Sequence[object], **options: bool) -> str:
        raise NotImplementedError


_Json: TypeAlias = "bool | int | float | str | Sequence[_Json] | Mapping[str, _Json] | None"


def _post(url: str, body: Mapping[str, _Json]) -> tuple[int, str]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def test_server_speaks_openai_embeddings(tmp_path: Path) -> None:
    from linnet.serve.server import Server

    model = _model(tmp_path)
    letters = _Letters()
    server = Server(Embedder(model, max_len=32), letters, name="tiny-embed", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.address
    base = f"http://{host}:{port}/v1"
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=60) as response:
            assert json.loads(response.read())["data"][0]["id"] == "tiny-embed"

        texts = ["Hello world", "Hi", "A longer sentence."]
        status, text = _post(f"{base}/embeddings", {"input": texts, "model": "tiny-embed"})
        answer = json.loads(text)
        assert status == 200 and answer["object"] == "list" and answer["model"] == "tiny-embed"
        assert [item["index"] for item in answer["data"]] == [0, 1, 2]
        assert all(item["object"] == "embedding" for item in answer["data"])
        for item, sentence in zip(answer["data"], texts, strict=True):
            expected = _alone(model, letters.encode(sentence))
            torch.testing.assert_close(
                torch.tensor(item["embedding"]), expected, atol=1e-5, rtol=1e-5
            )
        tokens = sum(len(t) for t in texts)
        assert answer["usage"] == {"prompt_tokens": tokens, "total_tokens": tokens}

        # One string, token ids, and lists of token ids.
        one = json.loads(_post(f"{base}/embeddings", {"input": "Hi"})[1])["data"]
        ids = json.loads(_post(f"{base}/embeddings", {"input": letters.encode("Hi")})[1])["data"]
        lists = json.loads(_post(f"{base}/embeddings", {"input": [[33, 34], "Hi"]})[1])["data"]
        assert len(one) == len(ids) == 1 and len(lists) == 2
        assert one[0]["embedding"] == ids[0]["embedding"] == answer["data"][1]["embedding"]

        # base64: the same vector as little-endian f32 bytes.
        packed = json.loads(
            _post(f"{base}/embeddings", {"input": "Hi", "encoding_format": "base64"})[1]
        )["data"][0]["embedding"]
        raw = base64.b64decode(packed)
        unpacked = struct.unpack(f"<{len(raw) // 4}f", raw)
        torch.testing.assert_close(torch.tensor(unpacked), torch.tensor(one[0]["embedding"]))
        width = len(one[0]["embedding"])
        assert _post(f"{base}/embeddings", {"input": "Hi", "dimensions": width})[0] == 200

        assert _post(f"{base}/embeddings", {"input": "Hi", "dimensions": 4})[0] == 400
        assert _post(f"{base}/embeddings", {"input": "Hi", "encoding_format": "x"})[0] == 400
        assert _post(f"{base}/embeddings", {"input": []})[0] == 400
        assert _post(f"{base}/embeddings", {"input": [1.5]})[0] == 400
        assert _post(f"{base}/embeddings", {"input": "x" * 33})[0] == 400
        assert _post(f"{base}/completions", {"prompt": "Hi"})[0] == 404
        assert _post(f"{base}/chat/completions", {"messages": []})[0] == 404
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=60) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        thread.join(timeout=10)
