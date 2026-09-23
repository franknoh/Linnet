"""The model examples (Llama, GPT-2, ViT) materialize into PyTorch and agree
with straightforward PyTorch implementations of the same architectures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.torch import LinnetModule, load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"
EXAMPLES = REPO / "examples"


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]


def _with_random_weights(
    source: Path,
    generics: dict[str, int | str],
    tmp_path: Path,
    scale: float = 0.3,
    skip_optional_biases: bool = False,
) -> tuple[LinnetModule, dict[str, torch.Tensor]]:
    """Loads the model twice: once to learn its parameters, once with random
    weights for them, returned by their Linnet paths. `skip_optional_biases`
    leaves the standard library's optional `Linear` biases absent."""
    skeleton = load(source, generics=generics, std_root=STDLIB)
    weights: dict[str, torch.Tensor] = {}
    for name, parameter in skeleton.named_parameters():
        path = name.removeprefix("root.")
        if skip_optional_biases and path.endswith(".bias"):
            continue
        weights[path] = torch.randn(parameter.shape) * scale
    save_file(weights, str(tmp_path / "model.safetensors"))
    return load(source, generics=generics, std_root=STDLIB, weights=tmp_path), weights


def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, [x.shape[-1]], w, b, eps=1e-5)


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, causal: bool
) -> torch.Tensor:
    score = (q @ k.transpose(-1, -2)) * scale
    if causal:
        mask = torch.tril(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool))
        score = torch.where(mask, score, torch.full_like(score, -1e30))
    return torch.softmax(score, dim=-1) @ v


def test_llama_matches_reference(tmp_path: Path) -> None:
    H, heads, kv_heads, inner, layers, vocab = 8, 4, 2, 16, 2, 11  # noqa: N806
    generics: dict[str, int | str] = {
        "Vocab": vocab,
        "H": H,
        "Heads": heads,
        "KvHeads": kv_heads,
        "Inner": inner,
        "Layers": layers,
        "Batch": 2,
        "MaxSeq": 8,
        "T": "f32",
    }
    source = EXAMPLES / "05-llama/src/lib.linnet"
    model, w = _with_random_weights(source, generics, tmp_path, skip_optional_biases=True)
    B, S, D = 2, 5, H // heads  # noqa: N806
    tokens = torch.randint(0, vocab, (B, S), dtype=torch.int32)
    out = model(tokens)
    assert out.shape == (B, S, vocab)
    last = model.run_entry("next_token", [tokens])

    # Rotary tables exactly as the example computes them from iota.
    positions = torch.arange(S, dtype=torch.float32)[:, None]
    inv_freq = torch.exp(
        -(torch.arange(0, D // 2, dtype=torch.float32) * 2.0 / D)
        * torch.log(torch.tensor(500000.0))
    )
    angles = positions * inv_freq
    cos_table = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin_table = torch.cat([angles.sin(), angles.sin()], dim=-1)

    def rope(x: torch.Tensor) -> torch.Tensor:
        first, second = x[..., : D // 2], x[..., D // 2 :]
        return x * cos_table + torch.cat([-second, first], dim=-1) * sin_table

    def split(x: torch.Tensor, n: int) -> torch.Tensor:
        return x.reshape(B, S, n, D).permute(0, 2, 1, 3)

    x = w["embedding.weight"][tokens.long()]
    for i in range(layers):
        p = f"layers.{i}."
        h = rms_norm(x, w[p + "attention_norm.weight"])
        a = p + "attention."
        q = rope(split(h @ w[a + "q_proj.weight"].T, heads))
        k = rope(split(h @ w[a + "k_proj.weight"].T, kv_heads))
        v = split(h @ w[a + "v_proj.weight"].T, kv_heads)
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
        mixed = causal_attention(q, k, v, D**-0.5, causal=True)
        x = x + mixed.permute(0, 2, 1, 3).reshape(B, S, H) @ w[a + "o_proj.weight"].T
        h = rms_norm(x, w[p + "mlp_norm.weight"])
        gate, up = h @ w[p + "mlp.gate.weight"].T, h @ w[p + "mlp.up.weight"].T
        x = x + (torch.nn.functional.silu(gate) * up) @ w[p + "mlp.down.weight"].T
    hidden = rms_norm(x, w["norm.weight"])
    reference = hidden @ w["lm_head.weight"].T
    torch.testing.assert_close(out, reference, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(last, reference[:, -1, :], atol=1e-4, rtol=1e-4)


def test_gpt2_matches_reference(tmp_path: Path) -> None:
    H, heads, layers, vocab, max_positions = 8, 2, 2, 11, 16  # noqa: N806
    generics: dict[str, int | str] = {
        "Vocab": vocab,
        "MaxPositions": max_positions,
        "H": H,
        "Heads": heads,
        "Layers": layers,
        "T": "f32",
    }
    source = EXAMPLES / "06-gpt2/gpt2.linnet"
    model, w = _with_random_weights(source, generics, tmp_path)
    B, S, D = 2, 5, H // heads  # noqa: N806
    tokens = torch.randint(0, vocab, (B, S), dtype=torch.int32)
    out = model(tokens)

    def linear(x: torch.Tensor, prefix: str) -> torch.Tensor:
        # GPT-2's projections are stored as [in, out].
        return x @ w[prefix + ".weight"] + w[prefix + ".bias"]

    x = w["wte"][tokens.long()] + w["wpe"][:S]
    for i in range(layers):
        p = f"blocks.{i}."
        h = layer_norm(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
        qkv = linear(h, p + "attn.qkv")
        q, k, v = (
            qkv[..., j * H : (j + 1) * H].reshape(B, S, heads, D).permute(0, 2, 1, 3)
            for j in range(3)
        )
        mixed = causal_attention(q, k, v, D**-0.5, causal=True)
        x = x + linear(mixed.permute(0, 2, 1, 3).reshape(B, S, H), p + "attn.out")
        h = layer_norm(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
        x = x + linear(
            torch.nn.functional.gelu(linear(h, p + "mlp.up"), approximate="tanh"), p + "mlp.down"
        )
    h = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
    reference = h @ w["wte"].T
    torch.testing.assert_close(out, reference, atol=1e-4, rtol=1e-4)


def test_vit_matches_reference(tmp_path: Path) -> None:
    height, width, channels, patch, D, heads, inner, layers, classes = 8, 8, 3, 4, 8, 2, 16, 2, 5  # noqa: N806
    generics: dict[str, int | str] = {
        "Height": height,
        "Width": width,
        "Channels": channels,
        "Patch": patch,
        "D": D,
        "Heads": heads,
        "Inner": inner,
        "Layers": layers,
        "Classes": classes,
        "T": "f32",
    }
    source = EXAMPLES / "07-vit/vit.linnet"
    model, w = _with_random_weights(source, generics, tmp_path)
    B = 2  # noqa: N806
    image = torch.randn(B, channels, height, width)
    out = model(image)
    assert out.shape == (B, classes)

    def linear(x: torch.Tensor, prefix: str) -> torch.Tensor:
        return x @ w[prefix + ".weight"].T + w[prefix + ".bias"]

    n = (height // patch) * (width // patch)
    patches = (
        image.reshape(B, channels, height // patch, patch, width // patch, patch)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(B, n, channels * patch * patch)
    )
    x = torch.cat([w["class_token"].expand(B, 1, D), linear(patches, "patch_embed")], dim=1)
    x = x + w["positions"]
    dh = D // heads
    for i in range(layers):
        p = f"layers.{i}."
        h = layer_norm(x, w[p + "norm_1.weight"], w[p + "norm_1.bias"])
        q, k, v = (
            linear(h, p + f"attention.{name}").reshape(B, n + 1, heads, dh).permute(0, 2, 1, 3)
            for name in ("q", "k", "v")
        )
        mixed = causal_attention(q, k, v, dh**-0.5, causal=False)
        x = x + linear(mixed.permute(0, 2, 1, 3).reshape(B, n + 1, D), p + "attention.out")
        h = layer_norm(x, w[p + "norm_2.weight"], w[p + "norm_2.bias"])
        x = x + linear(
            torch.nn.functional.gelu(linear(h, p + "up"), approximate="tanh"), p + "down"
        )
    pooled = layer_norm(x, w["norm.weight"], w["norm.bias"])[:, 0, :]
    reference = linear(pooled, "head")
    torch.testing.assert_close(out, reference, atol=1e-4, rtol=1e-4)


def test_clip_matches_reference(tmp_path: Path) -> None:
    height, width, channels, patch = 8, 8, 3, 4
    vocab, context, D, heads, inner, layers, embed = 11, 16, 8, 2, 16, 2, 6  # noqa: N806
    generics: dict[str, int | str] = {
        "Height": height,
        "Width": width,
        "Channels": channels,
        "Patch": patch,
        "Vocab": vocab,
        "Context": context,
        "D": D,
        "Heads": heads,
        "Inner": inner,
        "Layers": layers,
        "Embed": embed,
        "T": "f32",
    }
    source = EXAMPLES / "08-clip/src/lib.linnet"
    model, w = _with_random_weights(source, generics, tmp_path)
    B, C, S = 2, 3, 5  # noqa: N806
    image = torch.randn(B, channels, height, width)
    tokens = torch.randint(0, vocab, (C, S), dtype=torch.int32)
    similarity = model.run_entry("similarity", [image, tokens])
    assert similarity.shape == (B, C)

    def linear(x: torch.Tensor, prefix: str) -> torch.Tensor:
        return x @ w[prefix + ".weight"].T + w[prefix + ".bias"]

    def encoder(x: torch.Tensor, prefix: str, causal: bool) -> torch.Tensor:
        n = x.shape[1]
        for i in range(layers):
            p = f"{prefix}.layers.{i}."
            h = layer_norm(x, w[p + "norm_1.weight"], w[p + "norm_1.bias"])
            qkv = linear(h, p + "qkv")
            q, k, v = (
                qkv[..., j * D : (j + 1) * D].reshape(-1, n, heads, D // heads).permute(0, 2, 1, 3)
                for j in range(3)
            )
            mixed = causal_attention(q, k, v, (D // heads) ** -0.5, causal)
            x = x + linear(mixed.permute(0, 2, 1, 3).reshape(-1, n, D), p + "out")
            h = layer_norm(x, w[p + "norm_2.weight"], w[p + "norm_2.bias"])
            x = x + linear(
                torch.nn.functional.gelu(linear(h, p + "up"), approximate="tanh"), p + "down"
            )
        return x

    def normalize(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-12)

    n = (height // patch) * (width // patch)
    patches = (
        image.reshape(B, channels, height // patch, patch, width // patch, patch)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(B, n, channels * patch * patch)
    )
    v = torch.cat(
        [w["vision.class_token"].expand(B, 1, D), linear(patches, "vision.patch_embed")], 1
    )
    v = layer_norm(
        v + w["vision.positions"], w["vision.pre_norm.weight"], w["vision.pre_norm.bias"]
    )
    v = encoder(v, "vision.encoder", causal=False)[:, 0, :]
    v = linear(
        layer_norm(v, w["vision.post_norm.weight"], w["vision.post_norm.bias"]), "vision.projection"
    )
    t = w["text.token_embedding"][tokens.long()] + w["text.positions"][:S]
    t = encoder(t, "text.encoder", causal=True)[:, S - 1, :]
    t = linear(
        layer_norm(t, w["text.final_norm.weight"], w["text.final_norm.bias"]), "text.projection"
    )
    reference = (normalize(v) @ normalize(t).T) * torch.exp(w["logit_scale"])
    torch.testing.assert_close(similarity, reference, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        model.run_entry("embed_image", [image]), normalize(v), atol=1e-4, rtol=1e-4
    )


def test_llama_decode_matches_full_prefix(tmp_path: Path) -> None:
    """Decoding token by token through the KV caches gives, at every step,
    the logits `next_token` computes over the whole prefix."""
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 4,
        "KvHeads": 2,
        "Inner": 16,
        "Layers": 2,
        "Batch": 2,
        "MaxSeq": 6,
        "T": "f32",
    }
    source = EXAMPLES / "05-llama/src/lib.linnet"
    model, _ = _with_random_weights(source, generics, tmp_path, skip_optional_biases=True)
    assert model.state_paths() == [
        "layers[*].attention.cache_k",
        "layers[*].attention.cache_v",
    ]
    tokens = torch.randint(0, 11, (2, 5), dtype=torch.int32)
    for pos in range(5):
        step = model.run_entry(
            "decode", [tokens[:, pos : pos + 1], torch.tensor(pos, dtype=torch.int32)]
        )
        full = model.run_entry("next_token", [tokens[:, : pos + 1]])
        torch.testing.assert_close(step, full, atol=1e-4, rtol=1e-4)
    # A fresh cache restarts the sequence.
    model.reset_state()
    first = model.run_entry("decode", [tokens[:, :1], torch.tensor(0, dtype=torch.int32)])
    torch.testing.assert_close(first, model.run_entry("next_token", [tokens[:, :1]]))


def test_llama_generate_matches_stepwise_decoding(tmp_path: Path) -> None:
    """`generate` inside the graph produces the tokens a host loop of `decode`
    and argmax produces."""
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 4,
        "KvHeads": 2,
        "Inner": 16,
        "Layers": 2,
        "Batch": 2,
        "MaxSeq": 8,
        "T": "f32",
    }
    source = EXAMPLES / "05-llama/src/lib.linnet"
    model, _ = _with_random_weights(source, generics, tmp_path, skip_optional_biases=True)
    prompt = torch.randint(0, 11, (2, 1), dtype=torch.int32)
    generated = model.run_entry(
        "generate", [prompt, torch.tensor(0, dtype=torch.int32)], generics={"Steps": 4}
    )
    assert generated.shape == (2, 4) and generated.dtype == torch.int32

    model.reset_state()
    current = prompt
    expected: list[torch.Tensor] = []
    for pos in range(4):
        logits = model.run_entry("decode", [current, torch.tensor(pos, dtype=torch.int32)])
        current = logits.argmax(-1, keepdim=True).to(torch.int32)
        expected.append(current)
    torch.testing.assert_close(generated, torch.cat(expected, dim=1))


def test_llama_sample_is_deterministic_and_greedy_at_low_temperature(tmp_path: Path) -> None:
    """`sample` draws with an explicit key: the same key reproduces the tokens,
    and at a tiny temperature it agrees with greedy `generate`."""
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 4,
        "KvHeads": 2,
        "Inner": 16,
        "Layers": 2,
        "Batch": 2,
        "MaxSeq": 8,
        "T": "f32",
    }
    source = EXAMPLES / "05-llama/src/lib.linnet"
    model, _ = _with_random_weights(source, generics, tmp_path, skip_optional_biases=True)
    prompt = torch.randint(0, 11, (2, 1), dtype=torch.int32)
    start = torch.tensor(0, dtype=torch.int32)
    key = torch.tensor([0, 7], dtype=torch.int64)

    def sample(temperature: float) -> torch.Tensor:
        model.reset_state()
        return model.run_entry(
            "sample", [prompt, start, key, torch.tensor(temperature)], generics={"Steps": 4}
        )

    first = sample(1.0)
    assert first.shape == (2, 4) and first.dtype == torch.int32
    torch.testing.assert_close(sample(1.0), first)
    model.reset_state()
    greedy = model.run_entry("generate", [prompt, start], generics={"Steps": 4})
    torch.testing.assert_close(sample(1e-3), greedy)


def test_llama_generate_until_stops_at_eos(tmp_path: Path) -> None:
    """The `while` entry produces what `generate` produces until every row has
    emitted the end token, then stops and reports the count."""
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 4,
        "KvHeads": 2,
        "Inner": 16,
        "Layers": 2,
        "Batch": 2,
        "MaxSeq": 8,
        "T": "f32",
    }
    source = EXAMPLES / "05-llama/src/lib.linnet"
    model, _ = _with_random_weights(source, generics, tmp_path, skip_optional_biases=True)
    prompt = torch.randint(0, 11, (2, 1), dtype=torch.int32)
    start = torch.tensor(0, dtype=torch.int32)
    greedy = model.run_entry("generate", [prompt, start], generics={"Steps": 5})

    # An end token nobody produces: the full budget is used.
    model.reset_state()
    tokens, count = model.run_entry(
        "generate_until",
        [prompt, start, torch.tensor(99, dtype=torch.int32)],
        generics={"MaxNew": 5},
    )
    assert int(count) == 5
    torch.testing.assert_close(tokens, greedy)

    # An end token taken from the greedy tokens: the loop stops after the
    # first step at which every row has produced it.
    eos = greedy[0, 2].to(torch.int32)
    stop = next(s for s in range(5) if bool((greedy[:, s] == eos).all()))
    model.reset_state()
    tokens, count = model.run_entry("generate_until", [prompt, start, eos], generics={"MaxNew": 5})
    assert int(count) == stop + 1
    torch.testing.assert_close(tokens[:, : stop + 1], greedy[:, : stop + 1])
    assert int(tokens[:, stop + 1 :].abs().sum()) == 0
