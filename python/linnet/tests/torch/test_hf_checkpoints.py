"""Real checkpoints: the Llama example loaded with TinyLlama's weights gives
the logits `transformers` gives. Downloads about 2 GB, so it runs only with
`LINNET_HF_TESTS=1`."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from linnet.torch import bind_weights, load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

pytestmark = pytest.mark.skipif(
    os.environ.get("LINNET_HF_TESTS") != "1", reason="set LINNET_HF_TESTS=1 to download checkpoints"
)


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


def _tinyllama_bindings(layers: int) -> dict[str, str]:
    bindings = {
        "embedding.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for i in range(layers):
        ours, theirs = f"layers.{i}.", f"model.layers.{i}."
        bindings[ours + "attention_norm.weight"] = theirs + "input_layernorm.weight"
        bindings[ours + "mlp_norm.weight"] = theirs + "post_attention_layernorm.weight"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            bindings[f"{ours}attention.{name}.weight"] = f"{theirs}self_attn.{name}.weight"
        for ours_name, theirs_name in (
            ("gate", "gate_proj"),
            ("up", "up_proj"),
            ("down", "down_proj"),
        ):
            bindings[f"{ours}mlp.{ours_name}.weight"] = f"{theirs}mlp.{theirs_name}.weight"
    return bindings


def test_tinyllama_logits_match_transformers(tmp_path: Path) -> None:
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]
    from safetensors.torch import load_file, save_file  # pyright: ignore[reportUnknownVariableType]
    from transformers import AutoModelForCausalLM  # pyright: ignore[reportUnknownVariableType]

    repo = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    checkpoint = Path(hf_hub_download(repo, "model.safetensors"))

    # The example is Llama 3 (rope base 500000); TinyLlama is Llama 2 (10000).
    package = tmp_path / "llama"
    shutil.copytree(REPO / "examples/05-llama", package)
    rope = package / "src/rope.linnet"
    rope.write_text(rope.read_text().replace("500000.0", f"{config['rope_theta']:.1f}"))

    # Weights in f32 so the comparison is tight; the file is bf16.
    tensors = {
        name: tensor.float().contiguous() for name, tensor in load_file(str(checkpoint)).items()
    }
    weights = tmp_path / "weights"
    weights.mkdir()
    save_file(tensors, str(weights / "model.safetensors"))
    (weights / "bindings.json").write_text(
        json.dumps(_tinyllama_bindings(config["num_hidden_layers"]))
    )

    generics: dict[str, int | str] = {
        "Vocab": config["vocab_size"],
        "H": config["hidden_size"],
        "Heads": config["num_attention_heads"],
        "KvHeads": config["num_key_value_heads"],
        "Inner": config["intermediate_size"],
        "Layers": config["num_hidden_layers"],
        "Batch": 1,
        "MaxSeq": 16,
        "T": "f32",
    }
    model = load(
        package / "src/lib.linnet",
        generics=generics,
        std_root=STDLIB,
        numerics="equivalent",
        compile=True,
    )
    bind_weights(model, weights / "model.safetensors", weights / "bindings.json")

    reference = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        repo, torch_dtype=torch.float32
    )
    reference.eval()
    tokens = torch.tensor([[1, 450, 4996, 17354, 1701, 29916, 432, 17204]], dtype=torch.int32)
    with torch.no_grad():
        expected = reference(tokens.long()).logits
        actual = model(tokens)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-3)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))

    # Decoding token by token through the KV caches predicts the same next tokens.
    model.reset_state()
    step = torch.empty(0)
    for position in range(tokens.shape[1]):
        step = model.run_entry(
            "decode",
            [tokens[:, position : position + 1], torch.tensor(position, dtype=torch.int32)],
        )
    torch.testing.assert_close(step, expected[:, -1, :], atol=2e-2, rtol=1e-3)


def test_gpt2_logits_match_transformers(tmp_path: Path) -> None:
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]
    from transformers import AutoModelForCausalLM  # pyright: ignore[reportUnknownVariableType]

    repo = "openai-community/gpt2"
    config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    checkpoint = Path(hf_hub_download(repo, "model.safetensors"))

    # The example keeps GPT-2's [in, out] projection layout, so the published
    # file binds as it is; only the names differ.
    bindings = {"wte": "wte.weight", "wpe": "wpe.weight"}
    for i in range(config["n_layer"]):
        ours, theirs = f"blocks.{i}.", f"h.{i}."
        for ours_name, theirs_name in (
            ("ln_1", "ln_1"),
            ("ln_2", "ln_2"),
            ("attn.qkv", "attn.c_attn"),
            ("attn.out", "attn.c_proj"),
            ("mlp.up", "mlp.c_fc"),
            ("mlp.down", "mlp.c_proj"),
        ):
            for leaf in ("weight", "bias"):
                bindings[f"{ours}{ours_name}.{leaf}"] = f"{theirs}{theirs_name}.{leaf}"
    (tmp_path / "bindings.json").write_text(json.dumps(bindings))

    generics: dict[str, int | str] = {
        "Vocab": config["vocab_size"],
        "MaxPositions": config["n_positions"],
        "H": config["n_embd"],
        "Heads": config["n_head"],
        "Layers": config["n_layer"],
        "T": "f32",
    }
    model = load(
        REPO / "examples/06-gpt2/gpt2.linnet",
        generics=generics,
        std_root=STDLIB,
        numerics="equivalent",
        compile=True,
    )
    bind_weights(model, checkpoint, tmp_path / "bindings.json")

    reference = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        repo, torch_dtype=torch.float32
    )
    reference.eval()
    tokens = torch.tensor([[464, 3139, 286, 4881, 318, 6342, 13, 383]], dtype=torch.int32)
    with torch.no_grad():
        expected = reference(tokens.long()).logits
        actual = model(tokens)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-3)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))


def test_resnet18_logits_match_transformers(tmp_path: Path) -> None:
    """The convolutional path end to end: `std.nn.conv`, `std.nn.pool`, and
    inference batch normalization against the published ResNet-18."""
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]
    from transformers import (
        AutoModelForImageClassification,  # pyright: ignore[reportUnknownVariableType]
    )

    from linnet import nest

    repo = "microsoft/resnet-18"
    checkpoint = Path(hf_hub_download(repo, "model.safetensors"))
    model = nest.load(
        REPO.parent / "nest/models/resnet-18",
        backend="torch",
        std_root=STDLIB,
        weights=checkpoint,
    )
    reference = cast(
        Any,
        AutoModelForImageClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            repo, torch_dtype=torch.float32
        ),
    )
    reference.eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        expected = cast(torch.Tensor, reference(images).logits)
        actual = cast(torch.Tensor, model(images))
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
