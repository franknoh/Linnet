"""transformers checkpoints as Nest model directories, against a fake Hub."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from linnet import convert, nest
from linnet.weights import TensorHeader

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

CONFIG: dict[str, object] = {
    "model_type": "llama",
    "vocab_size": 32,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 24,
    "num_hidden_layers": 2,
    "max_position_embeddings": 64,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "rope_theta": 250000.0,
    "tie_word_embeddings": False,
}

BASE_CARD = """\
[model]
name = "tinyllama-1.1b-chat"
title = "A Llama"
summary = "The Llama example as a card."
license = "MIT"

[links]
huggingface = "https://huggingface.co/org/llama"

[source]
path = "src/lib.linnet"
root = "Model"
entry = "forward"

[generics]
Vocab = 32000

[check]
B = 1
S = 8

[weights]
repo = "org/llama"
files = ["model.safetensors"]
bindings = "bindings.json"
"""

LAYER = {
    "attention_norm.weight": "input_layernorm.weight",
    "mlp_norm.weight": "post_attention_layernorm.weight",
    "attention.q_proj.weight": "self_attn.q_proj.weight",
    "attention.k_proj.weight": "self_attn.k_proj.weight",
    "attention.v_proj.weight": "self_attn.v_proj.weight",
    "attention.o_proj.weight": "self_attn.o_proj.weight",
    "mlp.gate.weight": "mlp.gate_proj.weight",
    "mlp.up.weight": "mlp.up_proj.weight",
    "mlp.down.weight": "mlp.down_proj.weight",
}


def size(config: Mapping[str, object], key: str) -> int:
    value = config[key]
    assert isinstance(value, int)
    return value


def checkpoint(config: Mapping[str, object]) -> dict[str, list[int]]:
    """The tensor shapes a `transformers` Llama checkpoint of `config` holds."""
    h, inner = size(config, "hidden_size"), size(config, "intermediate_size")
    vocab = size(config, "vocab_size")
    kv = size(config, "num_key_value_heads") * h // size(config, "num_attention_heads")
    shapes = {"model.embed_tokens.weight": [vocab, h], "model.norm.weight": [h]}
    if not config["tie_word_embeddings"]:
        shapes["lm_head.weight"] = [vocab, h]
    per_layer = {
        "input_layernorm.weight": [h],
        "post_attention_layernorm.weight": [h],
        "self_attn.q_proj.weight": [h, h],
        "self_attn.k_proj.weight": [kv, h],
        "self_attn.v_proj.weight": [kv, h],
        "self_attn.o_proj.weight": [h, h],
        "mlp.gate_proj.weight": [inner, h],
        "mlp.up_proj.weight": [inner, h],
        "mlp.down_proj.weight": [h, inner],
    }
    for i in range(size(config, "num_hidden_layers")):
        shapes |= {f"model.layers.{i}.{name}": shape for name, shape in per_layer.items()}
    return shapes


@dataclass
class Hub:
    """A Hub holding one `transformers` repo, `org/tiny`."""

    root: Path
    config: dict[str, object]
    extra: dict[str, list[int]] = field(default_factory=lambda: dict[str, list[int]]())

    def headers(self) -> dict[str, TensorHeader]:
        tensors = checkpoint(self.config) | self.extra
        return {
            k: {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]} for k, v in tensors.items()
        }


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Hub:
    fake = Hub(tmp_path / "hub", dict(CONFIG))
    base = tmp_path / "registry" / "tinyllama-1.1b-chat"
    shutil.copytree(REPO / "examples/01-llama/src", base / "src")
    shutil.copy(REPO / "examples/01-llama/linnet.toml", base / "linnet.toml")
    (base / "nest.toml").write_text(BASE_CARD, encoding="utf-8")
    bindings = {
        "embedding.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for i in range(2):
        bindings |= {f"layers.{i}.{k}": f"model.layers.{i}.{v}" for k, v in LAYER.items()}
    (base / "bindings.json").write_text(json.dumps(bindings), encoding="utf-8")

    @dataclass
    class Sibling:
        rfilename: str

    @dataclass
    class CardData:
        license: str = "apache-2.0"

    @dataclass
    class Info:
        sha: str = "0123abc"
        siblings: list[Sibling] = field(
            default_factory=lambda: [Sibling("config.json"), Sibling("model.safetensors")]
        )
        card_data: CardData = field(default_factory=CardData)

    class HfApi:
        def model_info(self, repo: str, revision: str | None = None) -> Info:
            assert repo == "org/tiny"
            return Info()

    def hf_hub_download(repo: str, filename: str, revision: str | None = None) -> str:
        if filename != "config.json":
            from huggingface_hub.errors import EntryNotFoundError

            raise EntryNotFoundError(filename)
        path = fake.root / repo / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fake.config), encoding="utf-8")
        return str(path)

    def header(repo: str, filename: str, revision: str | None = None) -> dict[str, TensorHeader]:
        assert (repo, filename, revision) == ("org/tiny", "model.safetensors", "0123abc")
        return fake.headers()

    def fetch(name: str, **options: object) -> Path:
        assert name == "tinyllama-1.1b-chat"
        return base

    monkeypatch.setattr("huggingface_hub.HfApi", HfApi)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hf_hub_download)
    monkeypatch.setattr(nest, "hub_safetensors_header", header)
    monkeypatch.setattr(nest, "fetch", fetch)
    monkeypatch.setenv("LINNET_HOME", str(tmp_path / "home"))
    return fake


def test_a_llama_checkpoint_becomes_a_card(hub: Hub, tmp_path: Path) -> None:
    directory = convert.convert("org/tiny", output=tmp_path / "tiny", std_root=STDLIB)
    card = nest.Card.read(directory)
    assert card.name == "tiny"
    assert card.generics == {
        "Vocab": 32,
        "H": 16,
        "Heads": 4,
        "KvHeads": 2,
        "Inner": 24,
        "Layers": 2,
        "Batch": 1,
        "MaxSeq": 64,
        "T": "bf16",
    }
    assert card.weights is not None
    assert (card.weights.repo, card.weights.revision) == ("org/tiny", "0123abc")
    assert card.weights.files == ("model.safetensors",) and card.license == "apache-2.0"
    assert "pub const THETA: f32 = 250000.0" in (directory / "src/rope.linnet").read_text("utf-8")
    bindings = json.loads((directory / "bindings.json").read_text("utf-8"))
    assert bindings["layers.1.mlp.down.weight"] == "model.layers.1.mlp.down_proj.weight"
    assert bindings["lm_head.weight"] == "lm_head.weight"
    assert nest.check(card, std_root=STDLIB, exports=False) == []


def test_a_tied_head_binds_the_embedding(hub: Hub, tmp_path: Path) -> None:
    hub.config["tie_word_embeddings"] = True
    directory = convert.convert("org/tiny", output=tmp_path / "out", std_root=STDLIB)
    bindings = json.loads((directory / "bindings.json").read_text("utf-8"))
    assert bindings["lm_head.weight"] == "model.embed_tokens.weight"


def test_what_the_source_cannot_express_is_refused(hub: Hub, tmp_path: Path) -> None:
    hub.config |= {"hidden_act": "gelu", "rms_norm_eps": 1e-6}
    with pytest.raises(nest.NestError) as error:
        convert.convert("org/tiny", output=tmp_path / "out", std_root=STDLIB)
    assert "`hidden_act` is 'gelu'" in str(error.value)
    assert "`rms_norm_eps` is 1e-06" in str(error.value)

    hub.config = dict(CONFIG) | {"model_type": "falcon"}
    with pytest.raises(nest.NestError, match="`falcon` model; Linnet converts"):
        convert.convert("org/tiny", output=tmp_path / "out", std_root=STDLIB)


def test_a_tied_head_the_checkpoint_stores_binds_its_own(hub: Hub, tmp_path: Path) -> None:
    hub.config["tie_word_embeddings"] = True
    hub.extra = {"lm_head.weight": [32, 16]}
    directory = convert.convert("org/tiny", output=tmp_path / "out", std_root=STDLIB)
    bindings = json.loads((directory / "bindings.json").read_text("utf-8"))
    assert bindings["lm_head.weight"] == "lm_head.weight"


def test_biases_beside_their_weights_bind(hub: Hub, tmp_path: Path) -> None:
    hub.config["attention_bias"] = True
    hub.extra = {
        f"model.layers.{i}.self_attn.{p}_proj.bias": [16 if p == "q" else 8]
        for i in range(2)
        for p in "qkv"
    }
    directory = convert.convert("org/tiny", output=tmp_path / "out", std_root=STDLIB)
    bindings = json.loads((directory / "bindings.json").read_text("utf-8"))
    assert bindings["layers.1.attention.k_proj.bias"] == "model.layers.1.self_attn.k_proj.bias"
    assert "layers.0.attention.o_proj.bias" not in bindings


def test_a_tensor_nothing_binds_is_refused(hub: Hub, tmp_path: Path) -> None:
    hub.extra = {"model.layers.0.self_attn.q_norm.weight": [4]}
    with pytest.raises(nest.NestError, match=r"binds no parameter to `model\.layers\.0"):
        convert.convert("org/tiny", output=tmp_path / "out", std_root=STDLIB)
    assert not (tmp_path / "out").exists()


def test_a_hub_repo_without_a_card_is_converted(hub: Hub) -> None:
    card = nest.resolve("org/tiny")
    assert card.name == "org--tiny" and card.directory.parent.name == "0123abc"
    assert nest.check(card, std_root=STDLIB, exports=False) == []
    assert nest.resolve("hf://org/tiny").directory == card.directory
