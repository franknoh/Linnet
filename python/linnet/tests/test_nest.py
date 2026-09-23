"""Nest cards: reading, validation without the Hub, the index, and previews."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from linnet import nest

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

CARD = """\
[model]
name = "gpt2-tiny"
title = "GPT-2, tiny"
summary = "The GPT-2 example at a toy size."
license = "MIT"
family = "gpt2"
tags = ["test"]

[links]
huggingface = "https://huggingface.co/openai-community/gpt2"

[source]
path = "gpt2.linnet"
root = "Model"
entry = "forward"

[generics]
Vocab = 11
MaxPositions = 16
H = 8
Heads = 2
Layers = 2
T = "f32"

[check]
B = 1
S = 4

[weights]
repo = "openai-community/gpt2"
files = ["model.safetensors"]
bindings = "bindings.json"
"""


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "gpt2-tiny"
    directory.mkdir()
    shutil.copy(REPO / "examples/06-gpt2/gpt2.linnet", directory / "gpt2.linnet")
    (directory / "nest.toml").write_text(CARD, encoding="utf-8")
    (directory / "bindings.json").write_text(json.dumps({"wte": "wte.weight"}), encoding="utf-8")
    (directory / "README.md").write_text(
        "# GPT-2, tiny\n\n" + "A toy card. " * 30, encoding="utf-8"
    )
    return directory


def test_card_reads_every_table(model_dir: Path) -> None:
    card = nest.Card.read(model_dir)
    assert card.name == "gpt2-tiny" and card.family == "gpt2" and card.tags == ("test",)
    assert card.links.huggingface is not None and card.links.arxiv is None
    assert card.generics["H"] == 8 and card.generics["T"] == "f32"
    assert card.check == {"B": 1, "S": 4}
    assert card.weights is not None and card.weights.files == ("model.safetensors",)
    assert card.bindings_path == model_dir / "bindings.json"
    program = card.program(STDLIB)
    assert program.root.name == "Model" and program.entry("forward").short_name == "forward"


def test_check_passes_offline_and_reports_problems(model_dir: Path) -> None:
    card = nest.Card.read(model_dir)
    assert nest.check(card, std_root=STDLIB, hub=False) == []

    (model_dir / "README.md").write_text("short", encoding="utf-8")
    broken = CARD.replace('license = "MIT"\n', "").replace("Layers = 2\n", "")
    (model_dir / "nest.toml").write_text(broken, encoding="utf-8")
    problems = nest.check(nest.Card.read(model_dir), std_root=STDLIB, hub=False, exports=False)
    assert any("README" in p for p in problems)
    assert any("model.license" in p for p in problems)
    assert any("`Layers` needs a value" in p for p in problems)

    (model_dir / "README.md").write_text(
        "# GPT-2, tiny\n\n" + "A toy card. " * 30, encoding="utf-8"
    )
    (model_dir / "nest.toml").write_text(CARD.replace("S = 4\n", ""), encoding="utf-8")
    problems = nest.check(nest.Card.read(model_dir), std_root=STDLIB, hub=False)
    assert problems == ["[check] must bind the entry generics S to export `forward`"]


def test_index_and_preview(model_dir: Path) -> None:
    document = nest.index(model_dir.parent, STDLIB)
    assert document["version"] == 1 and len(document["models"]) == 1
    model = document["models"][0]
    assert model["name"] == "gpt2-tiny" and model["root"] == "Model"
    assert model["parameters"] == nest.ir.parameter_count(
        nest.Card.read(model_dir).program(STDLIB),
        {"Vocab": 11, "MaxPositions": 16, "H": 8, "Heads": 2, "Layers": 2, "T": "f32"},
    )
    # wte 11*8, wpe 16*8, ln_f 16; per block: two norms 32, qkv 8*24+24,
    # out 8*8+8, up 8*32+32, down 32*8+8.
    assert model["parameters"] == 88 + 128 + 16 + 2 * (32 + 216 + 72 + 288 + 264)
    assert model["entries"][0]["signature"].startswith("pub entry forward<B: Dim, S: Dim>")
    assert set(model["files"]) == {"README.md", "bindings.json", "gpt2.linnet", "nest.toml"}
    assert nest.format_parameters(model["parameters"]) == "2K"
    assert nest.format_parameters(1_100_048_384) == "1.1B"

    svg = nest.preview(nest.Card.read(model_dir), std_root=STDLIB)
    assert svg.startswith("<svg") and "for layer in blocks" in svg


def test_cli_check_and_index(
    model_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = nest.main(["--std", str(STDLIB), "check", "--no-hub", str(model_dir)])
    assert code == 0 and "gpt2-tiny: ok" in capsys.readouterr().out
    out = tmp_path / "index.json"
    assert nest.main(["--std", str(STDLIB), "index", str(model_dir.parent), "-o", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["models"][0]["name"] == "gpt2-tiny"
