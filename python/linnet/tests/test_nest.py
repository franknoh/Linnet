"""Nest cards: reading, validation without the Hub, the index, and previews."""

from __future__ import annotations

import fnmatch
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file  # type: ignore[import-untyped]

from linnet import ir, nest

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
    shutil.copy(REPO / "tests/fixtures/gpt2/gpt2.linnet", directory / "gpt2.linnet")
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


def test_tied_parameters_are_counted_once(model_dir: Path) -> None:
    card = nest.Card.read(model_dir)
    program = card.program(STDLIB)
    untied = nest.parameter_count(card, program)
    # Bind a second path to a tensor already bound: the count must not grow.
    (model_dir / "bindings.json").write_text(
        json.dumps({"wte": "shared", "wpe": "shared"}), encoding="utf-8"
    )
    tied = nest.parameter_count(nest.Card.read(model_dir), program)
    assert tied == untied - 16 * 8  # `wpe` no longer counted on its own


def test_a_card_can_state_its_published_parameter_count(model_dir: Path) -> None:
    """Packed 4-bit weights hold two parameters per stored element, so a
    card whose checkpoint packs them states the published count instead."""
    counted = nest.describe(nest.Card.read(model_dir), STDLIB)["parameters"]
    card_path = model_dir / "nest.toml"
    card_path.write_text(
        card_path.read_text(encoding="utf-8").replace(
            "[model]\n", "[model]\nparameters = 20914757184\n", 1
        ),
        encoding="utf-8",
    )
    stated = nest.describe(nest.Card.read(model_dir), STDLIB)["parameters"]
    assert stated == 20914757184 != counted


def test_cli_check_and_index(
    model_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = nest.main(["--std", str(STDLIB), "check", "--no-hub", str(model_dir)])
    assert code == 0 and "gpt2-tiny: ok" in capsys.readouterr().out
    out = tmp_path / "index.json"
    assert nest.main(["--std", str(STDLIB), "index", str(model_dir.parent), "-o", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["models"][0]["name"] == "gpt2-tiny"


def test_hub_headers_come_from_the_files_the_card_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint named the way `diffusers` names one is still readable.

    `huggingface_hub.get_safetensors_metadata` only looks for
    `model.safetensors`, so the header is read directly, with two ranged
    requests and no tensor data.
    """
    header = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            "decoder.conv_in.weight": {
                "dtype": "F32",
                "shape": [512, 4, 3, 3],
                "data_offsets": [0, 24],
            },
        }
    ).encode("utf-8")
    blob = struct.pack("<Q", len(header)) + header + bytes(24)
    asked: list[str] = []

    class Response:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.status_code = 206

        def raise_for_status(self) -> None:
            return None

    class Session:
        def get(self, url: str, headers: dict[str, str], timeout: int) -> Response:
            asked.append(headers["Range"])
            first, last = (int(v) for v in headers["Range"][len("bytes=") :].split("-"))
            return Response(blob[first : last + 1])

    def url(repo: str, filename: str, revision: str | None = None) -> str:
        return f"https://example.invalid/{repo}/{filename}"

    monkeypatch.setattr("huggingface_hub.hf_hub_url", url)
    monkeypatch.setattr("huggingface_hub.utils.get_session", Session)

    parsed = nest.hub_safetensors_header("org/repo", "diffusion_pytorch_model.safetensors")
    assert parsed["decoder.conv_in.weight"]["shape"] == [512, 4, 3, 3]
    assert asked == ["bytes=0-7", f"bytes=8-{7 + len(header)}"]


def test_fetch_falls_back_to_the_cached_copy(tmp_path: Path) -> None:
    """A registry that cannot be reached leaves the cached card in use, with
    a warning; with nothing cached, it is an error."""
    registry = tmp_path / "registry"
    (registry / "models/tiny").mkdir(parents=True)
    (registry / "models/tiny/nest.toml").write_text("name = 'tiny'\n", encoding="utf-8")
    index = {"models": [{"name": "tiny", "files": ["nest.toml"]}]}
    (registry / "index.json").write_text(json.dumps(index), encoding="utf-8")
    cache = tmp_path / "cache"

    fetched = nest.fetch("tiny", registry=registry.as_uri(), cache=cache)
    assert (fetched / "nest.toml").read_text(encoding="utf-8") == "name = 'tiny'\n"
    with pytest.raises(nest.NestError, match="no model"):
        nest.fetch("other", registry=registry.as_uri(), cache=cache)

    gone = (tmp_path / "gone").as_uri()
    with pytest.warns(UserWarning, match="unreachable"):
        assert nest.fetch("tiny", registry=gone, cache=cache) == fetched
    with pytest.raises(nest.NestError, match="cannot fetch"):
        nest.fetch("tiny", registry=gone, cache=tmp_path / "empty")


def write_checkpoint(directory: Path, card: nest.Card) -> dict[str, tuple[int, ...]]:
    """Every parameter of the card's model, under its checkpoint name."""
    program = card.program(STDLIB)
    bindings = ir.bind_generics(program.root.generics, card.generics)
    mapping: dict[str, str] = json.loads((directory / "bindings.json").read_text("utf-8"))
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for entry in program.manifest:
        if entry.kind != "param":
            continue
        shape = ir.evaluate_shape(entry.shape, bindings)
        for path in nest.expand_paths(entry, bindings):
            arrays[mapping.get(path, path)] = np.full(shape, 0.01, dtype=np.float32)
    save_file(arrays, str(directory / "model.safetensors"))
    return {name: array.shape for name, array in arrays.items()}


def offline(*args: object, **kwargs: object) -> str:
    raise AssertionError("the Hub was asked")


def test_a_checkpoint_beside_the_card_needs_no_hub(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory holding its checkpoint is checked and loaded from disk."""
    card = nest.Card.read(model_dir)
    assert nest.local_weights(card) is None
    shapes = write_checkpoint(model_dir, card)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", offline)
    monkeypatch.setattr("huggingface_hub.hf_hub_url", offline)
    assert nest.download_weights(card) == model_dir / "model.safetensors"
    assert nest.check(card, std_root=STDLIB, exports=False) == []

    # Without a Hub repo, the files must be there.
    (model_dir / "nest.toml").write_text(
        CARD.replace('repo = "openai-community/gpt2"\n', ""), encoding="utf-8"
    )
    assert nest.check(nest.Card.read(model_dir), std_root=STDLIB, exports=False) == []
    save_file(
        {"wte.weight": np.zeros((3, 8), dtype=np.float32)}, str(model_dir / "model.safetensors")
    )
    problems = nest.check(nest.Card.read(model_dir), std_root=STDLIB, exports=False)
    assert f"`wte.weight` has shape [3, 8], `wte` needs {list(shapes['wte.weight'])}" in problems
    (model_dir / "model.safetensors").unlink()
    with pytest.raises(nest.NestError, match="holds no checkpoint"):
        nest.download_weights(nest.Card.read(model_dir))
    assert "weights.repo is required unless weights.files are beside the card" in nest.check(
        nest.Card.read(model_dir), std_root=STDLIB, exports=False
    )


def test_a_model_directory_loads_in_torch(model_dir: Path) -> None:
    torch = pytest.importorskip("torch")
    write_checkpoint(model_dir, nest.Card.read(model_dir))
    model = nest.load(model_dir, std_root=STDLIB)
    logits = model(torch.tensor([[1, 2, 3]], dtype=torch.int32))
    assert tuple(logits.shape) == (1, 3, 11)


def test_hub_repos_resolve_by_name(
    model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`org/name` is a Hub repo with the card at its root. Its checkpoint
    comes along when the card names no other repo for it."""
    snapshots = tmp_path / "snapshots"
    asked: list[tuple[str, str | None, tuple[str, ...]]] = []

    def hf_hub_download(repo: str, filename: str, revision: str | None = None) -> str:
        target = snapshots / repo / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_dir / filename, target)
        return str(target)

    def snapshot_download(
        repo: str, revision: str | None = None, allow_patterns: list[str] | None = None
    ) -> str:
        patterns = tuple(allow_patterns or ["*"])
        asked.append((repo, revision, patterns))
        for file in model_dir.rglob("*"):
            relative = file.relative_to(model_dir).as_posix()
            if file.is_file() and any(fnmatch.fnmatch(relative, p) for p in patterns):
                target = snapshots / repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(file, target)
        return str(snapshots / repo)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", hf_hub_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    write_checkpoint(model_dir, nest.Card.read(model_dir))

    # The card names its checkpoint's repo: the weights stay there.
    card = nest.resolve("me/gpt2-tiny")
    assert card.name == "gpt2-tiny" and card.source_path.is_file()
    assert asked[-1][:2] == ("me/gpt2-tiny", None)
    assert not any(fnmatch.fnmatch("model.safetensors", p) for p in asked[-1][2])
    assert nest.local_weights(card) is None

    # The checkpoint is beside the card: it comes with the snapshot.
    (model_dir / "nest.toml").write_text(
        CARD.replace('repo = "openai-community/gpt2"\n', ""), encoding="utf-8"
    )
    card = nest.resolve("hf://me/gpt2-tiny@v1")
    assert asked[-1][:2] == ("me/gpt2-tiny", "v1")
    assert nest.local_weights(card) == card.directory / "model.safetensors"


def test_a_directory_without_a_card_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(nest.NestError, match=r"has no nest\.toml"):
        nest.resolve(tmp_path)
