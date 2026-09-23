"""`linnet.gguf.export` drives llama.cpp's converter over the Transformers
export and writes an Ollama Modelfile. The converter is stubbed here."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet import LinnetError, gguf
from linnet.torch import load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"
GPT2 = REPO / "examples/06-gpt2/gpt2.linnet"
GENERICS: dict[str, int | str] = {
    "Vocab": 11,
    "MaxPositions": 16,
    "H": 8,
    "Heads": 2,
    "Layers": 2,
    "T": "f32",
}

# Stands in for convert_hf_to_gguf.py: records its arguments, writes the file.
STUB = """\
import json, sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--outfile") + 1])
out.write_bytes(b"GGUF" + json.dumps(args).encode())
assert Path(args[0], "config.json").exists()
"""


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    skeleton = load(GPT2, generics=GENERICS, std_root=STDLIB)
    save_file(
        {
            n.removeprefix("root."): torch.randn(p.shape) * 0.3
            for n, p in skeleton.named_parameters()
        },
        str(tmp_path / "gpt2.safetensors"),
    )
    return tmp_path / "gpt2.safetensors"


def test_export_runs_the_converter_and_writes_a_modelfile(tmp_path: Path, checkpoint: Path) -> None:
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text(STUB, encoding="utf-8")
    exported = gguf.export(
        GPT2,
        tmp_path / "out",
        converter=converter,
        outtype="q8_0",
        generics=GENERICS,
        weights=checkpoint,
        std_root=STDLIB,
    )
    assert exported.gguf == tmp_path / "out/gpt2.gguf"
    recorded = json.loads(exported.gguf.read_bytes()[4:])
    assert recorded[-2:] == ["--outtype", "q8_0"] and recorded[0].endswith("hf")
    assert exported.modelfile.read_text(encoding="utf-8") == "FROM ./gpt2.gguf\n"
    assert not (tmp_path / "out/hf").exists()  # removed unless kept
    assert exported.checkpoint.family.name == "gpt2"


def test_converter_lookup_and_failures(
    tmp_path: Path, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLAMA_CPP", raising=False)
    monkeypatch.delenv("LLAMA_CPP_CONVERTER", raising=False)
    with pytest.raises(LinnetError, match=r"convert_hf_to_gguf\.py was not found"):
        gguf.find_converter()
    checkout = tmp_path / "llama.cpp"
    checkout.mkdir()
    (checkout / "convert_hf_to_gguf.py").write_text("import sys; sys.exit(3)", encoding="utf-8")
    monkeypatch.setenv("LLAMA_CPP", str(checkout))
    assert gguf.find_converter() == checkout / "convert_hf_to_gguf.py"
    with pytest.raises(LinnetError, match=r"convert_hf_to_gguf\.py failed"):
        gguf.export(GPT2, tmp_path / "out", generics=GENERICS, weights=checkpoint, std_root=STDLIB)
    with pytest.raises(LinnetError, match="outtype"):
        gguf.export(
            GPT2,
            tmp_path / "out",
            outtype="q4",
            generics=GENERICS,
            weights=checkpoint,
            std_root=STDLIB,
        )


def test_chat_templates_translate_known_shapes(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": (
                    "{% for m in messages %}<|user|>{{ m.content }}<|assistant|>{% endfor %}"
                )
            }
        ),
        encoding="utf-8",
    )
    template = gguf.chat_template_of(tmp_path)
    assert template is not None and "<|user|>" in template and "{{ .Prompt }}" in template
    text = gguf.modelfile_text(Path("m.gguf"), template=template, parameters={"temperature": "0.7"})
    assert text.startswith("FROM ./m.gguf\nPARAMETER temperature 0.7\nTEMPLATE ")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ bos_token }}{{ x }}"})
    )
    assert gguf.chat_template_of(tmp_path) is None
    assert gguf.chat_template_of(tmp_path / "missing") is None


def test_cli(tmp_path: Path, checkpoint: Path, capsys: pytest.CaptureFixture[str]) -> None:
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text(STUB, encoding="utf-8")
    code = gguf.main(
        [
            "export",
            str(GPT2),
            "-o",
            str(tmp_path / "out"),
            "--converter",
            str(converter),
            "--std",
            str(STDLIB),
            "--weights",
            str(checkpoint),
            "--keep-checkpoint",
            *[f"--bind={k}={v}" for k, v in GENERICS.items()],
        ]
    )
    assert code == 0 and "ollama create gpt2" in capsys.readouterr().out
    assert (tmp_path / "out/hf/config.json").exists()
    assert os.path.getsize(tmp_path / "out/gpt2.gguf") > 4
