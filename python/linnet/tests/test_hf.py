"""`linnet.hf.export` writes Transformers checkpoints that `transformers` loads
and that compute what the Linnet program computes."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file  # type: ignore[import-untyped]

from linnet import LinnetError, hf
from linnet.torch import load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"
GPT2 = REPO / "examples/06-gpt2/gpt2.linnet"
LLAMA = REPO / "examples/05-llama/src/lib.linnet"
GPT2_GENERICS: dict[str, int | str] = {
    "Vocab": 11,
    "MaxPositions": 16,
    "H": 8,
    "Heads": 2,
    "Layers": 2,
    "T": "f32",
}
LLAMA_GENERICS: dict[str, int | str] = {
    "Vocab": 11,
    "H": 8,
    "Heads": 2,
    "KvHeads": 1,
    "Inner": 16,
    "Layers": 2,
    "Batch": 1,
    "MaxSeq": 16,
    "T": "f32",
}


def _random_checkpoint(source: Path, generics: dict[str, int | str], path: Path) -> torch.nn.Module:
    torch.manual_seed(0)
    skeleton = load(source, generics=generics, std_root=STDLIB)
    weights = {
        name.removeprefix("root."): torch.randn(parameter.shape) * 0.3
        for name, parameter in skeleton.named_parameters()
    }
    save_file(weights, str(path))
    return load(source, generics=generics, std_root=STDLIB, weights=path)


def test_gpt2_export_matches_transformers(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    reference = _random_checkpoint(GPT2, GPT2_GENERICS, tmp_path / "gpt2.safetensors")
    exported = hf.export(
        GPT2,
        tmp_path / "out",
        generics=GPT2_GENERICS,
        weights=tmp_path / "gpt2.safetensors",
        std_root=STDLIB,
    )
    assert exported.family.name == "gpt2"
    config = json.loads((tmp_path / "out/config.json").read_text())
    assert config["architectures"] == ["GPT2LMHeadModel"] and config["n_embd"] == 8
    names = set(load_file(str(tmp_path / "out/model.safetensors")))
    assert {
        "wte.weight",
        "wpe.weight",
        "h.0.attn.c_attn.weight",
        "h.1.mlp.c_proj.bias",
        "ln_f.bias",
    } <= names

    model = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path / "out", torch_dtype=torch.float32
    )
    model.eval()
    tokens = torch.tensor([[1, 4, 7, 2, 9]], dtype=torch.int32)
    with torch.no_grad():
        expected = model(tokens.long()).logits
    torch.testing.assert_close(reference(tokens), expected, atol=1e-4, rtol=1e-4)


def test_llama_export_matches_transformers(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    skeleton = load(LLAMA, generics=LLAMA_GENERICS, std_root=STDLIB)
    weights = {
        name.removeprefix("root."): torch.randn(parameter.shape) * 0.3
        for name, parameter in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    save_file(weights, str(tmp_path / "llama.safetensors"))
    reference = load(
        LLAMA, generics=LLAMA_GENERICS, std_root=STDLIB, weights=tmp_path / "llama.safetensors"
    )

    exported = hf.export(
        LLAMA,
        tmp_path / "out",
        generics=LLAMA_GENERICS,
        weights=tmp_path / "llama.safetensors",
        std_root=STDLIB,
    )
    assert exported.family.name == "llama" and exported.tensors == 3 + 2 * 9
    config = exported.config
    assert config["rope_theta"] == 500000.0 and config["num_key_value_heads"] == 1
    assert config["attention_bias"] is False and config["torch_dtype"] == "float32"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path / "out", torch_dtype=torch.float32
    )
    model.eval()
    tokens = torch.tensor([[1, 4, 7, 2, 9]], dtype=torch.int32)
    with torch.no_grad():
        expected = model(tokens.long()).logits
    torch.testing.assert_close(reference(tokens), expected, atol=1e-4, rtol=1e-4)


def test_unknown_structures_are_refused(tmp_path: Path) -> None:
    source = REPO / "examples/04-tiny-transformer/src/lib.linnet"
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 2,
        "Inner": 16,
        "Layers": 1,
        "T": "f32",
    }
    skeleton = load(source, generics=generics, std_root=STDLIB)
    save_file(
        {n.removeprefix("root."): torch.zeros(p.shape) for n, p in skeleton.named_parameters()},
        str(tmp_path / "w.safetensors"),
    )
    with pytest.raises(LinnetError, match="not one of the architectures"):
        hf.export(
            source,
            tmp_path / "out",
            generics=generics,
            weights=tmp_path / "w.safetensors",
            std_root=STDLIB,
        )


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _random_checkpoint(GPT2, GPT2_GENERICS, tmp_path / "gpt2.safetensors")
    code = hf.main(
        [
            "export",
            str(GPT2),
            "-o",
            str(tmp_path / "out"),
            "--std",
            str(STDLIB),
            "--weights",
            str(tmp_path / "gpt2.safetensors"),
            *[f"--bind={k}={v}" for k, v in GPT2_GENERICS.items()],
        ]
    )
    assert code == 0 and "GPT2LMHeadModel" in capsys.readouterr().out
