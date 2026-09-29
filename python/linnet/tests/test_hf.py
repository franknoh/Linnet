"""`linnet.hf.export` writes Transformers checkpoints that `transformers` loads
and that compute what the Linnet program computes."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import json
import shutil
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


# Llama 3.1's rope module, as the Nest card writes it: `llama3` scaling of
# each frequency on top of the base, from four module constants.
LLAMA3_ROPE = """\
module llama.rope

pub const THETA: f32 = 500000.0
pub const FACTOR: f32 = 8.0
pub const LOW_FREQ_FACTOR: f32 = 1.0
pub const HIGH_FREQ_FACTOR: f32 = 4.0
pub const ORIGINAL_MAX_POSITION_EMBEDDINGS: f32 = 8192.0
pub const PI: f32 = 3.14159265

pub fn tables<S: Dim, D: Dim, T: Float>(theta: f32) -> (Tensor[S, D; T], Tensor[S, D; T])
where D % 2 == 0 {
    let base_inv_freq[i] = exp(-(cast<f32>(iota<i64>(D / 2)[i]) * 2.0 / cast<f32>(D)) * log(theta))
    let wavelen[i] = (2.0 * PI) / base_inv_freq[i]
    let low_freq_wavelen = ORIGINAL_MAX_POSITION_EMBEDDINGS / LOW_FREQ_FACTOR
    let high_freq_wavelen = ORIGINAL_MAX_POSITION_EMBEDDINGS / HIGH_FREQ_FACTOR
    let smooth[i] =
        (ORIGINAL_MAX_POSITION_EMBEDDINGS / wavelen[i] - LOW_FREQ_FACTOR) /
        (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
    let smoothed[i] = (1.0 - smooth[i]) * (base_inv_freq[i] / FACTOR) + smooth[i] * base_inv_freq[i]
    let inv_freq[i] = select(
        wavelen[i] > low_freq_wavelen,
        base_inv_freq[i] / FACTOR,
        select(wavelen[i] < high_freq_wavelen, base_inv_freq[i], smoothed[i]),
    )
    let angles[s, i] = cast<f32>(iota<i64>(S)[s]) * inv_freq[i]
    let full = concat(angles, angles, axis = -1)
    return (cast<T>(cos(full)), cast<T>(sin(full)))
}
"""


def test_llama3_rope_scaling_reaches_the_config(tmp_path: Path) -> None:
    """A program defining the four `llama3` constants exports `rope_scaling`,
    and transformers then rotates as the program does: without it, a server
    would use the base frequency alone."""
    transformers = pytest.importorskip("transformers")
    package = tmp_path / "llama31"
    shutil.copytree(LLAMA.parent.parent, package)
    (package / "src/rope.linnet").write_text(LLAMA3_ROPE, encoding="utf-8")
    source = package / "src/lib.linnet"
    torch.manual_seed(0)
    skeleton = load(source, generics=LLAMA_GENERICS, std_root=STDLIB)
    weights = {
        name.removeprefix("root."): torch.randn(parameter.shape) * 0.3
        for name, parameter in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    save_file(weights, str(tmp_path / "llama.safetensors"))
    reference = load(
        source, generics=LLAMA_GENERICS, std_root=STDLIB, weights=tmp_path / "llama.safetensors"
    )
    exported = hf.export(
        source,
        tmp_path / "out",
        generics=LLAMA_GENERICS,
        weights=tmp_path / "llama.safetensors",
        std_root=STDLIB,
    )
    assert exported.config["rope_scaling"] == {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    model = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path / "out", torch_dtype=torch.float32
    )
    model.eval()
    tokens = torch.tensor([[1, 4, 7, 2, 9, 3, 5, 8]], dtype=torch.int32)
    with torch.no_grad():
        expected = model(tokens.long()).logits
    torch.testing.assert_close(reference(tokens), expected, atol=1e-4, rtol=1e-4)


def test_a_checkpoint_of_another_dtype_is_refused(tmp_path: Path) -> None:
    """The tensors are copied as they are and `config.json` declares the
    program's dtype, so the two must agree."""
    _random_checkpoint(GPT2, GPT2_GENERICS, tmp_path / "gpt2.safetensors")
    halved = {k: v.half() for k, v in load_file(str(tmp_path / "gpt2.safetensors")).items()}
    save_file(halved, str(tmp_path / "half.safetensors"))
    with pytest.raises(LinnetError, match=r"is F16, .* needs f32"):
        hf.export(
            GPT2,
            tmp_path / "out",
            generics=GPT2_GENERICS,
            weights=tmp_path / "half.safetensors",
            std_root=STDLIB,
        )


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
