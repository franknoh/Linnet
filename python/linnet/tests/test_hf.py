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
GPT2 = REPO / "tests/fixtures/gpt2/gpt2.linnet"
LLAMA = REPO / "examples/01-llama/src/lib.linnet"
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


def _random_checkpoint(
    source: Path, generics: dict[str, int | str], path: Path, *, biases: bool = True
) -> torch.nn.Module:
    """Random weights for every parameter (optional biases only with
    `biases`), and the model loaded with them."""
    torch.manual_seed(0)
    skeleton = load(source, generics=generics, std_root=STDLIB)
    weights = {
        name.removeprefix("root."): torch.randn(parameter.shape) * 0.3
        for name, parameter in skeleton.named_parameters()
        if biases or not name.endswith(".bias")
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


def _matches_transformers(reference: torch.nn.Module, directory: Path) -> None:
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForCausalLM.from_pretrained(directory, torch_dtype=torch.float32)
    model.eval()
    tokens = torch.tensor([[1, 4, 7, 2, 9, 3, 5, 8]], dtype=torch.int32)
    with torch.no_grad():
        expected = model(tokens.long()).logits
    torch.testing.assert_close(reference(tokens), expected, atol=1e-4, rtol=1e-4)


def test_qwen2_export_has_biases_on_query_key_and_value_only(tmp_path: Path) -> None:
    """Llama's layout with biases on q, k, and v alone is Qwen2: Llama's
    `attention_bias` would give `o_proj` one too. A head bound to the
    embedding's tensor is written once and tied."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    skeleton = load(LLAMA, generics=LLAMA_GENERICS, std_root=STDLIB)
    qkv = (".q_proj.bias", ".k_proj.bias", ".v_proj.bias")
    weights = {
        name.removeprefix("root."): torch.randn(parameter.shape) * 0.3
        for name, parameter in skeleton.named_parameters()
        if (not name.endswith(".bias") or name.endswith(qkv)) and name != "root.lm_head.weight"
    }
    save_file(weights, str(tmp_path / "qwen2.safetensors"))
    bindings = tmp_path / "bindings.json"
    bindings.write_text(json.dumps({"lm_head.weight": "embedding.weight"}), encoding="utf-8")
    reference = load(
        LLAMA,
        generics=LLAMA_GENERICS,
        std_root=STDLIB,
        weights=tmp_path / "qwen2.safetensors",
        bindings=bindings,
    )
    exported = hf.export(
        LLAMA,
        tmp_path / "out",
        generics=LLAMA_GENERICS,
        weights=tmp_path / "qwen2.safetensors",
        bindings=bindings,
        std_root=STDLIB,
    )
    assert exported.family.name == "qwen2"
    assert exported.config["tie_word_embeddings"] is True
    names = set(load_file(str(tmp_path / "out/model.safetensors")))
    assert "lm_head.weight" not in names and "model.layers.1.self_attn.v_proj.bias" in names
    _matches_transformers(reference, tmp_path / "out")


def test_partial_llama_biases_are_refused(tmp_path: Path) -> None:
    """A bias on `o_proj` alone is neither Llama's `attention_bias` (all
    four projections) nor Qwen2's (q, k, and v)."""
    torch.manual_seed(0)
    skeleton = load(LLAMA, generics=LLAMA_GENERICS, std_root=STDLIB)
    save_file(
        {
            name.removeprefix("root."): torch.zeros(parameter.shape)
            for name, parameter in skeleton.named_parameters()
            if not name.endswith(".bias") or name.endswith(".o_proj.bias")
        },
        str(tmp_path / "w.safetensors"),
    )
    with pytest.raises(LinnetError, match="llama: only some of"):
        hf.export(
            LLAMA,
            tmp_path / "out",
            generics=LLAMA_GENERICS,
            weights=tmp_path / "w.safetensors",
            std_root=STDLIB,
        )


# Qwen3's decoder as the Nest card writes it, in one module: an RMSNorm
# over each query and key head before the rotation, a head width of its
# own, and a tighter epsilon than the stdlib `RmsNorm` block's.
QWEN3 = """\
module qwen3

use std.nn.attention::{causal_mask, grouped_attention}
use std.nn.embedding::{Embedding}
use std.nn.linear::{Linear}
use std.nn.mlp::{SwiGlu}
use std.nn.norm::{rms_norm}
use std.nn.rope::{rope}

pub const THETA: f32 = 1000000.0

fn tables<S: Dim, D: Dim, T: Float>(theta: f32) -> (Tensor[S, D; T], Tensor[S, D; T])
where D % 2 == 0 {
    let inv_freq[i] = exp(-(cast<f32>(iota<i64>(D / 2)[i]) * 2.0 / cast<f32>(D)) * log(theta))
    let angles[s, i] = cast<f32>(iota<i64>(S)[s]) * inv_freq[i]
    let full = concat(angles, angles, axis = -1)
    return (cast<T>(cos(full)), cast<T>(sin(full)))
}

pub block Norm<H: Dim, T: Float> {
    param weight: Tensor[H; T]

    pub fn forward<*S: Shape>(x: Tensor[*S, H; T]) -> Tensor[*S, H; T] {
        return rms_norm(x, weight, 1e-6)
    }
}

pub block Attention<H: Dim, Heads: Dim, KvHeads: Dim, HeadDim: Dim, T: Float>
where
    KvHeads > 0,
    Heads % KvHeads == 0,
    HeadDim % 2 == 0
{
    sub q_proj: Linear<H, Heads * HeadDim, T>
    sub q_norm: Norm<HeadDim, T>
    sub k_proj: Linear<H, KvHeads * HeadDim, T>
    sub k_norm: Norm<HeadDim, T>
    sub v_proj: Linear<H, KvHeads * HeadDim, T>
    sub o_proj: Linear<Heads * HeadDim, H, T>

    pub fn forward<B: Dim, S: Dim>(x: Tensor[B, S, H; T]) -> Tensor[B, S, H; T] {
        let (cos_table, sin_table) = tables<S, HeadDim, T>(THETA)
        let q = q_norm.forward(reshape(q_proj.forward(x), [B, S, Heads, HeadDim]))
        let k = k_norm.forward(reshape(k_proj.forward(x), [B, S, KvHeads, HeadDim]))
        let v = reshape(v_proj.forward(x), [B, S, KvHeads, HeadDim])
        let mixed = grouped_attention(
            rope(permute(q, [0, 2, 1, 3]), cos_table, sin_table),
            rope(permute(k, [0, 2, 1, 3]), cos_table, sin_table),
            permute(v, [0, 2, 1, 3]),
            rsqrt(cast<f32>(HeadDim)),
            some(causal_mask<S, S>()),
        )
        return o_proj.forward(reshape(permute(mixed, [0, 2, 1, 3]), [B, S, Heads * HeadDim]))
    }
}

pub block Layer<H: Dim, Heads: Dim, KvHeads: Dim, HeadDim: Dim, Inner: Dim, T: Float>
where
    KvHeads > 0,
    Heads % KvHeads == 0,
    HeadDim % 2 == 0
{
    sub attention_norm: Norm<H, T>
    sub attention: Attention<H, Heads, KvHeads, HeadDim, T>
    sub mlp_norm: Norm<H, T>
    sub mlp: SwiGlu<H, Inner, T>

    pub fn forward<B: Dim, S: Dim>(x: Tensor[B, S, H; T]) -> Tensor[B, S, H; T] {
        let attended = x + attention.forward(attention_norm.forward(x))
        return attended + mlp.forward(mlp_norm.forward(attended))
    }
}

pub block Model<
    Vocab: Dim,
    H: Dim,
    Heads: Dim,
    KvHeads: Dim,
    HeadDim: Dim,
    Inner: Dim,
    Layers: Dim,
    MaxSeq: Dim,
    T: Float = bf16,
>
where
    KvHeads > 0,
    Heads % KvHeads == 0,
    HeadDim % 2 == 0
{
    sub embedding: Embedding<Vocab, H, T>
    sub layers: [Layer<H, Heads, KvHeads, HeadDim, Inner, T>; Layers]
    sub norm: Norm<H, T>
    sub lm_head: Linear<H, Vocab, T>

    pub entry forward<B: Dim, S: Dim>(tokens: Tensor[B, S; i32]) -> Tensor[B, S, Vocab; T]
    where S <= MaxSeq {
        var x = embedding.forward(tokens)
        static for layer in layers {
            x = layer.forward(x)
        }
        return lm_head.forward(norm.forward(x))
    }
}
"""

# Phi-3's decoder as the Nest card writes it: one projection for the query,
# key, and value and one for the gate and the up projection, split along
# their outputs; as many key/value heads as query heads.
PHI3 = """\
module phi3

use std.nn.activations::{silu}
use std.nn.attention::{attention, causal_mask}
use std.nn.embedding::{Embedding}
use std.nn.linear::{Linear}
use std.nn.norm::{RmsNorm}
use std.nn.rope::{rope}

pub const THETA: f32 = 10000.0

fn tables<S: Dim, D: Dim, T: Float>(theta: f32) -> (Tensor[S, D; T], Tensor[S, D; T])
where D % 2 == 0 {
    let inv_freq[i] = exp(-(cast<f32>(iota<i64>(D / 2)[i]) * 2.0 / cast<f32>(D)) * log(theta))
    let angles[s, i] = cast<f32>(iota<i64>(S)[s]) * inv_freq[i]
    let full = concat(angles, angles, axis = -1)
    return (cast<T>(cos(full)), cast<T>(sin(full)))
}

fn heads<B: Dim, S: Dim, N: Dim, D: Dim, T: Float>(
    x: Tensor[B, S, N * D; T],
) -> Tensor[B, N, S, D; T] {
    return permute(reshape(x, [B, S, N, D]), [0, 2, 1, 3])
}

pub block Attention<H: Dim, Heads: Dim, T: Float>
where
    Heads > 0,
    H % Heads == 0,
    (H / Heads) % 2 == 0
{
    sub qkv: Linear<H, 3 * H, T>
    sub o_proj: Linear<H, H, T>

    pub fn forward<B: Dim, S: Dim>(x: Tensor[B, S, H; T]) -> Tensor[B, S, H; T] {
        let (cos_table, sin_table) = tables<S, H / Heads, T>(THETA)
        let projected = qkv.forward(x)
        let q = heads<B, S, Heads, H / Heads, T>(projected[:, :, 0:H])
        let k = heads<B, S, Heads, H / Heads, T>(projected[:, :, H:2 * H])
        let v = heads<B, S, Heads, H / Heads, T>(projected[:, :, 2 * H:3 * H])
        let mixed = attention(
            rope(q, cos_table, sin_table),
            rope(k, cos_table, sin_table),
            v,
            rsqrt(cast<f32>(H / Heads)),
            some(causal_mask<S, S>()),
        )
        return o_proj.forward(reshape(permute(mixed, [0, 2, 1, 3]), [B, S, H]))
    }
}

pub block FusedSwiGlu<H: Dim, Inner: Dim, T: Float> {
    sub gate_up: Linear<H, 2 * Inner, T>
    sub down: Linear<Inner, H, T>

    pub fn forward<*S: Shape>(x: Tensor[*S, H; T]) -> Tensor[*S, H; T] {
        let projected = gate_up.forward(x)
        return down.forward(silu(projected[..., 0:Inner]) * projected[..., Inner:2 * Inner])
    }
}

pub block Layer<H: Dim, Heads: Dim, Inner: Dim, T: Float>
where
    Heads > 0,
    H % Heads == 0,
    (H / Heads) % 2 == 0
{
    sub attention_norm: RmsNorm<H, T>
    sub attention: Attention<H, Heads, T>
    sub mlp_norm: RmsNorm<H, T>
    sub mlp: FusedSwiGlu<H, Inner, T>

    pub fn forward<B: Dim, S: Dim>(x: Tensor[B, S, H; T]) -> Tensor[B, S, H; T] {
        let attended = x + attention.forward(attention_norm.forward(x))
        return attended + mlp.forward(mlp_norm.forward(attended))
    }
}

pub block Model<
    Vocab: Dim,
    H: Dim,
    Heads: Dim,
    Inner: Dim,
    Layers: Dim,
    MaxSeq: Dim,
    T: Float = bf16,
>
where
    Heads > 0,
    H % Heads == 0,
    (H / Heads) % 2 == 0
{
    sub embedding: Embedding<Vocab, H, T>
    sub layers: [Layer<H, Heads, Inner, T>; Layers]
    sub norm: RmsNorm<H, T>
    sub lm_head: Linear<H, Vocab, T>

    pub entry forward<B: Dim, S: Dim>(tokens: Tensor[B, S; i32]) -> Tensor[B, S, Vocab; T]
    where S <= MaxSeq {
        var x = embedding.forward(tokens)
        static for layer in layers {
            x = layer.forward(x)
        }
        return lm_head.forward(norm.forward(x))
    }
}
"""


def test_qwen3_export_matches_transformers(tmp_path: Path) -> None:
    """Per-head query and key norms and a head width that is not
    `H / Heads`; the epsilon `config.json` declares is the one the norms
    pass, not the stdlib default."""
    pytest.importorskip("transformers")
    source = tmp_path / "qwen3.linnet"
    source.write_text(QWEN3, encoding="utf-8")
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 2,
        "KvHeads": 1,
        "HeadDim": 6,
        "Inner": 16,
        "Layers": 2,
        "MaxSeq": 16,
        "T": "f32",
    }
    reference = _random_checkpoint(source, generics, tmp_path / "qwen3.safetensors", biases=False)
    exported = hf.export(
        source,
        tmp_path / "out",
        generics=generics,
        weights=tmp_path / "qwen3.safetensors",
        std_root=STDLIB,
    )
    assert exported.family.name == "qwen3"
    assert exported.config["head_dim"] == 6 and exported.config["rms_norm_eps"] == 1e-6
    assert exported.config["tie_word_embeddings"] is False
    _matches_transformers(reference, tmp_path / "out")


def test_phi3_export_matches_transformers(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    source = tmp_path / "phi3.linnet"
    source.write_text(PHI3, encoding="utf-8")
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 2,
        "Inner": 16,
        "Layers": 2,
        "MaxSeq": 16,
        "T": "f32",
    }
    reference = _random_checkpoint(source, generics, tmp_path / "phi3.safetensors", biases=False)
    exported = hf.export(
        source,
        tmp_path / "out",
        generics=generics,
        weights=tmp_path / "phi3.safetensors",
        std_root=STDLIB,
    )
    assert exported.family.name == "phi3"
    assert exported.config["num_key_value_heads"] == 2 and exported.config["rms_norm_eps"] == 1e-5
    names = set(load_file(str(tmp_path / "out/model.safetensors")))
    assert {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.1.mlp.gate_up_proj.weight",
    } <= names
    _matches_transformers(reference, tmp_path / "out")


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
    source = REPO / "tests/fixtures/tiny-transformer/src/lib.linnet"
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
