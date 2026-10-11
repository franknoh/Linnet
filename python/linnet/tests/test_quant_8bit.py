"""8-bit weights with their scales per row, per group and per block
(`std.quant::Int8Linear`, `Int8GroupLinear`, `Fp8BlockLinear`): every
backend multiplies by the dequantized weight, and PyTorch on a GPU runs
Linnet's kernels -- the weight as it is, the input unrounded, or under
`numerics="fast"` rounded to 8 bits too."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.torch import load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.eight

use std.quant::{Fp8BlockLinear, Int8GroupLinear, Int8Linear}

pub block Model<In: Dim, Out: Dim, T: Float = f32>
where
    In % 128 == 0,
    Out % 128 == 0
{
    sub rows: Int8Linear<In, Out, T>
    sub groups: Int8GroupLinear<In, Out, 128, T>
    sub blocks: Fp8BlockLinear<In, Out, 128, T>

    pub entry by_rows<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        return rows.forward(x)
    }

    pub entry by_groups<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        return groups.forward(x)
    }

    pub entry by_blocks<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        return blocks.forward(x)
    }
}
"""

# Two layers of each kind reading one input: one product each.
SIBLINGS = """\
module tests.eight_siblings

use std.quant::{Fp8BlockLinear, Int8GroupLinear, Int8Linear}

pub block Model<In: Dim, Out: Dim, T: Float = f32>
where
    In % 128 == 0,
    Out % 128 == 0
{
    sub a: Int8Linear<In, Out, T>
    sub b: Int8Linear<In, Out, T>
    sub c: Int8GroupLinear<In, Out, 128, T>
    sub d: Int8GroupLinear<In, Out, 128, T>
    sub e: Fp8BlockLinear<In, Out, 128, T>
    sub f: Fp8BlockLinear<In, Out, 128, T>

    pub entry forward<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        let rows = a.forward(x) * b.forward(x)
        let groups = c.forward(x) * d.forward(x)
        return rows + groups + e.forward(x) * f.forward(x)
    }
}
"""

IN, OUT = 512, 256
ENTRIES = {"rows": "by_rows", "groups": "by_groups", "blocks": "by_blocks"}


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


def _quantize(
    full: torch.Tensor, kind: str, scale_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, NDArray[np.float32]]:
    """`full` [Out, In] as a layer of `kind` stores it: the 8-bit weight,
    its scales, and the weight they stand for."""
    out_features, in_features = full.shape
    if kind == "rows":
        scale = (full.abs().amax(1, keepdim=True) / 127.0).to(scale_dtype)
        wide = scale.float()
    elif kind == "groups":
        grouped = full.reshape(out_features, in_features // 128, 128)
        scale = (grouped.abs().amax(2) / 127.0).to(scale_dtype)
        wide = scale.float().repeat_interleave(128, 1)
    else:
        blocked = full.reshape(out_features // 128, 128, in_features // 128, 128)
        scale = (blocked.abs().amax((1, 3)) / 448.0).to(scale_dtype)
        wide = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    if kind == "blocks":
        q = (full / wide).to(torch.float8_e4m3fn)
        return q.view(torch.uint8), scale, (q.float() * wide).numpy()
    q = (full / wide).round().clamp(-127, 127).to(torch.int8)
    return q, scale, (q.float() * wide).numpy()


def _checkpoint(
    path: Path, layers: tuple[str, ...], kinds: tuple[str, ...], scale_dtype: torch.dtype, seed: int
) -> dict[str, NDArray[np.float32]]:
    """A checkpoint of each layer under its own parameter paths; the
    dequantized weights by layer."""
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    dequantized: dict[str, NDArray[np.float32]] = {}
    for layer, kind in zip(layers, kinds, strict=True):
        full = torch.randn(OUT, IN, generator=generator)
        weight, scale, dequantized[layer] = _quantize(full, kind, scale_dtype)
        tensors[f"{layer}.weight"] = weight.contiguous()
        tensors[f"{layer}.scale"] = scale.contiguous()
    save_file(tensors, str(path))
    return dequantized


@pytest.fixture
def model(tmp_path: Path) -> tuple[Path, Path, dict[str, NDArray[np.float32]]]:
    source = tmp_path / "eight.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    weights = tmp_path / "eight.safetensors"
    kinds = ("rows", "groups", "blocks")
    return source, weights, _checkpoint(weights, kinds, kinds, torch.float32, 0)


def _input(rows: int = 3) -> NDArray[np.float32]:
    return np.random.default_rng(1).standard_normal((rows, IN)).astype(np.float32)


@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize("kind", list(ENTRIES))
def test_torch_multiplies_dequantized(
    model: tuple[Path, Path, dict[str, NDArray[np.float32]]], kind: str, compile: bool
) -> None:
    source, weights, dequantized = model
    loaded = load(
        source,
        generics={"In": IN, "Out": OUT},
        std_root=STDLIB,
        weights=weights,
        compile=compile,
    )
    got = loaded.run_entry(ENTRIES[kind], [torch.from_numpy(_input())]).detach().numpy()
    np.testing.assert_allclose(got, _input() @ dequantized[kind].T, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("kind", list(ENTRIES))
def test_jax_multiplies_dequantized(
    model: tuple[Path, Path, dict[str, NDArray[np.float32]]], kind: str
) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load as load_jax
    from linnet.jax import load_source

    source, weights, dequantized = model
    for make in (load_jax, load_source):
        function = make(
            source,
            generics={"In": IN, "Out": OUT},
            weights=weights,
            std_root=STDLIB,
            entry=ENTRIES[kind],
            numerics="exact",  # f32 products in f32 on a GPU too
        )
        got = np.asarray(function(_input()))
        np.testing.assert_allclose(got, _input() @ dequantized[kind].T, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("kind", list(ENTRIES))
def test_onnx_multiplies_dequantized(
    model: tuple[Path, Path, dict[str, NDArray[np.float32]]], kind: str
) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_model

    source, weights, dequantized = model
    exported = export_model(
        source,
        generics={"In": IN, "Out": OUT, "B": 3},
        weights=weights,
        std_root=STDLIB,
        entry=ENTRIES[kind],
    )
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (got,) = session.run(None, {session.get_inputs()[0].name: _input()})
    np.testing.assert_allclose(got, _input() @ dequantized[kind].T, rtol=1e-5, atol=1e-4)


def test_siblings_run_as_one_product(tmp_path: Path) -> None:
    """Each kind's two layers multiply as one product over their weights
    and scales side by side, the operands prepared once."""
    from linnet.torch import CompiledLinnetModule

    source = tmp_path / "siblings.linnet"
    source.write_text(SIBLINGS, encoding="utf-8")
    weights = tmp_path / "siblings.safetensors"
    kinds = ("rows", "rows", "groups", "groups", "blocks", "blocks")
    dequantized = _checkpoint(weights, ("a", "b", "c", "d", "e", "f"), kinds, torch.float32, 3)
    loaded = load(
        source, generics={"In": IN, "Out": OUT}, std_root=STDLIB, weights=weights, compile=True
    )
    x = _input()
    got = loaded.run_entry("forward", [torch.from_numpy(x)]).detach().numpy()
    want = sum((x @ dequantized[a].T) * (x @ dequantized[b].T) for a, b in ("ab", "cd", "ef"))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)
    assert isinstance(loaded, CompiledLinnetModule)
    generated = loaded.generated_source("forward")
    for helper in ("_int8_linear(", "_int8_group_linear(", "_fp8_block_linear("):
        assert generated.count(helper) == 1, helper
    assert "def prepare(" in generated


GROUPED = """\
module tests.eight_groups

use std.quant::{Int8GroupLinear}

pub block Model<In: Dim, Out: Dim, T: Float = f32>
where In % 128 == 0 {
    sub proj: Int8GroupLinear<In, Out, 128, T>

    pub entry forward<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        return proj.forward(x)
    }
}
"""


def test_int8_groups_round_within_half_a_step() -> None:
    from linnet.quant import dequantize_int8_groups, quantize_int8_groups

    weight = np.random.default_rng(0).standard_normal((OUT, IN)).astype(np.float32)
    q, scale = quantize_int8_groups(weight, 128, "bf16")
    assert q.dtype == np.int8 and q.shape == (OUT, IN) and scale.shape == (OUT, IN // 128)
    error = np.abs(dequantize_int8_groups(q, scale) - weight)
    assert np.all(error <= np.repeat(scale, 128, axis=1) / 2 + 1e-6)


def test_a_checkpoint_quantizes_to_int8_groups(tmp_path: Path) -> None:
    """`quantize_checkpoint(bits=8)` writes what `Int8GroupLinear` loads."""
    from linnet.quant import dequantize_int8_groups, quantize_checkpoint, quantize_int8_groups

    source = tmp_path / "grouped.linnet"
    source.write_text(GROUPED, encoding="utf-8")
    weight = torch.randn(OUT, IN, generator=torch.Generator().manual_seed(4))
    save_file({"model.proj.weight": weight}, str(tmp_path / "float.safetensors"))
    quantized = quantize_checkpoint(
        tmp_path / "float.safetensors",
        tmp_path / "int8.safetensors",
        patterns=["proj.weight"],
        dtype="f32",
        bindings={"proj.weight": "model.proj.weight"},
        bits=8,
    )
    assert quantized == ["proj.weight"]
    loaded = load(
        source,
        generics={"In": IN, "Out": OUT},
        std_root=STDLIB,
        weights=tmp_path / "int8.safetensors",
    )
    q, scale = quantize_int8_groups(weight.numpy(), 128)
    x = _input()
    got = loaded.run_entry("forward", [torch.from_numpy(x)]).detach().numpy()
    np.testing.assert_allclose(got, x @ dequantize_int8_groups(q, scale).T, rtol=1e-5, atol=1e-4)


def _cuda(capability: tuple[int, int]) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= capability


@pytest.mark.skipif(not _cuda((9, 0)), reason="FP8 kernels need compute capability 9 or later")
@pytest.mark.parametrize("numerics", ["equivalent", "fast"])
@pytest.mark.parametrize("rows", [1, 3, 17, 64])
@pytest.mark.parametrize("kind", list(ENTRIES))
def test_cuda_kernels(tmp_path: Path, kind: str, rows: int, numerics: str) -> None:
    """One row through the GEMV, more through the tensor-core kernels, or
    rounded to 8 bits under `fast`: within bf16's rounding of the product
    with the dequantized weight, rounded inputs within 8 bits'."""
    source = tmp_path / "eight.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    weights = tmp_path / "eight.safetensors"
    kinds = ("rows", "groups", "blocks")
    dequantized = _checkpoint(weights, kinds, kinds, torch.bfloat16, 0)
    loaded = load(
        source,
        generics={"In": IN, "Out": OUT, "T": "bf16"},
        std_root=STDLIB,
        weights=weights,
        device="cuda",
        compile=True,
        numerics=numerics,
    )
    x = torch.from_numpy(_input(rows)).to("cuda", torch.bfloat16)
    got = loaded.run_entry(ENTRIES[kind], [x]).float().cpu().numpy()
    want = x.float().cpu().numpy() @ dequantized[kind].T
    rounded = numerics == "fast" and rows > 1 and kind != "groups"
    assert np.abs(got - want).max() <= (0.06 if rounded else 0.01) * np.abs(want).max()
