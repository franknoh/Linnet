"""NF4 linears (`std.quant::Nf4Linear`) from bitsandbytes checkpoints:
`import_quantized` repacks a layer's codes (bitsandbytes puts the first of
each pair in the high nibble) and decodes its nested absmax, and the layer
computes `x @ (nf4(q) * absmax).T + bias` on PyTorch, compiled and
interpreted, and JAX."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

from linnet import ir
from linnet.compiler import LinnetError
from linnet.plan import compile_plan
from linnet.quant import NF4, import_quantized
from linnet.torch import load
from linnet.weights import header_tensors, match_checkpoint, read_header
from tests.helpers import STDLIB

SOURCE = """\
module tests.nf4

use std.quant::{Nf4Linear}

pub block Model<In: Dim, Out: Dim>
where In % 64 == 0 {
    sub proj: Nf4Linear<In, Out, 64, f32>

    pub entry forward<B: Dim>(x: Tensor[B, In; f32]) -> Tensor[B, Out; f32] {
        return proj.forward(x)
    }
}
"""

IN, OUT, BLOCK, NESTED_BLOCK = 128, 8, 64, 256
CONFIG = {"quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}


def _checkpoint(path: Path, nested: bool) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """A bitsandbytes NF4 layer at `model.proj`, as its 4-bit layers save
    one; returns the weight and bias it stands for."""
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 16, OUT * IN).astype(np.uint8)
    blocks = OUT * IN // BLOCK
    tensors = {
        "model.proj.weight": torch.from_numpy((codes[1::2] | (codes[0::2] << 4)).reshape(-1, 1)),
        "model.proj.weight.quant_map": torch.from_numpy(NF4),
    }
    state: dict[str, object] = {
        "quant_type": "nf4",
        "blocksize": BLOCK,
        "dtype": "bfloat16",
        "shape": [OUT, IN],
    }
    if nested:
        absmax_codes = rng.integers(0, 256, blocks).astype(np.uint8)
        nested_absmax = rng.uniform(0.5, 1.5, -(-blocks // NESTED_BLOCK)).astype(np.float32)
        nested_map = np.sort(rng.uniform(-1.0, 1.0, 256)).astype(np.float32)
        offset = 0.75
        absmax = nested_map[absmax_codes] * nested_absmax[np.arange(blocks) // NESTED_BLOCK]
        absmax = (absmax + np.float32(offset)).astype(np.float32)
        tensors["model.proj.weight.absmax"] = torch.from_numpy(absmax_codes)
        tensors["model.proj.weight.nested_absmax"] = torch.from_numpy(nested_absmax)
        tensors["model.proj.weight.nested_quant_map"] = torch.from_numpy(nested_map)
        state |= {"nested_blocksize": NESTED_BLOCK, "nested_offset": offset}
    else:
        absmax = rng.uniform(0.1, 2.0, blocks).astype(np.float32)
        tensors["model.proj.weight.absmax"] = torch.from_numpy(absmax)
    encoded = np.frombuffer(json.dumps(state).encode("utf-8"), dtype=np.uint8).copy()
    tensors["model.proj.weight.quant_state.bitsandbytes__nf4"] = torch.from_numpy(encoded)
    bias = rng.standard_normal(OUT).astype(np.float32)
    tensors["model.proj.bias"] = torch.from_numpy(bias)
    save_file(tensors, str(path))
    weight = (NF4[codes].reshape(-1, BLOCK) * absmax[:, None]).reshape(OUT, IN)
    return weight.astype(np.float32), bias


@pytest.fixture(params=[True, False], ids=["nested", "plain"])
def imported(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[Path, Path, NDArray[np.float32], NDArray[np.float32]]:
    source = tmp_path / "nf4.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    weight, bias = _checkpoint(tmp_path / "bnb.safetensors", bool(request.param))
    quantized = import_quantized(
        tmp_path / "bnb.safetensors",
        tmp_path / "nf4.safetensors",
        bindings={"proj.weight": "model.proj.weight", "proj.bias": "model.proj.bias"},
        config=CONFIG,
    )
    assert quantized == ["proj.weight"]
    return source, tmp_path / "nf4.safetensors", weight, bias


def _input() -> NDArray[np.float32]:
    return np.random.default_rng(1).standard_normal((3, IN)).astype(np.float32)


@pytest.mark.parametrize("compile", [False, True])
def test_torch_multiplies_the_imported_layer(
    imported: tuple[Path, Path, NDArray[np.float32], NDArray[np.float32]], compile: bool
) -> None:
    source, weights, weight, bias = imported
    model = load(
        source, generics={"In": IN, "Out": OUT}, std_root=STDLIB, weights=weights, compile=compile
    )
    got = model.run_entry("forward", [torch.from_numpy(_input())])
    assert isinstance(got, torch.Tensor)
    np.testing.assert_allclose(
        got.detach().numpy(), _input() @ weight.T + bias, rtol=1e-5, atol=1e-5
    )


def test_jax_multiplies_the_imported_layer(
    imported: tuple[Path, Path, NDArray[np.float32], NDArray[np.float32]],
) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load as load_jax

    source, weights, weight, bias = imported
    function = load_jax(
        source, generics={"In": IN, "Out": OUT}, weights=weights, std_root=STDLIB, entry="forward"
    )
    got = np.asarray(function(_input()))
    np.testing.assert_allclose(got, _input() @ weight.T + bias, rtol=1e-5, atol=1e-5)


def test_the_checkpoint_matches_the_manifest(
    imported: tuple[Path, Path, NDArray[np.float32], NDArray[np.float32]],
) -> None:
    """The packed weight's bytes, `[Out, In / 2]`, match its `[Out, In]`
    parameter, as `nest check` and the Hub loader compare them."""
    source, weights, _, _ = imported
    program = compile_plan(source, std_root=STDLIB)
    bindings = ir.bind_root(program, {"In": IN, "Out": OUT})
    tensors = header_tensors(read_header(weights))
    found, problems = match_checkpoint(program, bindings, tensors, {})
    assert problems == []
    assert sorted(path for _, located in found for path, _ in located) == [
        "proj.absmax",
        "proj.bias",
        "proj.weight",
    ]


def test_a_root_weight_keeps_its_siblings_unprefixed(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "bnb.safetensors", nested=True)
    quantized = import_quantized(
        tmp_path / "bnb.safetensors",
        tmp_path / "nf4.safetensors",
        bindings={"weight": "model.proj.weight", "bias": "model.proj.bias"},
        config=CONFIG,
    )
    assert quantized == ["weight"]
    names = set(header_tensors(read_header(tmp_path / "nf4.safetensors")))
    assert {"weight", "absmax", "bias"} <= names


def test_other_types_do_not_import(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "bnb.safetensors", nested=False)
    with pytest.raises(LinnetError, match="fp4"):
        import_quantized(
            tmp_path / "bnb.safetensors",
            tmp_path / "out.safetensors",
            config={**CONFIG, "bnb_4bit_quant_type": "fp4"},
        )
    with pytest.raises(LinnetError, match="4-bit"):
        import_quantized(
            tmp_path / "bnb.safetensors",
            tmp_path / "out.safetensors",
            config={"quant_method": "bitsandbytes", "load_in_8bit": True},
        )
