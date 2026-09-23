"""`std.quant`: int8 and packed int4 weights dequantize to the values they
were quantized from, and the quantized linear layers agree with a float
linear over the dequantized weights."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from linnet_torch import load
from linnet_torch.module import BlockModule

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module quant_check

use std.quant::{Int4Linear, Int8Linear, unpack_int4}

pub block Model<In: Dim, Out: Dim>
where In % 2 == 0 {
    sub eight: Int8Linear<In, Out, f32>
    sub four: Int4Linear<In, Out, f32>

    pub entry run8<B: Dim>(x: Tensor[B, In; f32]) -> Tensor[B, Out; f32] {
        return eight.forward(x)
    }

    pub entry run4<B: Dim>(x: Tensor[B, In; f32]) -> Tensor[B, Out; f32] {
        return four.forward(x)
    }

    pub entry unpack(packed: Tensor[Out, In / 2; i8]) -> Tensor[Out, In; i8] {
        return unpack_int4(packed)
    }
}
"""


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]


def _pack_int4(values: np.ndarray) -> np.ndarray:
    """Two signed nibbles per byte: element 2k low, 2k + 1 high."""
    low = values[:, 0::2].astype(np.int32) & 15
    high = values[:, 1::2].astype(np.int32) & 15
    return ((high << 4) | low).astype(np.uint8).view(np.int8)


def test_quantized_linears_match_dequantized_float(tmp_path: Path) -> None:
    source = tmp_path / "quant.linnet"
    source.write_text(SOURCE)
    inputs, outputs = 16, 6
    model = load(source, generics={"In": inputs, "Out": outputs}, std_root=STDLIB)

    rng = np.random.default_rng(0)
    q8 = rng.integers(-128, 128, size=(outputs, inputs), dtype=np.int64).astype(np.int8)
    q4 = rng.integers(-8, 8, size=(outputs, inputs), dtype=np.int64).astype(np.int8)
    scale8 = rng.uniform(0.01, 0.1, size=outputs).astype(np.float32)
    scale4 = rng.uniform(0.1, 0.5, size=outputs).astype(np.float32)
    bias = rng.standard_normal(outputs).astype(np.float32)
    model.load_state_dict(
        {
            "root.eight.weight": torch.tensor(q8),
            "root.eight.scale": torch.tensor(scale8),
            "root.eight.bias": torch.tensor(bias),
            "root.four.weight": torch.tensor(_pack_int4(q4)),
            "root.four.scale": torch.tensor(scale4),
            "root.four.bias": torch.tensor(bias),
        },
        strict=False,
    )
    for name in ("root.eight", "root.four"):
        block = model.get_submodule(name)
        assert isinstance(block, BlockModule)
        block.absent_params.discard("bias")  # the biases above are present

    # Packing round-trips through `unpack_int4`.
    unpacked: torch.Tensor = model.run_entry("unpack", [torch.tensor(_pack_int4(q4))])
    np.testing.assert_array_equal(unpacked.numpy(), q4)

    x = torch.randn(3, inputs)
    expected8 = x @ (torch.tensor(q8).float() * torch.tensor(scale8)[:, None]).T + torch.tensor(
        bias
    )
    expected4 = x @ (torch.tensor(q4).float() * torch.tensor(scale4)[:, None]).T + torch.tensor(
        bias
    )
    torch.testing.assert_close(model.run_entry("run8", [x]), expected8, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model.run_entry("run4", [x]), expected4, atol=1e-5, rtol=1e-5)
