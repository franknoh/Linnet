"""Generic values given from outside the program are checked against the
root block's `where` clause: the checker proves it at every call inside the
program, never for the caller's values. `H % Heads == 0` with `H = 5,
Heads = 2` would otherwise split 5 features into two heads of 2."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from linnet import LinnetError
from linnet.torch import load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.constrained

pub block Model<H: Dim, Heads: Dim, T: Float = f32>
where
    Heads > 0,
    H % Heads == 0
{
    param w: Tensor[H; T]

    pub entry forward(x: Tensor[H; T]) -> Tensor[Heads, H / Heads; T] {
        let y[h] = x[h] * w[h]
        return reshape(y, [Heads, H / Heads])
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


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "constrained.linnet"
    path.write_text(SOURCE, encoding="utf-8")
    return path


@pytest.mark.parametrize("compile", [False, True])
def test_torch_refuses_generics_that_break_the_where_clause(source: Path, compile: bool) -> None:
    model = load(source, generics={"H": 6, "Heads": 2}, std_root=STDLIB, compile=compile)
    assert tuple(model.run_entry("forward", [torch.ones(6)]).shape) == (2, 3)
    with pytest.raises(LinnetError, match=r"where|H % Heads == 0"):
        broken = load(source, generics={"H": 5, "Heads": 2}, std_root=STDLIB, compile=compile)
        broken.run_entry("forward", [torch.ones(5)])


def test_exports_refuse_generics_that_break_the_where_clause(source: Path) -> None:
    pytest.importorskip("onnx")
    from linnet.onnx import export_model

    with pytest.raises(LinnetError, match=r"constraint `H % Heads == 0` does not hold"):
        export_model(
            source,
            generics={"H": 5, "Heads": 2},
            weights=source.parent / "model.safetensors",
            std_root=STDLIB,
        )
