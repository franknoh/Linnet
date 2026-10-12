"""DeepGEMM as a backend: it takes the block-scaled FP8 `linear` calls the
export lets round their input to FP8, and on Hopper multiplies as Linnet's
own FP8 path does."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

from linnet.backends import Graph, Target
from linnet.compiler import find_compiler
from linnet.torch import load
from linnet.torch.compiled import CompiledLinnetModule
from linnet.torch.deepgemm import DeepGemm
from tests.helpers import STDLIB, requires_sm90

SOURCE = """\
module tests.blocks

use std.quant::{Fp8BlockLinear}

pub block Model<In: Dim, Out: Dim, T: Float = bf16>
where
    In % 128 == 0,
    Out % 128 == 0
{
    sub proj: Fp8BlockLinear<In, Out, 128, T>

    pub entry forward<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        return proj.forward(x)
    }
}
"""

IN, OUT, ROWS = 512, 256, 96


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "blocks.linnet"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def _graph(source: Path, numerics: str) -> Graph:
    binds = ["--bind", f"In={IN}", "--bind", f"Out={OUT}", "--bind", f"B={ROWS}"]
    command = [find_compiler(), "graph", "--std", str(STDLIB), "--entry", "forward"]
    completed = subprocess.run(
        [*command, "--numerics", numerics, *binds, str(source)],
        capture_output=True,
        text=True,
        check=True,
    )
    return Graph.parse(completed.stdout)


def test_it_takes_only_calls_that_may_round_their_input(source: Path) -> None:
    hopper = Target("cuda", 0, "H100", (9, 0))
    assert len(list(DeepGemm().match(_graph(source, "fast"), hopper))) == 1
    assert list(DeepGemm().match(_graph(source, "equivalent"), hopper)) == []


@requires_sm90
def test_it_multiplies_as_the_entry_does(source: Path, tmp_path: Path) -> None:
    pytest.importorskip("deep_gemm")
    generator = torch.Generator().manual_seed(0)
    full = torch.randn(OUT, IN, generator=generator)
    blocks = full.reshape(OUT // 128, 128, IN // 128, 128)
    scale = blocks.abs().amax((1, 3)) / 448.0
    wide = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    weight = (full / wide).to(torch.float8_e4m3fn)
    save_file(
        {"proj.weight": weight, "proj.scale": scale.bfloat16()}, str(tmp_path / "w.safetensors")
    )
    planned, direct = (
        load(
            source,
            generics={"In": IN, "Out": OUT},
            std_root=STDLIB,
            weights=tmp_path / "w.safetensors",
            numerics="fast",
            device="cuda",
            compile=True,
        )
        for _ in range(2)
    )
    assert isinstance(planned, CompiledLinnetModule)
    x = torch.randn(ROWS, IN, device="cuda").bfloat16()
    plan = planned.plan_for("forward", [x], backends=[DeepGemm()], measure=False)
    assert [step.backend for step in plan.steps] == ["deepgemm"]
    with torch.no_grad():
        (got,) = plan.run([x], planned.context_for(plan))
        expected: object = direct.run_entry("forward", [x])
    assert isinstance(expected, torch.Tensor)
    torch.testing.assert_close(got.float(), expected.float(), rtol=2e-2, atol=2e-2)
    findings = planned.conform("forward", [x], DeepGemm(), tolerance=5e-2)
    assert len(findings) == 1 and findings[0].passed
