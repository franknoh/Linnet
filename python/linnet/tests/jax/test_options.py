"""XLA's options for a program by what bounds it: a pass over many tokens
does many FLOPs a byte of its weights, a step over one does about one."""

from __future__ import annotations

from pathlib import Path

from linnet.compiler import bind_arguments, run_compiler
from linnet.jax.load import compiler_options, intensity

from .test_round_trip import STDLIB

SOURCE = """\
module tests.options

use std.nn.linear::{Linear}

pub block Model<H: Dim, T: Float = bf16> {
    sub proj: Linear<H, H, T>

    pub entry forward<S: Dim>(x: Tensor[S, H; T]) -> Tensor[S, H; T] {
        return proj.forward(x)
    }
}
"""


def _module(tmp_path: Path, tokens: int) -> str:
    source = tmp_path / "options.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    bindings = bind_arguments({"H": 512, "S": tokens})
    return run_compiler(
        "stablehlo",
        "--root",
        "Model",
        "--entry",
        "forward",
        *bindings,
        "--std",
        str(STDLIB),
        str(source),
    )


def test_many_tokens_are_bound_by_arithmetic(tmp_path: Path) -> None:
    # 2 * S * H * H FLOPs over the weight's H * H and the input's S * H, in
    # bf16: about S for S well below H, a few hundred here.
    many = intensity(_module(tmp_path, 1024))
    one = intensity(_module(tmp_path, 1))
    assert 300 < many < 400
    assert one < 2


def test_off_gpus_the_defaults_stay(tmp_path: Path) -> None:
    assert compiler_options(_module(tmp_path, 1024)) is None
