"""XLA's options for a program by what bounds it: a pass over many tokens
does many FLOPs a byte of its weights, a step over one does about one."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linnet.compiler import bind_arguments, run_compiler
from linnet.jax import load, load_source
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


@pytest.mark.parametrize("loader", ["load", "load_source"])
def test_options_stay_on_the_outermost_jit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: str
) -> None:
    """A program given options runs inside a training step's `jit` and
    `grad` too (only the outermost `jit` takes options): as on a GPU."""

    def options(_: str) -> dict[str, str | bool]:
        return {"xla_gpu_enable_triton_gemm": False}

    # The module, not the function `linnet.jax` exports under its name.
    monkeypatch.setattr(importlib.import_module("linnet.jax.load"), "compiler_options", options)
    source = tmp_path / "options.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    weights = {"proj.weight": np.eye(8, dtype=np.float32)}
    make = load_source if loader == "load_source" else load
    function = make(source, generics={"H": 8, "T": "f32"}, weights=weights, std_root=STDLIB)
    x = jnp.ones((4, 8), jnp.float32)
    np.testing.assert_allclose(np.asarray(function(x)), np.ones((4, 8)))
    parameters = function.parameters_for(x)

    def run(values: dict[str, jax.Array]) -> jax.Array:
        return function.apply(values, x)

    def loss(values: dict[str, jax.Array]) -> jax.Array:
        return jnp.sum(run(values))

    np.testing.assert_allclose(np.asarray(jax.jit(run)(parameters)), np.ones((4, 8)))
    if loader == "load_source":
        grads = jax.jit(jax.grad(loss))(parameters)
        np.testing.assert_allclose(np.asarray(grads["proj.weight"]), np.full((8, 8), 4.0))
