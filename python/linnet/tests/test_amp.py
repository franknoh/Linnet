"""Mixed precision: f32 master weights, 16-bit compute.

PyTorch: `load(..., amp="bf16")` runs entries under `torch.autocast` while
the weights stay f32, so a linear layer computes in bf16 and gradients
arrive in f32. JAX: `load_source(..., generics T=bf16, cast_dtype=True)`
takes f32 master parameters, casts them for each call, and `jax.grad`
returns f32 gradients."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.torch import load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.amp

use std.nn.linear::{Linear}
use std.nn.softmax::{softmax}

pub block Model<I: Dim, O: Dim, T: Float = f32> {
    sub proj: Linear<I, O, T>

    pub entry forward<B: Dim>(x: Tensor[B, I; T]) -> Tensor[B, O; T] {
        return softmax(proj.forward(x))
    }
}
"""

GENERICS: dict[str, int | str] = {"I": 16, "O": 8}


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "amp.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generator = torch.Generator().manual_seed(0)
    weights = tmp_path / "model.safetensors"
    save_file(
        {
            "proj.weight": torch.randn(8, 16, generator=generator) * 0.5,
            "proj.bias": torch.randn(8, generator=generator),
        },
        str(weights),
    )
    return source, weights


@pytest.mark.parametrize("compile", [False, True])
def test_torch_autocast(files: tuple[Path, Path], compile: bool) -> None:
    source, weights = files
    x = torch.randn(4, 16, generator=torch.Generator().manual_seed(1))
    full = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=compile)
    mixed = load(
        source,
        generics=GENERICS,
        std_root=STDLIB,
        weights=weights,
        compile=compile,
        amp="bf16",
        trainable=True,
    )
    expected = full.run_entry("forward", [x])
    got = mixed.run_entry("forward", [x])
    # The product ran in bf16, so the results differ, by bf16 rounding and no
    # more. Which dtype the softmax returns is autocast's policy for the
    # device (f32 on CUDA; on the CPU it keeps bf16, accumulating in f32).
    assert got.dtype in (torch.float32, torch.bfloat16)
    assert not torch.equal(got.float(), expected)
    torch.testing.assert_close(got.float(), expected, atol=2e-2, rtol=2e-2)
    got.float().log().sum().backward()
    for parameter in mixed.parameters():
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0


def test_torch_rejects_an_unknown_precision(files: tuple[Path, Path]) -> None:
    source, weights = files
    with pytest.raises(Exception, match="amp"):
        load(source, generics=GENERICS, std_root=STDLIB, weights=weights, amp="f8")


def test_jax_master_weights(files: tuple[Path, Path]) -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from linnet.jax import load_source

    source, weights = files
    x = np.asarray(torch.randn(4, 16, generator=torch.Generator().manual_seed(1)))
    full = load_source(source, generics=GENERICS, weights=weights, std_root=STDLIB, entry="forward")
    mixed = load_source(
        source,
        generics={**GENERICS, "T": "bf16"},
        weights=weights,
        std_root=STDLIB,
        entry="forward",
        cast_dtype=True,
    )
    masters = {path: jnp.asarray(value, dtype=jnp.float32) for path, value in mixed.weights.items()}
    expected = np.asarray(full(jnp.asarray(x)))
    got = mixed.apply(masters, jnp.asarray(x, dtype=jnp.bfloat16))
    assert got.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(got, dtype=np.float32), expected, atol=3e-2, rtol=3e-2)

    def loss(parameters: dict[str, jax.Array]) -> jax.Array:
        out = mixed.apply(parameters, jnp.asarray(x, dtype=jnp.bfloat16))
        return jnp.log(out.astype(jnp.float32)).sum()

    grads = jax.grad(loss)(masters)
    for path, grad in grads.items():
        assert grad.dtype == jnp.float32, path
        assert bool(jnp.all(jnp.isfinite(grad))) and bool(jnp.any(grad != 0)), path
