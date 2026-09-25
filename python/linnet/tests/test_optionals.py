"""Optional parameters are present or absent one by one, as the checkpoint
has them, in every backend.

Real checkpoints mix them: a ResNet's convolutions have no bias and its
classifier has one; Qwen2.5 biases its query, key and value projections and
nothing else. The compiled PyTorch path used to decide for the whole model
from the first block with an optional parameter, which silently dropped the
classifier bias from every compiled ResNet; the ONNX export dropped every
optional by default; the JAX loader refused the mix outright.
"""

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
module tests.mixed

use std.nn.linear::{Linear}

pub block Model<D: Dim, T: Float = f32> {
    sub first: Linear<D, D, T>
    sub second: Linear<D, D, T>

    pub entry forward<B: Dim>(x: Tensor[B, D; T]) -> Tensor[B, D; T] {
        return second.forward(first.forward(x))
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
def checkpoint(tmp_path: Path) -> tuple[Path, Path, dict[str, torch.Tensor]]:
    """The first projection has no bias, the second has one -- in that order,
    so a decision made from the first block gets the second one wrong."""
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    weights = {
        "first.weight": torch.randn(8, 8),
        "second.weight": torch.randn(8, 8),
        "second.bias": torch.randn(8) * 3,
    }
    source = tmp_path / "mixed.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    path = tmp_path / "model.safetensors"
    save_file(weights, str(path))
    return source, path, weights


def _expected(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden = x @ weights["first.weight"].T
    return hidden @ weights["second.weight"].T + weights["second.bias"]


@pytest.mark.parametrize("compile", [False, True])
def test_torch_keeps_each_optional_as_the_checkpoint_has_it(
    checkpoint: tuple[Path, Path, dict[str, torch.Tensor]], compile: bool
) -> None:
    source, weights_path, weights = checkpoint
    model = load(source, generics={"D": 8}, std_root=STDLIB, weights=weights_path, compile=compile)
    x = torch.randn(3, 8)
    torch.testing.assert_close(model(x), _expected(x, weights), atol=1e-5, rtol=1e-5)


def test_onnx_embeds_the_bias_the_checkpoint_has(
    checkpoint: tuple[Path, Path, dict[str, torch.Tensor]],
) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_model

    source, weights_path, weights = checkpoint
    exported = export_model(
        source, generics={"D": 8, "B": 3}, weights=weights_path, std_root=STDLIB
    )
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = torch.randn(3, 8)
    (got,) = session.run(None, {session.get_inputs()[0].name: x.numpy()})
    np.testing.assert_allclose(got, _expected(x, weights).numpy(), atol=1e-5, rtol=1e-5)


def test_jax_accepts_a_mix_of_present_and_absent(
    checkpoint: tuple[Path, Path, dict[str, torch.Tensor]],
) -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from linnet.jax import load as load_jax

    source, weights_path, weights = checkpoint
    function = load_jax(source, generics={"D": 8}, weights=weights_path, std_root=STDLIB)
    x = torch.randn(3, 8)
    got = np.asarray(function(jnp.asarray(x.numpy())))  # pyright: ignore[reportUnknownMemberType]
    np.testing.assert_allclose(got, _expected(x, weights).numpy(), atol=1e-5, rtol=1e-5)
