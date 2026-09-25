"""`cast_dtype=True` runs an f32 checkpoint in bf16 under XLA, as the torch
loader does, so every backend's row in a benchmark computes in one dtype."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

pytest.importorskip("jax")

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.cast

pub block Model<T: Float> {
    param scale: Tensor[4; T]

    pub entry forward(x: Tensor[4; T]) -> Tensor[4; T] {
        return x * scale
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


def test_an_f32_checkpoint_runs_in_bf16(tmp_path: Path) -> None:
    import jax.numpy as jnp

    from linnet.jax import load

    source = tmp_path / "cast.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    scale = torch.tensor([0.5, 1.5, -2.0, 3.25])
    save_file({"scale": scale}, str(tmp_path / "f32.safetensors"))
    function = load(
        source,
        generics={"T": "bf16"},
        weights=tmp_path / "f32.safetensors",
        std_root=STDLIB,
        cast_dtype=True,
    )
    got = function(jnp.ones(4, dtype=jnp.bfloat16))  # pyright: ignore[reportUnknownMemberType]
    assert got.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(got, dtype=np.float32), scale.numpy())
