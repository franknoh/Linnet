"""`state` members materialize as execution state: kept between entry
calls, updated by the block's assignments, reset on request, and never part
of the weights."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet_torch import bind_weights, load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module kv

pub block Cache<B: Dim, Max: Dim, D: Dim> {
    param scale: Tensor[D; f32]
    state keys: Tensor[B, Max, D; f32]

    pub fn step(key: Tensor[B, 1, D; f32], pos: i32) -> Tensor[B, Max, D; f32] {
        let positions = iota<i32>(Max)
        let updated[b, s, d] = select(positions[s] == pos, key[b, 0, d] * scale[d], keys[b, s, d])
        keys = updated
        return keys
    }
}

pub block Model<B: Dim, Max: Dim, D: Dim> {
    sub layers: [Cache<B, Max, D>; 2]

    pub entry step(key: Tensor[B, 1, D; f32], pos: i32) -> Tensor[B, Max, D; f32] {
        var total = fill<f32>([B, Max, D], 0.0)
        static for layer in layers {
            total = total + layer.step(key, pos)
        }
        return total
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


def test_state_persists_between_calls(tmp_path: Path) -> None:
    source = tmp_path / "kv.linnet"
    source.write_text(SOURCE)
    model = load(source, generics={"B": 1, "Max": 4, "D": 3}, std_root=STDLIB)
    assert model.state_paths() == ["layers[*].keys"]
    # State is not a weight: binding the parameters alone is complete.
    weights = tmp_path / "model.safetensors"
    save_file({f"layers.{i}.scale": torch.ones(3) * (i + 1) for i in range(2)}, str(weights))
    bind_weights(model, weights)
    assert "root.layers.0.keys" not in model.state_dict()

    def state(path: str) -> torch.Tensor:
        return dict(model.named_buffers())[path]

    key0 = torch.arange(3, dtype=torch.float32).reshape(1, 1, 3)
    key1 = key0 + 10
    out0 = model.run_entry("step", [key0, torch.tensor(0, dtype=torch.int32)])
    out1 = model.run_entry("step", [key1, torch.tensor(2, dtype=torch.int32)])
    # Both layers hold both positions: scales 1 and 2 sum to 3 times the key.
    expected = torch.zeros(1, 4, 3)
    expected[0, 0] = key0[0, 0] * 3
    expected[0, 2] = key1[0, 0] * 3
    torch.testing.assert_close(out1, expected)
    torch.testing.assert_close(out0[0, 2], torch.zeros(3))  # first call had only position 0
    torch.testing.assert_close(state("root.layers.1.keys")[0, 2], key1[0, 0] * 2)

    model.reset_state()
    torch.testing.assert_close(state("root.layers.0.keys"), torch.zeros(1, 4, 3))
    again = model.run_entry("step", [key0, torch.tensor(0, dtype=torch.int32)])
    torch.testing.assert_close(again, out0)
