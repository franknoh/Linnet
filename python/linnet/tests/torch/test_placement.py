"""Placement: blocks spread over devices, and offloaded blocks streamed in,
must compute exactly what the model computes on one device."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch

from linnet.compiler import find_compiler
from linnet.torch import CompiledLinnetModule, Placement, load
from linnet.torch.placement import apply, parse_size, plan

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.placement

use std.nn.activations::{relu}
use std.nn.attention::{causal_mask}
use std.nn.linear::{Linear}

pub block Layer<D: Dim, T: Float> {
    sub proj: Linear<D, D, T>

    pub fn forward<B: Dim, S: Dim>(x: Tensor[B, S, D; T]) -> Tensor[B, S, D; T] {
        let mask = causal_mask<S, S>()
        let mixed[b, s, d] = sum[k] select(mask[s, k], x[b, k, d], cast<T>(0.0))
        return x + relu(proj.forward(mixed))
    }
}

pub block Model<D: Dim, Layers: Dim, T: Float = f32> {
    sub layers: [Layer<D, T>; Layers]
    sub head: Linear<D, D, T>

    pub entry forward<B: Dim, S: Dim>(x: Tensor[B, S, D; T]) -> Tensor[B, S, D; T] {
        var h = x
        static for layer in layers {
            h = layer.forward(h)
        }
        return head.forward(h)
    }
}
"""

GENERICS: dict[str, int | str] = {"D": 16, "Layers": 4, "T": "f32"}


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "placement.linnet"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def _randomize(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.3)


def _generate(source: Path, *flags: str) -> str:
    command = [find_compiler(), "torch", "--root", "Model", "--entry", "forward"]
    command += ["--bind", "D=16", "--bind", "Layers=4", "--bind", "B=1", "--bind", "S=4"]
    command += ["--numerics", "fast", "--std", str(STDLIB), *flags, str(source)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return completed.stdout


def _main_body(generated: str) -> str:
    return generated[generated.index("def main(") :]


def test_the_stream_crosses_to_the_next_device_once(source: Path) -> None:
    """Layers 2 and 3 and the head on slot 1: the residual stream moves once,
    when layer 2 first reads it, and never comes back."""
    body = _main_body(
        _generate(source, "--place", "layers.2=1", "--place", "layers.3=1", "--place", "head=1")
    )
    assert body.count(".to(_dev[1], non_blocking=True)") == 1
    assert ".to(_dev[0]" not in body
    assert "def main(" in body and body.splitlines()[0].endswith("_dev):")


def test_an_offloaded_block_streams_its_weights_and_drops_them(source: Path) -> None:
    body = _main_body(_generate(source, "--offload", "layers.1", "--optionals", "present"))
    # Weight and bias come in together and leave together, right after the
    # block returns, so only one offloaded block is ever resident.
    transfers = [line for line in body.splitlines() if ".to(_dev[0], non_blocking=True)" in line]
    assert len(transfers) == 2
    released = [line.strip() for line in body.splitlines() if line.strip().startswith("del ")]
    names = sorted(line.split(" = ")[0].strip() for line in transfers)
    assert released == ["del " + ", ".join(names)]


def test_without_placement_the_source_is_unchanged(source: Path) -> None:
    generated = _generate(source)
    assert "_dev[" not in generated and "SLOTS" not in generated


def test_placement_is_only_for_the_torch_target(source: Path) -> None:
    command = [find_compiler(), "jax", "--root", "Model", "--entry", "forward"]
    command += ["--bind", "D=16", "--bind", "Layers=4", "--bind", "B=1", "--bind", "S=4"]
    command += ["--std", str(STDLIB), "--place", "layers.1=1", str(source)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "placement is for `torch`" in completed.stderr


def test_sizes_read_like_people_write_them() -> None:
    assert parse_size("20GiB") == 20 << 30
    assert parse_size("1.5 GB") == 1_500_000_000
    assert parse_size(4096) == 4096


def _devices() -> tuple[torch.device, torch.device]:
    """Two devices to place across: a GPU and the host where there is one,
    otherwise the host twice, which still runs every transfer and release
    the placement compiles in. RunPod measures real multi-GPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", 0), torch.device("cpu")
    return torch.device("cpu"), torch.device("cpu")


def _reference_and_placed(
    source: Path, placement: Placement
) -> tuple[torch.Tensor, CompiledLinnetModule, torch.Tensor]:
    reference = load(source, generics=GENERICS, std_root=STDLIB, compile=True)
    _randomize(reference)
    x = torch.randn(1, 4, 16)
    expected = reference.run_entry("forward", [x])
    placed = load(source, generics=GENERICS, std_root=STDLIB, compile=True)
    assert isinstance(placed, CompiledLinnetModule)
    placed.load_state_dict(reference.state_dict())
    apply(placed, placement)
    placed.placement = placement
    return expected, placed, x


def test_offloaded_blocks_compute_what_the_resident_model_does(source: Path) -> None:
    first, _ = _devices()
    placement = Placement(
        (first,),
        dict.fromkeys(("layers.0", "layers.1", "layers.2", "layers.3", "head"), 0),
        ("layers.1", "layers.2"),
    )
    expected, placed, x = _reference_and_placed(source, placement)
    weight = placed.root.layers[1].proj.weight  # pyright: ignore[reportIndexIssue, reportUnknownMemberType]
    assert weight.device.type == "cpu"  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        assert weight.is_pinned()  # pyright: ignore[reportUnknownMemberType]
    got = placed.run_entry("forward", [x])
    assert got.device == first
    torch.testing.assert_close(got.cpu(), expected, atol=1e-5, rtol=1e-5)
    assert "del " in placed.generated_source("forward")


def test_two_devices_compute_what_one_does(source: Path) -> None:
    placement = Placement(
        _devices(),
        {"layers.0": 0, "layers.1": 0, "layers.2": 1, "layers.3": 1, "head": 1},
        (),
    )
    expected, placed, x = _reference_and_placed(source, placement)
    got = placed.run_entry("forward", [x])
    torch.testing.assert_close(got.cpu(), expected, atol=1e-5, rtol=1e-5)
    assert "SLOTS = 2" in placed.generated_source("forward")


def test_the_plan_offloads_what_the_budget_cannot_hold(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planned against a pretend 8 GiB GPU, so the arithmetic is checked
    anywhere; real memory is RunPod's to measure."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index=0: (8 << 30, 8 << 30))
    model = load(source, generics=GENERICS, std_root=STDLIB, compile=True)
    assert isinstance(model, CompiledLinnetModule)
    per_layer = 16 * 16 * 4
    roomy = plan(model)
    assert not roomy.offloaded and roomy.trivial
    # Room for three units: the rest is offloaded, and the first layers stay.
    squeezed = plan(model, max_memory={0: 3 * per_layer})
    assert squeezed.offloaded
    assert "layers.0" not in squeezed.offloaded
    assert squeezed.offloaded == tuple(sorted(squeezed.offloaded, key=list(roomy.slots).index))
    with pytest.raises(Exception, match="offload=False"):
        plan(model, max_memory={0: 3 * per_layer}, offload=False)
