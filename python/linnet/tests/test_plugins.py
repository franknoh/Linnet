"""A backend from another package (docs/backends.md): found through its
`linnet.backends` entry point, left out when written for another contract
or when it fails to load, used by `plan_for` by default, and checked against
the generated source by `conform`."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from linnet import backends
from linnet.torch import load
from linnet.torch.compiled import CompiledLinnetModule
from tests.helpers import LLAMA, STDLIB, TINY_LLAMA, random_llama

# A package that knows only Linnet's public contract.
PACKAGE = """\
from __future__ import annotations

import torch

from linnet.backends import Capabilities, ExecutableMeta, Region


class _Call:
    def __init__(self, weight, off_by):
        self.weight, self.off_by = weight, off_by
        self.meta = ExecutableMeta((), ())

    def prepare(self, context):
        return None

    def execute(self, inputs, outputs, context):
        out = torch.nn.functional.linear(inputs[0], context.parameters[self.weight])
        return [out + self.off_by]


class Linear:
    contract = 1

    def __init__(self, name="plugin-linear", off_by=0.0):
        self.name, self.version, self.off_by = name, "0.1", off_by

    def probe(self, target):
        return Capabilities()

    def match(self, graph, target):
        return [
            Region.of(graph, call.start, call.stop, call.index)
            for call in graph.semantic_calls("std.nn.linear::linear")
        ]

    def compile(self, region, target, options):
        return _Call(region.parameters[0], self.off_by)

    def estimate_cost(self, region, target):
        return None


class Future(Linear):
    contract = 2


def make():
    return Linear()


def future():
    return Future("plugin-future")
"""


@pytest.fixture
def plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """`plugin` installed: its module, and a distribution whose entry points
    name a backend, one written for another contract, and one that fails."""
    (tmp_path / "plugin.py").write_text(PACKAGE, encoding="utf-8")
    info = tmp_path / "plugin-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: plugin\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text(
        "[linnet.backends]\nlinear = plugin:make\nfuture = plugin:future\nbroken = plugin:missing\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setattr(backends, "_registered", [])
    monkeypatch.setattr(backends, "_skipped", {})
    backends._load_entry_points.cache_clear()  # pyright: ignore[reportPrivateUsage]
    yield
    backends._load_entry_points.cache_clear()  # pyright: ignore[reportPrivateUsage]
    sys.modules.pop("plugin", None)


def _model(tmp_path: Path) -> CompiledLinnetModule:
    weights = random_llama(tmp_path / "model.safetensors")
    model = load(LLAMA, generics=TINY_LLAMA, std_root=STDLIB, weights=weights, compile=True)
    assert isinstance(model, CompiledLinnetModule)
    return model


@pytest.mark.usefixtures("plugin")
def test_a_package_adds_a_backend_through_its_entry_point() -> None:
    assert [backend.name for backend in backends.available()] == ["plugin-linear"]
    left_out = backends.skipped()
    assert "written for contract 2" in left_out["plugin-future"]
    assert "failed to load" in left_out["broken"]


@pytest.mark.usefixtures("plugin")
def test_plans_use_the_installed_backends(tmp_path: Path) -> None:
    model = _model(tmp_path)
    token = torch.tensor([[3]], dtype=torch.int32)
    plan = model.plan_for("decode", [token, torch.tensor(0, dtype=torch.int32)], measure=False)
    assert [step.backend for step in plan.steps].count("plugin-linear") == 2 * 7 + 1
    with torch.no_grad():
        (logits,) = plan.run([token, torch.tensor(0, dtype=torch.int32)], model.context_for(plan))
    assert logits.shape == (1, 11)


@pytest.mark.usefixtures("plugin")
def test_conform_passes_a_backend_that_agrees_and_flags_one_that_does_not(
    tmp_path: Path,
) -> None:
    import plugin  # pyright: ignore[reportMissingImports]

    model = _model(tmp_path)
    inputs = [torch.tensor([[3]], dtype=torch.int32), torch.tensor(0, dtype=torch.int32)]
    agrees = model.conform("decode", inputs, plugin.Linear())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    assert len(agrees) == 2 * 7 + 1
    assert all(finding.passed for finding in agrees)
    differs = model.conform("decode", inputs, plugin.Linear("off", off_by=0.5))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    assert not any(finding.passed for finding in differs)
    assert all(finding.difference == pytest.approx(0.5) for finding in differs)
