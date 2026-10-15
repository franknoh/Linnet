"""The backend plugins in `examples/09-backends`, each a package of its own
that registers through the `linnet.backends` entry point: what each takes
from a graph, on any machine, and on a GPU with its library installed, that
a plan with it computes what the entry does."""

from __future__ import annotations

import importlib.util
import subprocess
import tomllib
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

from linnet.backends import Backend, Context, Graph, Target
from linnet.compiler import find_compiler
from linnet.torch import load
from linnet.torch.compiled import CompiledLinnetModule
from linnet.torch.planned import Captured
from tests.helpers import LLAMA, REPO, STDLIB, requires_cuda, requires_sm90

PLUGINS = REPO / "examples" / "09-backends"
HOPPER = Target("cuda", 0, "H100", (9, 0))


def _plugin(name: str, backend: str) -> Backend[torch.Tensor]:
    """Example plugin `name`'s `backend`, its module imported from the
    directory its entry point names, without installing the package."""
    project = tomllib.loads((PLUGINS / name / "pyproject.toml").read_text(encoding="utf-8"))
    entry = project["project"]["entry-points"]["linnet.backends"][name]
    module_name, made = entry.split(":")
    assert made == backend
    spec = importlib.util.spec_from_file_location(
        module_name, PLUGINS / name / module_name / "__init__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    made_backend: Backend[torch.Tensor] = getattr(module, backend)()
    return made_backend


def _graph(source: Path, entry: str, generics: dict[str, int | str], numerics: str) -> Graph:
    binds = [part for name, value in generics.items() for part in ("--bind", f"{name}={value}")]
    command = [find_compiler(), "graph", "--std", str(STDLIB), "--entry", entry]
    completed = subprocess.run(
        [*command, "--numerics", numerics, *binds, str(source)],
        capture_output=True,
        text=True,
        check=True,
    )
    return Graph.parse(completed.stdout)


def _compiled(
    source: Path, generics: dict[str, int | str], weights: Path, numerics: str = "fast"
) -> CompiledLinnetModule:
    model = load(
        source,
        generics=generics,
        std_root=STDLIB,
        weights=weights,
        device="cuda",
        compile=True,
        numerics=numerics,
    )
    assert isinstance(model, CompiledLinnetModule)
    return model


# ---- DeepGEMM: block-scaled FP8 `linear` calls the export lets round their
# input to FP8, on Hopper as Linnet's own FP8 path multiplies.

FP8 = """\
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
FP8_GENERICS: dict[str, int | str] = {"In": 512, "Out": 256}
FP8_ROWS = 96


@pytest.fixture
def fp8_source(tmp_path: Path) -> Path:
    path = tmp_path / "blocks.linnet"
    path.write_text(FP8, encoding="utf-8")
    return path


def test_deepgemm_takes_only_calls_that_may_round_their_input(fp8_source: Path) -> None:
    deepgemm = _plugin("deepgemm", "DeepGemm")
    generics = {**FP8_GENERICS, "B": FP8_ROWS}
    assert len(list(deepgemm.match(_graph(fp8_source, "forward", generics, "fast"), HOPPER))) == 1
    assert list(deepgemm.match(_graph(fp8_source, "forward", generics, "equivalent"), HOPPER)) == []


@requires_sm90
def test_deepgemm_multiplies_as_the_entry_does(fp8_source: Path, tmp_path: Path) -> None:
    pytest.importorskip("deep_gemm")
    deepgemm = _plugin("deepgemm", "DeepGemm")
    out_features, in_features = int(FP8_GENERICS["Out"]), int(FP8_GENERICS["In"])
    generator = torch.Generator().manual_seed(0)
    full = torch.randn(out_features, in_features, generator=generator)
    blocks = full.reshape(out_features // 128, 128, in_features // 128, 128)
    scale = blocks.abs().amax((1, 3)) / 448.0
    wide = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    weight = (full / wide).to(torch.float8_e4m3fn)
    save_file(
        {"proj.weight": weight, "proj.scale": scale.bfloat16()}, str(tmp_path / "w.safetensors")
    )
    planned, direct = (
        _compiled(fp8_source, FP8_GENERICS, tmp_path / "w.safetensors", numerics="fast")
        for _ in range(2)
    )
    x = torch.randn(FP8_ROWS, in_features, device="cuda").bfloat16()
    plan = planned.plan_for("forward", [x], backends=[deepgemm], measure=False)
    assert [step.backend for step in plan.steps].count("deepgemm") == 1
    with torch.no_grad():
        (got,) = plan.run([x], planned.context_for(plan))
        expected: object = direct.run_entry("forward", [x])
    assert isinstance(expected, torch.Tensor)
    torch.testing.assert_close(got.float(), expected.float(), rtol=2e-2, atol=2e-2)
    findings = planned.conform("forward", [x], deepgemm, tolerance=5e-2)
    assert len(findings) == 1 and findings[0].passed


# ---- bitsandbytes: `linear_nf4` calls (`Nf4Linear`), one row by its 4-bit
# product, more over the weight it widens.

NF4 = """\
module tests.nf4

use std.quant::{Nf4Linear}

pub block Model<In: Dim, Out: Dim, T: Float = bf16>
where In % 64 == 0 {
    sub proj: Nf4Linear<In, Out, 64, T>

    pub entry forward<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        return proj.forward(x)
    }
}
"""
NF4_GENERICS: dict[str, int | str] = {"In": 512, "Out": 256}


@pytest.fixture
def nf4_source(tmp_path: Path) -> Path:
    path = tmp_path / "nf4.linnet"
    path.write_text(NF4, encoding="utf-8")
    return path


def test_bitsandbytes_takes_the_nf4_products(nf4_source: Path) -> None:
    bitsandbytes = _plugin("bitsandbytes", "BitsAndBytes")
    graph = _graph(nf4_source, "forward", {**NF4_GENERICS, "B": 4}, "equivalent")
    assert len(list(bitsandbytes.match(graph, HOPPER))) == 1


@requires_cuda
@pytest.mark.parametrize("rows", [1, 5])
def test_bitsandbytes_multiplies_as_the_entry_does(
    nf4_source: Path, tmp_path: Path, rows: int
) -> None:
    pytest.importorskip("bitsandbytes")
    bitsandbytes = _plugin("bitsandbytes", "BitsAndBytes")
    out_features, in_features = int(NF4_GENERICS["Out"]), int(NF4_GENERICS["In"])
    generator = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (out_features, in_features), generator=generator)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    absmax = torch.rand(out_features, in_features // 64, generator=generator) + 0.5
    bias = torch.randn(out_features, generator=generator).bfloat16()
    save_file(
        {"proj.weight": packed, "proj.absmax": absmax, "proj.bias": bias},
        str(tmp_path / "w.safetensors"),
    )
    planned, direct = (
        _compiled(nf4_source, NF4_GENERICS, tmp_path / "w.safetensors") for _ in range(2)
    )
    x = (torch.randn(rows, in_features, device="cuda") * 0.1).bfloat16()
    plan = planned.plan_for("forward", [x], backends=[bitsandbytes], measure=False)
    assert [step.backend for step in plan.steps].count("bitsandbytes") == 1
    with torch.no_grad():
        (got,) = plan.run([x], planned.context_for(plan))
        expected: object = direct.run_entry("forward", [x])
    assert isinstance(expected, torch.Tensor)
    torch.testing.assert_close(got.float(), expected.float(), rtol=2e-2, atol=2e-2)
    findings = planned.conform("forward", [x], bitsandbytes, tolerance=5e-2)
    assert len(findings) == 1 and findings[0].passed


# ---- FlashAttention: decode attention over the KV cache, in a plan replayed
# as one CUDA graph.

LLAMA_64: dict[str, int | str] = {
    "Vocab": 11,
    "H": 128,
    "Heads": 2,
    "KvHeads": 1,
    "Inner": 64,
    "Layers": 2,
    "Batch": 3,
    "MaxSeq": 16,
    "T": "bf16",
}


@pytest.fixture
def llama_weights(tmp_path: Path) -> Path:
    skeleton = load(LLAMA, generics=LLAMA_64, std_root=STDLIB)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        name.removeprefix("root."): (torch.randn(p.shape, generator=generator) * 0.3).bfloat16()
        for name, p in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return path


def test_flash_attn_takes_each_layers_decode_attention() -> None:
    flash = _plugin("flash-attn", "FlashAttention")
    ampere = Target("cuda", 0, "A100", (8, 0))
    assert len(list(flash.match(_graph(LLAMA, "decode", LLAMA_64, "equivalent"), ampere))) == 2


@requires_cuda
def test_flash_attn_attends_as_the_entry_does(llama_weights: Path) -> None:
    pytest.importorskip("flash_attn")
    flash = _plugin("flash-attn", "FlashAttention")
    planned, direct = (_compiled(LLAMA, LLAMA_64, llama_weights) for _ in range(2))
    tokens = torch.tensor([[3], [5], [7]], dtype=torch.int32, device="cuda")

    def position(step: int) -> torch.Tensor:
        return torch.tensor(step, dtype=torch.int32, device="cuda")

    plan = planned.plan_for("decode", [tokens, position(0)], backends=[flash], measure=False)
    assert [step.backend for step in plan.steps].count("flash-attn") == 2
    parameters = {
        name.removeprefix("root."): parameter.detach()
        for name, parameter in planned.named_parameters()
    }
    runner = Captured(plan, Context(plan.target, parameters))
    with torch.no_grad():
        for step in range(5):  # as it is, captured, then replayed
            (logits,) = runner([tokens, position(step)])
            expected: object = direct.run_entry("decode", [tokens, position(step)])
            assert isinstance(expected, torch.Tensor)
            torch.testing.assert_close(logits, expected, rtol=3e-2, atol=3e-2)


# ---- FlashAttention over prompts: packed passes, and pages.

# The serving test's model, in bf16; paged with pages of 256 positions,
# which FlashAttention 2 reads: page 0 takes the writes nobody reads.
SERVE_GENERICS: dict[str, int | str] = {
    "Vocab": 64,
    "H": 64,
    "Heads": 4,
    "Batch": 3,
    "MaxSeq": 48,
    "T": "bf16",
}
PAGED_GENERICS: dict[str, int | str] = {
    **SERVE_GENERICS,
    "Batch": 1,
    "MaxSeq": 768,
    "PageSize": 256,
}


def _serve_model(tmp_path: Path, generics: dict[str, int | str]) -> tuple[Path, Path]:
    from tests.test_serve import SOURCE as SERVE

    source = tmp_path / "serve.linnet"
    source.write_text(SERVE, encoding="utf-8")
    skeleton = load(source, generics=generics, std_root=STDLIB)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        name.removeprefix("root."): (torch.randn(p.shape, generator=generator) * 0.3).bfloat16()
        for name, p in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    weights = tmp_path / "serve.safetensors"
    save_file(tensors, str(weights))
    return source, weights


def _ints(values: list[int], *shape: int) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.int32, device="cuda")
    return tensor.reshape(shape) if shape else tensor


def test_flash_attn_takes_packed_and_paged_prompts(tmp_path: Path) -> None:
    flash = _plugin("flash-attn", "FlashAttention")
    ampere = Target("cuda", 0, "A100", (8, 0))
    source, _ = _serve_model(tmp_path, SERVE_GENERICS)
    packed = _graph(source, "prefill_packed", {**SERVE_GENERICS, "P": 16}, "fast")
    assert len(list(flash.match(packed, ampere))) == 1
    paged = {**PAGED_GENERICS, "P": 16, "Rows": 2, "Pages": 1}
    assert len(list(flash.match(_graph(source, "prefill_paged", paged, "fast"), ampere))) == 1
    small = {**paged, "PageSize": 64}  # pages FlashAttention 2 does not read
    assert list(flash.match(_graph(source, "prefill_paged", small, "fast"), ampere)) == []


@requires_cuda
def test_flash_attn_attends_packed_prompts_as_the_entry_does(tmp_path: Path) -> None:
    pytest.importorskip("flash_attn")
    flash = _plugin("flash-attn", "FlashAttention")
    source, weights = _serve_model(tmp_path, SERVE_GENERICS)
    planned, direct = (_compiled(source, SERVE_GENERICS, weights) for _ in range(2))
    # Prompts of 5 and 7 tokens, then 4 of padding, as the engine packs them.
    tokens = _ints([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 0, 0, 0, 0])
    rows = _ints([0] * 5 + [1] * 7 + [0] * 4)
    at = _ints([*range(5), *range(7), *[47] * 4])
    segments = _ints([0] * 5 + [1] * 7 + [-1] * 4)
    last = _ints([4, 11, 0])
    inputs = [tokens, rows, at, segments, last]
    plan = planned.plan_for("prefill_packed", inputs, backends=[flash], measure=False)
    assert [step.backend for step in plan.steps].count("flash-attn") == 1
    with torch.no_grad():
        (got,) = plan.run(inputs, planned.context_for(plan))
        expected: object = direct.run_entry("prefill_packed", inputs)
    assert isinstance(expected, torch.Tensor)
    torch.testing.assert_close(got.float(), expected.float(), rtol=3e-2, atol=3e-2)


@requires_cuda
def test_flash_attn_attends_over_pages_as_the_entry_does(tmp_path: Path) -> None:
    pytest.importorskip("flash_attn")
    flash = _plugin("flash-attn", "FlashAttention")
    source, weights = _serve_model(tmp_path, PAGED_GENERICS)
    planned, direct = (_compiled(source, PAGED_GENERICS, weights) for _ in range(2))
    # Two rows on pages 1 and 2: prompts of 6 and 8 tokens, 2 of padding.
    table = _ints([1, 2], 2, 1)
    lengths = (6, 8)
    tokens = _ints([7, 1, 2, 3, 4, 5, 9, 8, 7, 6, 5, 4, 3, 2, 0, 0])
    rows = _ints([0] * 6 + [1] * 8 + [0, 0])
    at = _ints([*range(6), *range(8), 0, 0])
    slots = _ints([256 + p for p in range(6)] + [512 + p for p in range(8)] + [0, 0])
    last = _ints([5, 13])
    prompts = [tokens, at, rows, slots, last, table]
    step = [_ints([1, 2], 2, 1), _ints(list(lengths)), table]
    with torch.no_grad():
        for entry, inputs in (("prefill_paged", prompts), ("decode_paged", step)):
            plan = planned.plan_for(entry, inputs, backends=[flash], measure=False)
            assert [s.backend for s in plan.steps].count("flash-attn") == 1, entry
            (got,) = plan.run(inputs, planned.context_for(plan))
            expected: object = direct.run_entry(entry, inputs)
            assert isinstance(expected, torch.Tensor)
            torch.testing.assert_close(got.float(), expected.float(), rtol=3e-2, atol=3e-2)
