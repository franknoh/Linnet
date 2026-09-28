"""Weight-only work (`linnet torch --prepare`, `linnet jax --prepare`): what
reads nothing but parameters runs once, in `prepare`, and every entry that
computes the same thing shares the result -- with the same numbers as when
it runs on every call. Views of a weight are not prepared (that would copy
the weights), and a model being trained keeps everything in the graph."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.compiler import find_compiler
from linnet.torch import CompiledLinnetModule, load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.prepare

pub block Model<E: Dim, I: Dim, T: Float = f32> {
    param blocks: Tensor[E, I; T]
    param scale: Tensor[E; T]

    // Two entries over the same weight-only value, as a model's prompt and
    // step entries dequantize the same experts.
    pub entry first(x: Tensor[I; T]) -> Tensor[E; T] {
        let w = weights()
        let y[e] = sum[i] w[e, i] * x[i]
        return y
    }

    pub entry second(x: Tensor[I; T]) -> Tensor[E; T] {
        let w = weights()
        let y[e] = sum[i] w[e, i] * x[i] * 2.0
        return y
    }

    // Only a view of a weight: nothing to prepare.
    pub entry viewed(x: Tensor[E; T]) -> Tensor[I; T] {
        let t = permute(blocks, [1, 0])
        let y[i] = sum[e] t[i, e] * x[e]
        return y
    }

    fn weights() -> Tensor[E, I; T] {
        let w[e, i] = exp(blocks[e, i] * scale[e]) + 1.0
        return w
    }
}
"""

GENERICS: dict[str, int | str] = {"E": 3, "I": 5}


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path, dict[str, np.ndarray]]:
    source = tmp_path / "prepare.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    rng = np.random.default_rng(0)
    arrays = {
        "blocks": rng.standard_normal((3, 5)).astype(np.float32) * 0.3,
        "scale": rng.standard_normal(3).astype(np.float32),
    }
    weights = tmp_path / "model.safetensors"
    save_file({k: torch.tensor(v) for k, v in arrays.items()}, str(weights))
    return source, weights, arrays


def _expected(arrays: dict[str, np.ndarray], x: np.ndarray, entry: str) -> np.ndarray:
    w = np.exp(arrays["blocks"] * arrays["scale"][:, None]) + 1.0
    return w @ x * (2.0 if entry == "second" else 1.0)


def _source(source: Path, entry: str, prepare: bool) -> str:
    command = [find_compiler(), "torch", "--root", "Model", "--entry", entry]
    command += [arg for k, v in GENERICS.items() for arg in ("--bind", f"{k}={v}")]
    command += ["--std", str(STDLIB), *(["--prepare"] if prepare else []), str(source)]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def test_weight_only_work_moves_into_prepare(
    files: tuple[Path, Path, dict[str, np.ndarray]],
) -> None:
    source, _, _ = files
    text = _source(source, "first", prepare=True)
    main = text[text.index("def main(") :]
    assert "def prepare(" in text and "torch.exp(" not in main
    assert "def prepare(" not in _source(source, "first", prepare=False)
    # A permuted weight is a view: preparing it would only copy the weight.
    assert "def prepare(" not in _source(source, "viewed", prepare=True)


def test_torch_shares_prepared_values(files: tuple[Path, Path, dict[str, np.ndarray]]) -> None:
    source, weights, arrays = files
    model = load(source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    assert isinstance(model, CompiledLinnetModule)
    x = np.arange(5, dtype=np.float32) * 0.1
    for entry in ("first", "second"):
        got = model.run_entry(entry, [torch.tensor(x)]).numpy()
        np.testing.assert_allclose(got, _expected(arrays, x, entry), rtol=1e-5, atol=1e-5)
    # Both entries prepare the same `weights()`: computed once, kept once.
    assert len(model._prepared) == 1  # pyright: ignore[reportPrivateUsage]


def test_training_keeps_weight_work_in_the_graph(
    files: tuple[Path, Path, dict[str, np.ndarray]],
) -> None:
    source, weights, _ = files
    model = load(
        source, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True, trainable=True
    )
    x = torch.arange(5, dtype=torch.float32) * 0.1
    assert isinstance(model, CompiledLinnetModule)
    model.run_entry("first", [x]).sum().backward()
    assert "def prepare(" not in model.generated_source("first")
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.count_nonzero(g) > 0 for g in grads)


def test_jax_shares_prepared_values(files: tuple[Path, Path, dict[str, np.ndarray]]) -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from linnet.jax import load_model

    source, weights, arrays = files
    model = load_model(source, generics=GENERICS, weights=weights, std_root=STDLIB)
    x = np.arange(5, dtype=np.float32) * 0.1
    for entry in ("first", "second"):
        got = np.asarray(model.run_entry(entry, [jnp.asarray(x)]))
        np.testing.assert_allclose(got, _expected(arrays, x, entry), rtol=1e-5, atol=1e-5)
    assert len(model._prepared) == 1  # pyright: ignore[reportPrivateUsage]
