"""`linnet stablehlo` output runs under XLA and agrees with the PyTorch
materializer on the same weights: Linnet semantics are not Torch-specific."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet_torch import load
from linnet_torch.plan import find_compiler

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"
EXAMPLES = REPO / "examples"

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)  # `i64` inputs (PRNG keys) stay 64-bit on device


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]


def _export(source: Path, bindings: dict[str, str | int], entry: str | None = None) -> str:
    command = [find_compiler(), "stablehlo", "--std", str(STDLIB)]
    if entry is not None:
        command += ["--entry", entry]
    for name, value in bindings.items():
        command += ["--bind", f"{name}={value}"]
    completed = subprocess.run([*command, str(source)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _run_xla(text: str, arguments: list[Any]) -> Any:
    """Compiles the MLIR text with XLA's CPU client and runs it once."""
    import importlib

    import jax.extend as jex

    # jaxlib's client binding has no type information; it is used as Any.
    options: Any = importlib.import_module("jaxlib._jax")
    backend: Any = cast(Any, jex.backend).get_backend()
    device: Any = backend.local_devices()[0]
    executable: Any = backend.compile_and_load(
        text, options.DeviceList((device,)), options.CompileOptions()
    )
    buffers = [jax.device_put(argument, device) for argument in arguments]
    outputs: Any = executable.execute(buffers)
    return np.asarray(outputs[0])


def _parameter_paths(text: str) -> list[str]:
    return re.findall(r'linnet\.path = "([^"]+)"', text)


def _round_trip(
    source: Path,
    generics: dict[str, int | str],
    entry_dims: dict[str, int],
    inputs: list[torch.Tensor],
    tmp_path: Path,
) -> None:
    reference = load(source, generics=generics, std_root=STDLIB)
    weights: dict[str, torch.Tensor] = {}
    for name, parameter in reference.named_parameters():
        path = name.removeprefix("root.")
        if not path.endswith(".bias"):
            weights[path] = torch.randn(parameter.shape) * 0.3
    save_file(weights, str(tmp_path / "model.safetensors"))
    reference = load(source, generics=generics, std_root=STDLIB, weights=tmp_path)
    expected = reference(*inputs)

    text = _export(source, {**generics, **entry_dims})
    paths = _parameter_paths(text)
    assert sorted(paths) == sorted(weights)
    arguments = [tensor.numpy() for tensor in inputs] + [weights[path].numpy() for path in paths]
    actual = _run_xla(text, arguments)
    torch.testing.assert_close(torch.tensor(np.array(actual)), expected, atol=1e-4, rtol=1e-4)


def test_block_model_matches_torch(tmp_path: Path) -> None:
    generics: dict[str, int | str] = {"H": 8, "Inner": 16, "Layers": 2, "Vocab": 12, "T": "f32"}
    tokens = torch.randint(0, 12, (2, 3), dtype=torch.int32)
    _round_trip(
        EXAMPLES / "03-block-and-weights/model.linnet",
        generics,
        {"B": 2, "S": 3},
        [tokens],
        tmp_path,
    )


def test_tiny_transformer_matches_torch(tmp_path: Path) -> None:
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 2,
        "Inner": 16,
        "Layers": 2,
        "T": "f32",
    }
    seq, head = 5, 4
    tokens = torch.randint(0, 11, (2, seq), dtype=torch.int32)
    positions = torch.arange(seq, dtype=torch.float32)[:, None]
    frequencies = 1.0 / (10000 ** (torch.arange(0, head, 2, dtype=torch.float32) / head))
    angles = positions * frequencies
    cos_table = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin_table = torch.cat([angles.sin(), angles.sin()], dim=-1)
    _round_trip(
        EXAMPLES / "04-tiny-transformer/src/lib.linnet",
        generics,
        {"B": 2, "S": seq},
        [tokens, cos_table, sin_table],
        tmp_path,
    )


def test_llama_next_token_matches_torch(tmp_path: Path) -> None:
    """Grouped-query attention, rotary tables from `iota`, and a slice of the
    last position, through a named entry."""
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 4,
        "KvHeads": 2,
        "Inner": 16,
        "Layers": 2,
        "Batch": 2,
        "MaxSeq": 8,
        "T": "f32",
    }
    source = EXAMPLES / "05-llama/src/lib.linnet"
    reference = load(source, generics=generics, std_root=STDLIB)
    weights: dict[str, torch.Tensor] = {}
    for name, parameter in reference.named_parameters():
        path = name.removeprefix("root.")
        if not path.endswith(".bias"):
            weights[path] = torch.randn(parameter.shape) * 0.3
    save_file(weights, str(tmp_path / "model.safetensors"))
    reference = load(source, generics=generics, std_root=STDLIB, weights=tmp_path)
    tokens = torch.randint(0, 11, (2, 5), dtype=torch.int32)
    expected = reference.run_entry("next_token", [tokens])

    text = _export(source, {**generics, "B": 2, "S": 5}, entry="next_token")
    paths = _parameter_paths(text)
    arguments = [tokens.numpy()] + [weights[path].numpy() for path in paths]
    actual = _run_xla(text, arguments)
    torch.testing.assert_close(torch.tensor(np.array(actual)), expected, atol=1e-4, rtol=1e-4)


def test_unbound_generic_is_reported() -> None:
    completed = subprocess.run(
        [
            find_compiler(),
            "stablehlo",
            "--std",
            str(STDLIB),
            "--bind",
            "H=8",
            str(EXAMPLES / "03-block-and-weights/model.linnet"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "needs a value" in completed.stderr


def test_llama_sample_draws_the_same_tokens_under_xla(tmp_path: Path) -> None:
    """`std.random` is arithmetic, so the sampled tokens for a key are the
    same in XLA as in the PyTorch interpreter."""
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 4,
        "KvHeads": 2,
        "Inner": 16,
        "Layers": 2,
        "Batch": 2,
        "MaxSeq": 8,
        "T": "f32",
    }
    source = EXAMPLES / "05-llama/src/lib.linnet"
    reference = load(source, generics=generics, std_root=STDLIB)
    weights: dict[str, torch.Tensor] = {}
    for name, parameter in reference.named_parameters():
        path = name.removeprefix("root.")
        if not path.endswith(".bias"):
            weights[path] = torch.randn(parameter.shape) * 0.3
    save_file(weights, str(tmp_path / "model.safetensors"))
    reference = load(source, generics=generics, std_root=STDLIB, weights=tmp_path)
    prompt = torch.randint(0, 11, (2, 1), dtype=torch.int32)
    key = torch.tensor([0, 7], dtype=torch.int64)
    temperature = torch.tensor(0.8)
    start = torch.tensor(0, dtype=torch.int32)
    expected = reference.run_entry(
        "sample", [prompt, start, key, temperature], generics={"Steps": 3}
    )

    text = _export(source, {**generics, "Steps": 3}, entry="sample")
    paths = _parameter_paths(text)
    arguments = [prompt.numpy(), start.numpy(), key.numpy(), temperature.numpy()]
    arguments += [weights[path].numpy() for path in paths]
    # The caches are read before written: zeros, as a fresh module holds.
    for _ in re.findall(r'linnet\.state = "([^"]+)"', text):
        arguments.append(np.zeros((2, 2, 8, 2), np.float32))
    actual = _run_xla(text, arguments)
    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())


def test_while_loop_runs_under_xla(tmp_path: Path) -> None:
    """A `while` exports as `stablehlo.while` and iterates as the interpreter
    does: here until an element reaches 1000 or the limit is hit."""
    source = REPO / "spec-tests/valid/023_while.linnet"
    reference = load(source, generics={"N": 3}, std_root=STDLIB)
    scale = torch.tensor([2.0, 3.0, 1.5])
    reference.load_state_dict({"root.scale": scale})
    x = torch.ones(3)
    for limit in (3, 100):
        expected = reference(x, torch.tensor(limit, dtype=torch.int32))
        text = _export(source, {"N": 3})
        assert '"stablehlo.while"' in text
        actual = _run_xla(text, [x.numpy(), np.int32(limit), scale.numpy()])
        torch.testing.assert_close(torch.tensor(np.array(actual)), expected)
