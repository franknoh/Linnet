"""ONNX graphs import as Linnet source that checks, formats, and — loaded
back through the PyTorch adapter with the graph's initializers — computes
what the graph computes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import onnx  # type: ignore[import-untyped]
import pytest
import torch
from linnet_torch import load
from onnx import TensorProto, helper, numpy_helper  # type: ignore[import-untyped]

from linnet_onnx import OnnxImportError, find_compiler, import_onnx

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]


def _compiler_ok(*args: str) -> None:
    completed = subprocess.run(
        [find_compiler(), *args], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _tensor(name: str, array: np.ndarray[Any, Any]) -> Any:
    return numpy_helper.from_array(array, name=name)


def _transformer_graph(
    vocab: int, width: int, heads: int, layers: int
) -> tuple[Any, dict[str, Any]]:
    """A GPT-style block built by hand: embeddings, layer norm, attention
    with a constant causal mask, a tanh GELU MLP, and a tied output head.
    The batch and sequence dimensions are symbolic."""
    rng = np.random.default_rng(0)
    w: dict[str, np.ndarray[Any, Any]] = {
        "wte": rng.normal(0, 0.3, (vocab, width)).astype(np.float32)
    }
    nodes: list[Any] = []
    inits: list[Any] = [_tensor("wte", w["wte"])]
    for i in range(layers):
        p = f"blocks.{i}."
        for name, shape in (
            ("ln1.weight", (width,)),
            ("ln1.bias", (width,)),
            ("attn.qkv.weight", (width, 3 * width)),
            ("attn.qkv.bias", (3 * width,)),
            ("attn.out.weight", (width, width)),
            ("attn.out.bias", (width,)),
            ("ln2.weight", (width,)),
            ("ln2.bias", (width,)),
            ("mlp.up.weight", (width, 4 * width)),
            ("mlp.up.bias", (4 * width,)),
            ("mlp.down.weight", (4 * width, width)),
            ("mlp.down.bias", (width,)),
        ):
            w[p + name] = rng.normal(0, 0.3, shape).astype(np.float32)
            inits.append(_tensor(p + name, w[p + name]))
    w["lnf.weight"] = np.ones((width,), np.float32)
    w["lnf.bias"] = np.zeros((width,), np.float32)
    inits += [_tensor("lnf.weight", w["lnf.weight"]), _tensor("lnf.bias", w["lnf.bias"])]
    # Small integer constants for shapes and slices.
    consts = {
        "c_qkv_split": np.array([0, width, 2 * width, 3 * width], np.int64),
        "c_axis2": np.array([2], np.int64),
        "c_scale": np.array(1.0 / np.sqrt(width // heads), np.float32),
        "c_neg": np.array(-1e30, np.float32),
        "c_heads": np.array([heads], np.int64),
        "c_dh": np.array([width // heads], np.int64),
        "c_width": np.array([width], np.int64),
        "c_zero": np.array([0], np.int64),
        "c_one": np.array([1], np.int64),
    }
    for name, array in consts.items():
        inits.append(_tensor(name, array))

    nodes.append(helper.make_node("Gather", ["wte", "tokens"], ["x0"], axis=0))
    # Shape arithmetic: [B, S, H] -> [B, S, heads, dh] -> [B, heads, S, dh].
    nodes.append(helper.make_node("Shape", ["x0"], ["shape"]))
    nodes.append(helper.make_node("Gather", ["shape", "c_zero"], ["dim_b"], axis=0))
    nodes.append(helper.make_node("Gather", ["shape", "c_one"], ["dim_s"], axis=0))
    nodes.append(
        helper.make_node("Concat", ["dim_b", "dim_s", "c_heads", "c_dh"], ["split_shape"], axis=0)
    )
    nodes.append(helper.make_node("Concat", ["dim_b", "dim_s", "c_width"], ["merge_shape"], axis=0))
    x = "x0"
    for i in range(layers):
        p = f"blocks.{i}."
        h = f"h{i}"
        nodes.append(
            helper.make_node(
                "LayerNormalization",
                [x, p + "ln1.weight", p + "ln1.bias"],
                [h],
                axis=-1,
                epsilon=1e-5,
            )
        )
        nodes.append(helper.make_node("MatMul", [h, p + "attn.qkv.weight"], [f"qkv{i}_"]))
        nodes.append(helper.make_node("Add", [f"qkv{i}_", p + "attn.qkv.bias"], [f"qkv{i}"]))
        for j, part in enumerate("qkv"):
            nodes.append(
                helper.make_node(
                    "Slice", [f"qkv{i}", f"s{j}", f"s{j + 1}", "c_axis2"], [f"{part}{i}_flat"]
                )
            )
            nodes.append(
                helper.make_node(
                    "Reshape", [f"{part}{i}_flat", "split_shape"], [f"{part}{i}_split"]
                )
            )
            nodes.append(
                helper.make_node(
                    "Transpose", [f"{part}{i}_split"], [f"{part}{i}"], perm=[0, 2, 1, 3]
                )
            )
        nodes.append(helper.make_node("Transpose", [f"k{i}"], [f"kt{i}"], perm=[0, 1, 3, 2]))
        nodes.append(helper.make_node("MatMul", [f"q{i}", f"kt{i}"], [f"score{i}_"]))
        nodes.append(helper.make_node("Mul", [f"score{i}_", "c_scale"], [f"score{i}"]))
        nodes.append(helper.make_node("Where", ["mask", f"score{i}", "c_neg"], [f"masked{i}"]))
        nodes.append(helper.make_node("Softmax", [f"masked{i}"], [f"weights{i}"], axis=-1))
        nodes.append(helper.make_node("MatMul", [f"weights{i}", f"v{i}"], [f"mixed{i}"]))
        nodes.append(
            helper.make_node("Transpose", [f"mixed{i}"], [f"mixed{i}_t"], perm=[0, 2, 1, 3])
        )
        nodes.append(helper.make_node("Reshape", [f"mixed{i}_t", "merge_shape"], [f"merged{i}"]))
        nodes.append(
            helper.make_node("MatMul", [f"merged{i}", p + "attn.out.weight"], [f"proj{i}_"])
        )
        nodes.append(helper.make_node("Add", [f"proj{i}_", p + "attn.out.bias"], [f"proj{i}"]))
        nodes.append(helper.make_node("Add", [x, f"proj{i}"], [f"res{i}"]))
        nodes.append(
            helper.make_node(
                "LayerNormalization",
                [f"res{i}", p + "ln2.weight", p + "ln2.bias"],
                [f"h2_{i}"],
                axis=-1,
                epsilon=1e-5,
            )
        )
        nodes.append(helper.make_node("MatMul", [f"h2_{i}", p + "mlp.up.weight"], [f"up{i}_"]))
        nodes.append(helper.make_node("Add", [f"up{i}_", p + "mlp.up.bias"], [f"up{i}"]))
        nodes.append(helper.make_node("Gelu", [f"up{i}"], [f"act{i}"], approximate="tanh"))
        nodes.append(helper.make_node("MatMul", [f"act{i}", p + "mlp.down.weight"], [f"down{i}_"]))
        nodes.append(helper.make_node("Add", [f"down{i}_", p + "mlp.down.bias"], [f"down{i}"]))
        nodes.append(helper.make_node("Add", [f"res{i}", f"down{i}"], [f"x{i + 1}"]))
        x = f"x{i + 1}"
    for j in range(4):
        inits.append(_tensor(f"s{j}", np.array([j * width], np.int64)))
    nodes.append(
        helper.make_node(
            "LayerNormalization", [x, "lnf.weight", "lnf.bias"], ["final"], axis=-1, epsilon=1e-5
        )
    )
    nodes.append(helper.make_node("Transpose", ["wte"], ["wte_t"], perm=[1, 0]))
    nodes.append(helper.make_node("MatMul", ["final", "wte_t"], ["logits"]))

    graph = helper.make_graph(
        nodes,
        "gpt",
        [
            helper.make_tensor_value_info("tokens", TensorProto.INT32, ["batch", "seq"]),
            helper.make_tensor_value_info("mask", TensorProto.BOOL, ["seq", "seq"]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "seq", vocab])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model, w


def _reference(
    w: dict[str, Any], tokens: torch.Tensor, mask: torch.Tensor, heads: int, layers: int
) -> torch.Tensor:
    t = {k: torch.from_numpy(v) for k, v in w.items()}
    B, S = tokens.shape  # noqa: N806
    width = t["wte"].shape[1]
    dh = width // heads

    def ln(x: torch.Tensor, p: str) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, [width], t[p + ".weight"], t[p + ".bias"], 1e-5)

    x = t["wte"][tokens.long()]
    for i in range(layers):
        p = f"blocks.{i}."
        h = ln(x, p + "ln1")
        qkv = h @ t[p + "attn.qkv.weight"] + t[p + "attn.qkv.bias"]
        q, k, v = (
            qkv[..., j * width : (j + 1) * width].reshape(B, S, heads, dh).permute(0, 2, 1, 3)
            for j in range(3)
        )
        score = (q @ k.transpose(-1, -2)) / np.sqrt(dh)
        score = torch.where(mask, score, torch.full_like(score, -1e30))
        mixed = (torch.softmax(score, -1) @ v).permute(0, 2, 1, 3).reshape(B, S, width)
        x = x + mixed @ t[p + "attn.out.weight"] + t[p + "attn.out.bias"]
        h = ln(x, p + "ln2")
        up = torch.nn.functional.gelu(
            h @ t[p + "mlp.up.weight"] + t[p + "mlp.up.bias"], approximate="tanh"
        )
        x = x + up @ t[p + "mlp.down.weight"] + t[p + "mlp.down.bias"]
    return ln(x, "lnf") @ t["wte"].T


def test_hand_built_transformer_round_trips(tmp_path: Path) -> None:
    vocab, width, heads, layers = 11, 8, 2, 2
    model, w = _transformer_graph(vocab, width, heads, layers)
    path = tmp_path / "gpt.onnx"
    onnx.save(model, str(path))
    result = import_onnx(
        path, output=tmp_path / "src/gpt.linnet", weights=tmp_path / "weights", std_root=STDLIB
    )
    source = result.source.read_text()
    assert "pub entry forward<batch: Dim, seq: Dim>" in source
    assert "sub blocks: [Block; 2]" in source
    assert "layer_norm<" in source and "softmax<" in source
    _compiler_ok("lint", "--std", str(STDLIB), str(result.source))
    _compiler_ok("fmt", "--check", str(result.source))

    module = load(
        result.source,
        generics={},
        std_root=STDLIB,
        weights=tmp_path / "weights",
        bindings=result.bindings,
    )
    for batch, seq in [(2, 5), (1, 3)]:
        tokens = torch.randint(0, vocab, (batch, seq), dtype=torch.int32)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        torch.testing.assert_close(
            module(tokens, mask), _reference(w, tokens, mask, heads, layers), atol=1e-4, rtol=1e-4
        )

    again = import_onnx(path, output=tmp_path / "again/gpt.linnet", std_root=STDLIB)
    assert again.source.read_text() == source


def test_torch_exported_mlp_imports(tmp_path: Path) -> None:
    """A model exported by PyTorch's own ONNX exporter imports, and the
    Linnet version agrees with the module."""
    pytest.importorskip("onnxscript")
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    model = torch.nn.Sequential(
        torch.nn.Linear(6, 10),
        torch.nn.LayerNorm(10),
        torch.nn.GELU("tanh"),
        torch.nn.Linear(10, 3),
    )
    x = torch.randn(4, 6)
    path = tmp_path / "mlp.onnx"
    torch.onnx.export(model, (x,), str(path), dynamo=True)  # pyright: ignore[reportUnknownMemberType]
    result = import_onnx(
        path, output=tmp_path / "mlp.linnet", weights=tmp_path / "w", std_root=STDLIB
    )
    module = load(
        result.source,
        generics={},
        std_root=STDLIB,
        weights=tmp_path / "w",
        bindings=result.bindings,
    )
    torch.testing.assert_close(module(x), model(x), atol=1e-5, rtol=1e-5)


def test_unsupported_operations_are_named(tmp_path: Path) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Erf", ["x_scaled"], ["y"]),
            helper.make_node("Mul", ["x", "w"], ["x_scaled"]),
        ][::-1],
        "odd",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        initializer=[_tensor("w", np.ones((4,), np.float32))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    with pytest.raises(OnnxImportError, match="Erf"):
        import_onnx(model, output=tmp_path / "odd.linnet", std_root=STDLIB)
