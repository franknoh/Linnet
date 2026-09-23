"""The typed program (`linnet.ir`) and the diagrams drawn from it."""

from __future__ import annotations

from pathlib import Path

import pytest

from linnet import LinnetError, Program, diagram, ir, load_program

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"
LLAMA = REPO / "examples/05-llama/src/lib.linnet"


@pytest.fixture(scope="module")
def llama() -> Program:
    return load_program(LLAMA, std_root=STDLIB)


def test_program_is_typed_and_prints_like_the_language(llama: Program) -> None:
    root = llama.root_block
    assert llama.root.name == "Model" and root.module == "llama"
    layers = root.member("layers")
    assert layers.kind == "sub" and isinstance(layers.type, ir.ArrayType)
    element = layers.type.element
    assert isinstance(element, ir.NamedType) and element.name == "DecoderLayer"
    assert ir.format_type(layers.type) == (
        "[DecoderLayer<H, Heads, KvHeads, Inner, Batch, MaxSeq, T>; Layers]"
    )

    forward = llama.entry("forward")
    assert forward.kind == "entry" and forward.short_name == "forward"
    assert [p.name for p in forward.params] == ["tokens"]
    assert ir.format_signature(forward) == (
        "pub entry forward<B: Dim, S: Dim>(tokens: Tensor[B, S; i32]) -> Tensor[B, S, Vocab; T]"
    )
    assert {e.short_name for e in llama.entries()} >= {"forward", "decode", "generate"}
    with pytest.raises(LinnetError):
        llama.entry()  # six entries: one must be named

    text = str(llama)
    assert text.startswith("llama::Model<Vocab: Dim, H: Dim")
    assert "T: Float = bf16>" in text
    assert "  sub norm: RmsNorm<H, T>" in text

    manifest = {e.path: e for e in llama.manifest}
    k_proj = manifest["layers[*].attention.k_proj.weight"]
    assert k_proj.kind == "param" and not k_proj.optional
    assert ir.format_shape(k_proj.shape) == "KvHeads * (H / Heads), H"
    assert ir.format_dim(k_proj.repeat[0]) == "Layers"
    assert manifest["layers[*].attention.q_proj.bias"].optional
    assert manifest["layers[*].attention.cache_k"].kind == "state"


def test_program_dataclasses_are_immutable(llama: Program) -> None:
    with pytest.raises(AttributeError):
        llama.root_block.name = "Other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        llama.blocks["Model"] = llama.root_block  # type: ignore[index]


def test_format_dim_parenthesizes_by_precedence() -> None:
    h = ir.DimSymbol(0, "H")
    n = ir.DimSymbol(1, "N")
    two = 2
    assert ir.format_dim(ir.DimExpr("mul", (n, ir.DimExpr("floordiv", (h, n))))) == "N * (H / N)"
    assert ir.format_dim(ir.DimExpr("add", (ir.DimExpr("mul", (h, two)), n))) == "H * 2 + N"
    assert ir.format_dim(ir.DimExpr("mul", (ir.DimExpr("add", (h, n)), two))) == "(H + N) * 2"
    assert ir.format_dim(ir.DimExpr("floordiv", (h, ir.DimExpr("mul", (n, two))))) == "H / (N * 2)"
    assert ir.format_dim(ir.DimExpr("max", (h, n))) == "max(H, N)"


def test_substitution_rewrites_callee_types(llama: Program) -> None:
    forward = llama.entry("forward")
    calls = [op for op in forward.body.walk() if op.kind == "call"]
    linear = next(op for op in calls if str(op.attrs["callee"]).endswith("Linear.forward"))
    callee = llama.functions[str(linear.attrs["callee"])]
    substitution = ir.call_substitution(linear)
    result = ir.substitute(callee.results[0], substitution)
    # `Tensor[*S, Out; T]` in the callee is the caller's `Tensor[B, S, Vocab; T]`.
    assert ir.format_type(callee.results[0]) == "Tensor[*S, Out; T]"
    assert ir.format_type(result) == "Tensor[B, S, Vocab; T]"


def test_diagram_follows_the_block_structure(llama: Program) -> None:
    graph = diagram.build(llama, "forward")
    labels = [n.label for n in graph.nodes]
    assert labels[0] == "tokens" and labels[-1] == "result"
    assert "embedding.forward" in labels and "lm_head.forward" in labels
    loop = next(g for g in graph.groups if g.label.startswith("for "))
    assert loop.label == "for layer in layers"
    inside = [n for n in graph.nodes if n.group == loop.id]
    assert [n.label for n in inside] == ["layers[layer].forward"]
    edge_labels = {e.label for e in graph.edges}
    assert "i32[B, S]" in edge_labels and "T[B, S, H]" in edge_labels
    assert "T[B, S, Vocab]" in edge_labels

    expanded = diagram.build(llama, "forward", expand=1)
    labels = {n.label for n in expanded.nodes}
    # Inside the expanded layer, calls read relative to it and the residual adds show.
    assert "attention.forward" in labels and "mlp.forward" in labels and "+" in labels
    node = next(n for n in expanded.nodes if n.label == "attention.forward")
    assert node.sublabel.startswith("GroupedQueryAttention<H, Heads")
    # Types inside the callee are expressed in the caller's generics.
    assert all("*S" not in e.label for e in expanded.edges)


def test_diagram_renders_every_format(llama: Program, tmp_path: Path) -> None:
    svg = diagram.render(llama, "forward", format="svg", expand=1)
    assert svg.startswith("<svg") and "for layer in layers" in svg and "T[B, S, H]" in svg
    dark = diagram.render(llama, "decode", format="svg", theme="dark")
    assert "hsl(0, 0%, 2%)" in dark
    tikz = diagram.render(llama, "forward", format="tikz")
    assert "\\begin{tikzpicture}" in tikz and "\\end{tikzpicture}" in tikz
    assert "lm\\_head.forward" in tikz
    dot = diagram.render(llama, "forward", format="dot", expand=2)
    assert dot.startswith("digraph") and "subgraph cluster_" in dot

    out = tmp_path / "forward.svg"
    code = diagram.main([str(LLAMA), "--std", str(STDLIB), "--entry", "forward", "-o", str(out)])
    assert code == 0 and out.read_text(encoding="utf-8").startswith("<svg")
    assert diagram.main([str(LLAMA), "--std", str(STDLIB), "--entry", "missing"]) == 1


def test_state_entries_show_reads_and_writes(llama: Program) -> None:
    graph = diagram.build(llama, "decode", expand=2)
    kinds = {n.kind for n in graph.nodes}
    assert "state" in kinds
    labels = [n.label for n in graph.nodes if n.kind == "state"]
    assert any(label.endswith("cache_k") for label in labels)
