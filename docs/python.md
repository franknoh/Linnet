# Python package

`linnet-lang` is a NumPy-only core that runs the compiler and exposes the
program as typed objects, plus a backend per framework behind an extra.

## Install

```bash
cd python/linnet
pip install ".[torch]"     # linnet.torch
pip install ".[jax]"       # linnet.jax
pip install ".[flax]"      # linnet.jax.load_nnx
pip install ".[onnx]"      # linnet.onnx (add onnxruntime or onnxruntime-gpu to run models)
```

It is not on PyPI yet; outside a checkout, install
`"linnet-lang[torch] @ git+https://github.com/franknoh/Linnet#subdirectory=python/linnet"`.
The compiler comes from `LINNET_BIN` or `PATH`. In a checkout,
`uv sync --all-extras` installs every backend and test dependency.

## Modules

| Module | Contents |
| --- | --- |
| `linnet` | `load_program`, `Program`, `compile_plan`, `read_arrays`, `read_bindings` |
| `linnet.ir`, `linnet.diagram` | [typed program](#typed-program), [diagrams](#diagrams) |
| `linnet.nest` | [Nest](nest.md) model zoo |
| `linnet.torch`, `linnet.jax`, `linnet.onnx` | backends for [PyTorch](torch.md), [JAX and Flax](jax.md), [ONNX](onnx.md) |
| `linnet.triton`, `linnet.hf`, `linnet.gguf` | [integrations](integrations.md): Triton Inference Server, Transformers (vLLM, SGLang, TGI), GGUF and Ollama |
| `linnet.train` | [supervised fine-tuning](training.md#supervised-fine-tuning) over packed sequences; `linnet.train.grpo`, [reinforcement learning](training.md#reinforcement-learning); `linnet.train.dpo`, [preferences](training.md#preferences) |

## Typed program

```python
from linnet import load_program

program = load_program("examples/05-llama/src/lib.linnet", std_root="stdlib")
print(program)
```

```text
llama::Model<Vocab: Dim, H: Dim, Heads: Dim, KvHeads: Dim, Inner: Dim, Layers: Dim, Batch: Dim, MaxSeq: Dim, T: Float = bf16>
  sub embedding: Embedding<Vocab, H, T>
  sub layers: [DecoderLayer<H, Heads, KvHeads, Inner, Batch, MaxSeq, T>; Layers]
  sub norm: RmsNorm<H, T>
  sub lm_head: Linear<H, Vocab, T>
  pub entry forward<B: Dim, S: Dim>(tokens: Tensor[B, S; i32]) -> Tensor[B, S, Vocab; T]
  pub entry decode(token: Tensor[Batch, 1; i32], pos: i32) -> Tensor[Batch, Vocab; T]
  ...
```

`load_program` reads `linnet plan` output into frozen `linnet.ir`
dataclasses; nothing executes. Dimensions stay symbolic:

```python
from linnet import ir

entry = program.entry("forward")
ir.format_signature(entry)
# 'pub entry forward<B: Dim, S: Dim>(tokens: Tensor[B, S; i32]) -> Tensor[B, S, Vocab; T]'

weight = next(e for e in program.manifest if e.path.endswith("k_proj.weight"))
ir.format_shape(weight.shape)        # 'KvHeads * (H / Heads), H'
weight.repeat                        # (DimSymbol(id=21, name='Layers'),)

for op in entry.body.walk():         # every operation, nested regions included
    if op.kind == "call":
        print(op.attrs["callee"], [ir.format_type(r.type) for r in op.results])
```

| Class | Fields or cases |
| --- | --- |
| `Program` | `root`, `manifest`, `blocks`, `functions`, `constants` |
| `Function` | `generics`, `params`, `results`, `states`, `body` (a `Region` of `Op`s) |
| `Dim` | `int`, `DimSymbol`, `PackSize`, `DimExpr(op, args)` with `add`, `mul`, `floordiv`, `mod`, `min`, `max` |
| `Unit` | a `Dim` or a `Pack` (`*S`) |
| `DType` | a name such as `"bf16"`, or `DTypeVar` |
| `Type` | `ScalarType`, `TensorType`, `TupleType`, `OptionalType`, `ArrayType`, `NamedType` (block, struct, enum), `ShapeType`, `UnitType` |

`ir.substitute(type, ir.call_substitution(op))` rewrites a callee's types
in the caller's generics. `format_dim`, `format_shape`, `format_type` and
`format_signature` print Linnet syntax.

## Diagrams

```bash
python -m linnet.diagram examples/05-llama/src/lib.linnet --std stdlib \
    --entry forward --expand 1 -o forward.svg
```

`linnet.diagram` draws an entry as a dataflow graph, with each edge's
inferred tensor type (`T[B, S, H]`).

| Option | Effect |
| --- | --- |
| `--expand N` | inline sub-block calls `N` levels deep; `0` shows blocks only |
| `--params` | a node per parameter read |
| `--no-arithmetic` | elementwise operations folded into edges |
| `--format svg\|tikz\|dot` | inferred from `-o` (`.svg`, `.tex`, `.dot`) |
| `--theme light\|dark` | SVG colours |

`diagram.build(program, "forward", expand=1)` returns a `Graph` for
`to_svg`, `to_tikz` or `to_dot`. TikZ output needs
`\usetikzlibrary{fit, arrows.meta}`; DOT needs Graphviz.
