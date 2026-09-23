# Python package

`linnet-lang` is one package, `linnet`, with a backend per framework behind
an extra. The core needs only NumPy: it runs the compiler, reads plans and
SafeTensors checkpoints, and gives you the compiled program as typed objects.

```bash
pip install "linnet-lang[torch]"     # linnet.torch
pip install "linnet-lang[jax]"       # linnet.jax
pip install "linnet-lang[flax]"      # linnet.jax.load_nnx
pip install "linnet-lang[onnx]"      # linnet.onnx
```

| Module | |
| --- | --- |
| `linnet` | `load_program`, `Program`, `compile_plan`, checkpoint reading (`read_arrays`, `read_bindings`), the compiler subprocess |
| `linnet.ir` | the typed program: blocks, members, functions, regions, operations, symbolic dimensions and types |
| `linnet.diagram` | architecture diagrams as SVG, TikZ, or Graphviz |
| `linnet.nest` | [Nest](nest.md), the model zoo: cards, checks, `nest.load` |
| `linnet.triton` | [Triton Inference Server](integrations.md) model repositories |
| `linnet.torch` | [PyTorch](torch.md): `load`, `bind_weights`, `export_linnet` |
| `linnet.jax` | [JAX and Flax](jax.md): `load`, `load_source`, `load_nnx`, `export_linnet`, `import_stablehlo` |
| `linnet.onnx` | [ONNX](onnx.md): `import_onnx` |

The compiler binary comes from `LINNET_BIN` or `PATH`. In the repository,
`cd python/linnet && uv sync --all-extras` installs every backend and the
test dependencies.

## The typed program

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

`load_program` runs `linnet plan` and reads the result into frozen
dataclasses (`linnet.ir`). Nothing executes. Dimensions stay symbolic:

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

| Type | Cases |
| --- | --- |
| `Dim` | `int`, `DimSymbol`, `PackSize`, `DimExpr(op, args)` with `add`, `mul`, `floordiv`, `mod`, `min`, `max` |
| `Unit` | a `Dim` or a `Pack` (`*S`) |
| `DType` | a name such as `"bf16"`, or `DTypeVar` |
| `Type` | `ScalarType`, `TensorType`, `TupleType`, `OptionalType`, `ArrayType`, `NamedType` (block, struct, enum), `ShapeType`, `UnitType` |

`Program` holds `root`, `manifest`, `blocks`, `functions`, and `constants`;
`Function` has `generics`, `params`, `results`, `states`, and a `body`
`Region` of `Op`s. `ir.substitute(type, ir.call_substitution(op))` rewrites a
callee's types in the caller's generics. `format_dim`, `format_shape`,
`format_type`, and `format_signature` print everything as the language
spells it.

## Diagrams

```bash
python -m linnet.diagram examples/05-llama/src/lib.linnet --std stdlib \
    --entry forward --expand 1 -o forward.svg
```

`linnet.diagram` draws one entry as a dataflow graph: inputs, the calls and
operations of its body, loops and expanded calls as nested groups, outputs.
Every edge is labelled with the tensor type the compiler inferred at that
point, in the caller's generics (`T[B, S, H]`), because the plan carries it.

| Option | |
| --- | --- |
| `--expand N` | inline calls to sub-block methods `N` levels deep as groups; `0` shows the block structure only |
| `--params` | add a node per parameter read |
| `--no-arithmetic` | fold elementwise operations into the edges around them |
| `--format svg\|tikz\|dot` | the output; inferred from `-o`'s extension (`.svg`, `.tex`, `.dot`) |
| `--theme light\|dark` | SVG colours |

From Python, `diagram.build(program, "forward", expand=1)` returns the
`Graph` (nodes, edges, groups), and `to_svg`, `to_tikz`, `to_dot` render it.
The TikZ output needs `\usetikzlibrary{fit, arrows.meta}` and lays the nodes
out itself, so it drops into a paper without Graphviz; the DOT output hands
the layout to Graphviz instead.
