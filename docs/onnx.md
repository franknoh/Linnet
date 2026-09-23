# Importing ONNX graphs

`python/linnet_onnx` turns an ONNX graph into Linnet source.

```bash
cd python/linnet_onnx
uv sync --all-extras
```

```python
from linnet_onnx import import_onnx

result = import_onnx("model.onnx", output="src/model.linnet", weights="weights/", std_root="stdlib")
print(result.notes)  # anything the translation had to drop
```

`import_onnx` runs ONNX shape inference, then translates the graph node by
node into a Core IR plan (docs/plan-format.md), which `linnet emit` prints as
formatted source; the result is checked before it is written.

- **Parameters.** Tensor initializers become `param`s. Their dotted names form
  the block hierarchy (`blocks.0.attn.qkv.weight` → `blocks: [Block; N]`,
  `attn: Attn`, `param qkv...`); consecutive numbered children with the same
  structure become a sub array, so parameter paths match the ONNX names. With
  `weights=`, the initializers are saved as SafeTensors under their ONNX
  names, plus `bindings.json` when a path had to change.
- **Inputs.** Graph inputs become the entry's inputs; named symbolic
  dimensions (`dim_param`) become the entry's generic parameters, so a graph
  exported with a dynamic batch and sequence imports as `forward<batch: Dim,
  seq: Dim>`.
- **Shape arithmetic.** `Shape`, `Gather`/`Concat`/`Unsqueeze` on shapes, and
  small integer constants are folded at import time into dimension
  expressions, so `Reshape`, `Slice`, `Expand`, and `ConstantOfShape` get
  compile-time shapes. Where ONNX's own inference gives up after such a
  reshape, the importer propagates shapes itself.
- **Operations.** Elementwise arithmetic, comparisons, `Where`, `Cast`,
  `Transpose`, `Reshape`, `Slice`, `Concat`, `Expand`, `Identity`, `Pow` with constant
  exponents, `Reciprocal`; `MatMul` and `Gemm` as `std.linalg::matmul` /
  `batched_matmul` (or a contraction in index notation when only the right
  operand is a matrix); `Softmax`, `LayerNormalization`, `Gelu` (tanh),
  `Sigmoid`, and `Relu` as their standard-library operations; `Gather` along
  axis 0 as an element lookup; the `Reduce*` family as comprehensions.
- **Recovered decompositions.** Exporters spell some library operations out
  as primitives. The importer recognizes the shapes it knows and emits the
  library operation instead: PyTorch's RMS norm (`Mul(Mul(x,
  Reciprocal(Sqrt(Add(ReduceMean(Pow(x, 2)), eps)))), w)` and its `Div`,
  `Mul(x, x)`, and `Pow(-0.5)` spellings) becomes `std.nn.norm::rms_norm`,
  `Mul(x, Sigmoid(x))` becomes `silu`, and `Div(Exp(Sub(x, ReduceMax(x))),
  ReduceSum(Exp(...)))` over the last axis becomes `softmax`. The match is
  structural — the same operation types, the same operands, the axis, and
  the constants — so a variant is left as written, and every recovery is
  listed in `result.notes`; the replaced nodes are dropped as dead code.

Anything else — `Erf`, custom domains, `Gather` on other axes, symbolic
broadcasting the checker cannot prove — stops the import with a message
naming every such node. Nothing is executed and no weight enters the source.

## Linnet to ONNX

```bash
linnet onnx --std stdlib --bind Vocab=11 --bind H=8 --bind Heads=2 --bind Inner=16 \
            --bind Layers=2 --bind T=f32 --bind B=2 --bind S=5 \
            examples/04-tiny-transformer/src/lib.linnet > model.onnx.txt
```

`linnet onnx` takes the same options as `linnet stablehlo` and prints the
entry as an ONNX model in the ONNX text format (`onnx.parser.parse_model`
reads it; `onnx.save` turns it into a `.onnx` file). The graph is `main`; its
inputs are the entry's inputs followed by the parameters of the block
hierarchy as `param<N>`, and the model's `metadata_props` map each
`linnet.path.param<N>` to its parameter path, so weights bind by name and the
model itself carries none. Both exporters share one evaluator
(`backend::export_graph`): the same static-shape lowering of calls, loops,
and index notation feeds a `GraphTarget` that spells the primitives in one
format or the other — `GatherND` for element lookups, `Range` for `iota`,
`Reshape`+`Expand` (after a `Transpose` when axes reorder) for broadcasts,
`ReduceSum`/`ReduceMax`/... with an `axes` input for reductions.

An entry that touches `state` members gets them threaded explicitly: the
members it reads are inputs `state<N>` (mapped to their paths by
`linnet.state.state<N>`), and the members it assigns are outputs
`next_state<N>` after the entry's own `output<N>` results (mapped by
`linnet.next_state.next_state<N>`), so a runtime feeds each call's outputs
back as the next call's inputs. Entries returning a tuple have one output
per element.

`import_onnx` recognizes the metadata, so an exported model imports back into
Linnet with its parameters intact.

`tests/test_export.py` exports the tiny transformer, runs it under onnxruntime
with the same weights as the PyTorch materializer, checks the outputs agree,
and imports it back into Linnet that agrees too.

`tests/test_import.py` builds a GPT-style graph with `onnx.helper` (dynamic
batch and sequence, shape arithmetic feeding the reshapes, a causal mask as
an input), imports it, loads the result through the PyTorch adapter with the
graph's initializers, and compares it with a PyTorch reference; a second
import is byte-identical. It also imports a model exported by
`torch.onnx.export(..., dynamo=True)`.
