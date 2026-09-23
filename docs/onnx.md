# ONNX

`linnet.onnx` imports ONNX graphs as Linnet source; `linnet onnx`
exports an entry as an ONNX model. Together they round-trip: an exported
model imports back with its parameters intact.

```bash
cd python/linnet && uv sync --extra onnx    # or: pip install "linnet-lang[onnx]"
```

## Importing

```python
from linnet.onnx import import_onnx

result = import_onnx("model.onnx", output="src/model.linnet", weights="weights/", std_root="stdlib")
print(result.notes)      # anything the translation dropped or recovered
```

`import_onnx` runs shape inference, translates the graph node by node into a
Core IR plan, and prints it as formatted source with `linnet emit`. The
result is checked before it is written; nothing is executed.

| ONNX | Linnet |
| --- | --- |
| tensor initializers, by dotted name | `param` members; numbered children with the same structure become a sub array (`blocks.0.attn.qkv.weight` is `blocks: [Block; N]`) |
| graph inputs with `dim_param` | entry inputs with generic dimensions (`forward<batch: Dim, seq: Dim>`) |
| `Shape`, `Gather`, `Concat`, `Unsqueeze` on shapes | folded into dimension expressions, so `Reshape`, `Slice`, `Expand`, `ConstantOfShape` get compile-time shapes |
| arithmetic, comparisons, `Where`, `Cast`, `Transpose`, `Reshape`, `Slice`, `Concat`, `Expand`, `Identity`, `Pow` (constant exponent), `Reciprocal` | the primitive with the same meaning |
| `MatMul`, `Gemm` | `std.linalg::matmul` / `batched_matmul`, or index notation |
| `Softmax`, `LayerNormalization`, `Gelu` (tanh), `Sigmoid`, `Relu` | the standard-library operation |
| `Gather` along axis 0 | an element lookup |
| `Reduce*` | comprehensions |

Decompositions exporters produce are recognized and folded back: PyTorch's
RMS norm (`Mul(Mul(x, Reciprocal(Sqrt(Add(ReduceMean(Pow(x, 2)), eps)))), w)`
and its `Div`, `Mul(x, x)`, `Pow(-0.5)` spellings), `Mul(x, Sigmoid(x))` as
`silu`, and the hand-written softmax over the last axis. The match is
structural, so a variant with a different constant stays as written; every
recovery is listed in `notes`.

With `weights=`, initializers are saved as SafeTensors under their ONNX
names, plus `bindings.json` when a path had to be renamed. Anything without
a mapping (`Erf`, custom domains, `Gather` on other axes) stops the import
with a message naming every such node.

## Exporting

```bash
linnet onnx --std stdlib --bind Vocab=11 --bind H=8 --bind Heads=2 --bind Inner=16 \
            --bind Layers=2 --bind T=f32 --bind B=2 --bind S=5 \
            examples/04-tiny-transformer/src/lib.linnet > model.onnx.txt
```

The output is the ONNX text format (`onnx.parser.parse_model` reads it,
`onnx.save` writes a `.onnx` file). The graph is `main`:

| | |
| --- | --- |
| inputs | the entry's inputs, then parameters as `param<N>` |
| `metadata_props` | `linnet.path.param<N>` maps each parameter to its path |
| state | read members are inputs `state<N>` (`linnet.state.state<N>`); assigned members are outputs `next_state<N>` after the entry's `output<N>` results (`linnet.next_state.next_state<N>`) |
| loops | `while` becomes `Loop` |
| tuple results | one output per element |

The model carries no weights; a runtime binds them by name and feeds each
call's `next_state` outputs back as the next call's `state` inputs.

## Tests

`tests/test_export.py` exports the tiny transformer, runs it under
onnxruntime with the PyTorch materializer's weights, compares the outputs,
and imports it back. `tests/test_import.py` imports a GPT-style graph built
with `onnx.helper` and a model from `torch.onnx.export(..., dynamo=True)`,
and compares both against PyTorch references.
