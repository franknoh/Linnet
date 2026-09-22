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
  `Transpose`, `Reshape`, `Slice`, `Concat`, `Expand`, `Pow` with constant
  exponents, `Reciprocal`; `MatMul` and `Gemm` as `std.linalg::matmul` /
  `batched_matmul` (or a contraction in index notation when only the right
  operand is a matrix); `Softmax`, `LayerNormalization`, `Gelu` (tanh),
  `Sigmoid`, and `Relu` as their standard-library operations; `Gather` along
  axis 0 as an element lookup; the `Reduce*` family as comprehensions.

Anything else — `Erf`, custom domains, `Gather` on other axes, symbolic
broadcasting the checker cannot prove — stops the import with a message
naming every such node. Nothing is executed and no weight enters the source.

`tests/test_import.py` builds a GPT-style graph with `onnx.helper` (dynamic
batch and sequence, shape arithmetic feeding the reshapes, a causal mask as
an input), imports it, loads the result through the PyTorch adapter with the
graph's initializers, and compares it with a PyTorch reference; a second
import is byte-identical. It also imports a model exported by
`torch.onnx.export(..., dynamo=True)`.
