# ONNX

`linnet.onnx` imports ONNX graphs as Linnet source; `linnet onnx`
exports an entry as an ONNX model. Together they round-trip: an exported
model imports back with its parameters intact.

```bash
cd python/linnet && uv sync --extra onnx    # or: pip install ".[onnx]"
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
call's `next_state` outputs back as the next call's `state` inputs. A KV
cache write (`std.nn.cache::write_at`, `write_span`, `write_rows`,
`write_slot`, `write_slots`) is one `ScatterND` over the positions it
changes.

`linnet.onnx.export_model(source, generics=..., weights=..., entry=...)`
embeds a checkpoint as initializers; `cast_dtype=True` converts its
floating-point tensors to the dtype the graph declares, so an f32
checkpoint exports as an `f16` or `bf16` model (`--bind T=f16`).

An `entry` declared at module level is a function of its inputs alone (a
loss, a preprocessing step, a reward) and has no weights, so
`export_function` makes a complete model from the source:

```python
from linnet.onnx import export_function

exported = export_function("functions.linnet", "normalize_images",
                           generics={"B": 1, "H": 224, "W": 224, "C": 3})
exported.save("normalize_images.onnx")
```

## Running on ONNX Runtime

```python
from linnet.onnx import load_model

model = load_model("model.linnet", generics={..., "T": "f16"},
                   weights="model.safetensors", cast_dtype=True)
logits = model.run_entry("prefill", [tokens, np.int32(0)])
logits = model.run_entry("decode", [token, np.int32(position)])
```

`load_model` runs every entry of the root block on ONNX Runtime (CUDA where
it has it, else the CPU), each entry exported for the shapes it is called
with. No graph embeds the weights: they go to the device once, as
`OrtValue`s, and are bound to every entry's session, so a prompt entry and
a step entry share one copy, and a model past ONNX's 2 GB file limit loads
all the same. The block's state stays on the device between calls.
`linnet.serve.Engine` takes such a model, and `nest.load(...,
backend="onnx_model")` loads a zoo card this way.

An entry without state (an encoder's, a classifier's) takes the weights as
initializers instead, from the same host copy, so ONNX Runtime can fold
what reads only weights, such as a batch norm into its convolution. An
entry with state runs what reads only weights (a dequantizer unpacking
every expert) once, in a session of its own, and binds the result; entries
computing the same value share it. On CUDA, every session allocates from
one shared arena. A new shape reads no weight again: those already on the
device are only checked against the checkpoint's header.

A state an entry writes with a scatter it alone reads (a KV cache row) is
updated where it lies: the next value is bound to the state's own buffer,
so no call copies the rest of the cache. `run_entry(..., argmax=True)`
takes the argmax of the first result in the graph, and `cuda_graph=True`
captures the entry as a CUDA graph and replays it, its inputs copied into
buffers that stay put, when every state it writes is updated in place and
no result is `bf16` (it runs as usual otherwise). `linnet.serve` does both
for its decoding step (a step that samples takes the logits instead).

`model.place(array, "f16")` puts an input on the device once, for a call
that repeats; `run_entry(..., keep_on_device=True)` leaves the results there
as `OrtValue`s. `providers` takes what `InferenceSession` does, TensorRT's
`(name, options)` pairs included. Operations ONNX Runtime has no `bf16`
kernel for (contractions, convolution, pooling, reductions, resizing) run
in `f32` and round back, and 8- and 16-bit integer arithmetic runs in
`i32`; its CPU kernels have no `bf16` arithmetic at all, so use
`f16` or `f32` there.

## Tests

`tests/onnx/test_export.py` exports the tiny transformer, runs it under
onnxruntime with the PyTorch materializer's weights, compares the outputs,
and imports it back. `tests/onnx/test_import.py` imports a GPT-style graph built
with `onnx.helper` and a model from `torch.onnx.export(..., dynamo=True)`,
and compares both against PyTorch references.
