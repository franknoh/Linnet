# ONNX

`linnet.onnx` runs Linnet models on ONNX Runtime, exports entries to ONNX,
and imports ONNX graphs as Linnet source. Exported models import back with
parameters intact.

## Install

```bash
cd python/linnet && uv sync --extra onnx    # or: pip install ".[onnx]"
```

## Load a model

```python
from linnet.onnx import load_model

model = load_model("model.linnet", generics={..., "T": "f16"},
                   weights="model.safetensors", cast_dtype=True)
logits = model.run_entry("prefill", [tokens, np.int32(0)])
logits = model.run_entry("decode", [token, np.int32(position)])
```

`load_model` runs every entry of the root block on ONNX Runtime (CUDA if
available, else the CPU), exporting each for the shapes it receives.
The weights go to the device once and every entry shares them, so a model
past ONNX's 2 GB file limit still loads. The block's state stays on the
device between calls. `linnet.serve.Engine` takes such a model, and
`nest.load(..., backend="onnx_model")` loads a zoo card this way.

## Load options

| Option | Effect |
| --- | --- |
| `cast_dtype=True` | converts floating-point weights to the dtype the generics give |
| `providers=` | passed to `InferenceSession`, TensorRT's `(name, options)` pairs included |

ONNX Runtime runs `bf16` contractions, convolution, pooling, reductions
and resizing in `f32`, and 8- and 16-bit integer arithmetic in `i32`. Its
CPU kernels have no `bf16` arithmetic; use `f16` or `f32` there.

## Entries and state

| `model.run_entry(...)` option | Effect |
| --- | --- |
| `argmax=True` | takes the argmax of the graph's first result |
| `cuda_graph=True` | captures and replays the entry as a CUDA graph when every state it writes updates in place (a KV cache row) and no result is `bf16` |
| `keep_on_device=True` | leaves the results on the device as `OrtValue`s |

`model.place(array, "f16")` puts an input on the device once, for
repeated calls.

## Functions

A module-level `entry` (a loss, a preprocessing step, a reward) has no
weights, so `export_function` makes a complete model from the source:

```python
from linnet.onnx import export_function

exported = export_function("functions.linnet", "normalize_images",
                           generics={"B": 1, "H": 224, "W": 224, "C": 3})
exported.save("normalize_images.onnx")
```

## Linnet to ONNX

```bash
linnet onnx --std stdlib --bind Vocab=11 --bind H=8 --bind Heads=2 --bind Inner=16 \
            --bind Layers=2 --bind T=f32 --bind B=2 --bind S=5 \
            examples/04-tiny-transformer/src/lib.linnet > model.onnx.txt
```

`linnet onnx` prints ONNX text: `onnx.parser.parse_model` reads it and
`onnx.save` writes a `.onnx` file. The graph `main` has:

| Part | Contents |
| --- | --- |
| inputs | the entry's inputs, then parameters as `param<N>` |
| `metadata_props` | `linnet.path.param<N>` maps each parameter to its path |
| state | read members are inputs `state<N>` (`linnet.state.state<N>`); assigned members are outputs `next_state<N>` after the entry's `output<N>` results (`linnet.next_state.next_state<N>`) |
| loops | `while` becomes `Loop` |
| tuple results | one output per element |

The model carries no weights. Bind them by name, and feed each call's
`next_state` outputs back as the next call's `state` inputs.

`linnet.onnx.export_model(source, generics=..., weights=..., entry=...)`
embeds a checkpoint as initializers; with `cast_dtype=True`, an f32
checkpoint exports as `f16` or `bf16` (`--bind T=f16`).

## ONNX to Linnet

```python
from linnet.onnx import import_onnx

result = import_onnx("model.onnx", output="src/model.linnet", weights="weights/", std_root="stdlib")
print(result.notes)      # anything the translation dropped or recovered
```

`import_onnx` translates the graph into checked Linnet source, executing
nothing.

| ONNX | Linnet |
| --- | --- |
| tensor initializers, by dotted name | `param` members; numbered children with the same structure become a sub array (`blocks.0.attn.qkv.weight` is `blocks: [Block; N]`) |
| graph inputs with `dim_param` | entry inputs with generic dimensions (`forward<batch: Dim, seq: Dim>`) |
| `Shape`, `Gather`, `Concat`, `Unsqueeze` on shapes | compile-time dimension expressions |
| arithmetic, comparisons, `Where`, `Cast`, `Transpose`, `Reshape`, `Slice`, `Concat`, `Expand`, `Identity`, `Pow` (constant exponent), `Reciprocal` | the primitive with the same meaning |
| `MatMul`, `Gemm` | `std.linalg::matmul` / `batched_matmul`, or index notation |
| `Softmax`, `LayerNormalization`, `Gelu` (tanh), `Sigmoid`, `Relu` | the standard-library operation |
| `Gather` along axis 0 | an element lookup |
| `Reduce*` | comprehensions |

PyTorch's RMS norm decompositions, `Mul(x, Sigmoid(x))` and a last-axis
softmax fold back into `rms_norm`, `silu` and `softmax`; `notes` lists
each recovery. A variant with a different constant stays as written.

`weights=` saves initializers as SafeTensors under their ONNX names, plus
`bindings.json` for renamed paths. An unmapped node
(`Erf`, custom domains, `Gather` on other axes) stops the import with a
message naming each one.

## Tests

`tests/onnx/` round-trips the tiny transformer through onnxruntime and
imports a GPT-style graph and a `torch.onnx.export(..., dynamo=True)`
model, comparing each against PyTorch.
