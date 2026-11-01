# MLX

`linnet.mlx` runs Linnet models on [MLX](https://github.com/ml-explore/mlx):
Apple silicon's GPU through Metal, NVIDIA GPUs through MLX's CUDA build,
and the CPU.

## Install

```bash
uv add "linnet-lang[mlx] @ git+https://github.com/franknoh/Linnet#subdirectory=python/linnet"
```

The `mlx` extra is `mlx` on macOS and `mlx[cpu]` on Linux. For an NVIDIA
GPU on Linux, install `mlx[cuda13]` (or `mlx[cuda12]`) instead.

## Load a model

```python
from linnet.mlx import load_model

model = load_model("model.linnet", generics={..., "T": "bf16"},
                   weights="model.safetensors", cast_dtype=True)
logits = model.run_entry("prefill", [tokens, np.int32(0)])
logits = model.run_entry("decode", [token, np.int32(position)])
```

`load_model` generates each entry as `mlx.core` source (`linnet mlx`) for
the shapes it receives and compiles it with `mx.compile`. Every entry shares
one copy of the weights, read from the checkpoint a tensor at a time when an
entry first needs it. The block's state stays on the device between calls
(`model.state`, `model.reset_state()`). Results come back as evaluated MLX
arrays; `np.asarray` takes them to the host.

`linnet.serve.Engine` takes such a model, `nest.load(...,
backend="mlx_model")` loads a Nest card this way, and `linnet serve MODEL
--backend mlx` serves one over HTTP ([Serving](integrations.md#over-http)).

| Option | Effect |
| --- | --- |
| `cast_dtype=True` | converts floating-point weights to the dtype the generics give |
| `compile=False` | runs the generated code without `mx.compile` |
| `numerics=` | the numerics policy; `fast` by default, which uses `mx.fast.rms_norm` and `layer_norm` |

## What runs natively

Attention is `mx.fast.scaled_dot_product_attention`, with grouped heads and
a causal or boolean mask. Linear layers, matmuls, embeddings, softmax,
activations, RMS and layer norms, convolutions and KV cache writes are MLX
operations; everything else is the library's canonical body. A
`while` loop reads its condition back each iteration, so an entry with one
runs uncompiled. MLX has no 8-bit or 4-bit float dtypes: FP8 and the 4-bit
formats are held as bytes and converted by `linnet.mlx.formats` (FP8 E4M3 by
MLX's own `from_fp8` and `to_fp8`).

## Not yet

- Tile kernels (`kernel`): ops run their bodies.
- Gradients: an op's `grad` is not used, and there is no MLX trainer.
- Serving: no pages, packed prompts or plans; embeddings serve on PyTorch.
- `f64` on a GPU, which MLX runs only on the CPU.

## Tested

- On CPU (`mlx[cpu]`) and on a GB10's GPU (`mlx[cuda13]`), every entry of
  the test models matches PyTorch's. Serving gives PyTorch's greedy and
  seeded tokens, and log-probabilities within TF32's rounding on the GPU.
- `linnet serve tinyllama-1.1b-chat --backend mlx` on the GB10 answers as the
  PyTorch, JAX and ONNX backends do. It decodes one request at 58 tokens a
  second, against PyTorch's 101 with CUDA graphs.
- Apple silicon is not tested yet.
