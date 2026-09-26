---
layout: home

hero:
  name: Linnet
  text: A typed tensor language
  tagline: Models as checked, weight-free source that runs in PyTorch, JAX, XLA, and ONNX Runtime.
  image:
    src: /logo.svg
    alt: Linnet
  actions:
    - theme: brand
      text: Get started
      link: /guide/installation
    - theme: alt
      text: Coming from PyTorch
      link: /guide/from-pytorch
    - theme: alt
      text: Examples
      link: /examples/

features:
  - title: Checked before it runs
    details: Every dimension is a symbol. Reshapes, head splits, and dtypes are verified at check time, with the shapes named in the message.
  - title: Source without weights
    details: A .linnet file declares parameters and carries no data. Checking executes nothing; weights bind by path from SafeTensors.
  - title: One source, every framework
    details: The same file becomes a torch.nn.Module, a JAX function, a StableHLO module, or an ONNX model, and models come back the other way.
  - title: A library you can read
    details: linear, attention, softmax, rms_norm, rope, and the PRNG are ordinary Linnet in stdlib. The compiler knows no model names.
---

## A block

```linnet
module tiny

use std.nn.attention::{attention, causal_mask}
use std.nn.linear::{Linear}
use std.nn.norm::{RmsNorm}

pub block Attention<H: Dim, Heads: Dim, T: Float = bf16>
where
    Heads > 0,
    H % Heads == 0
{
    sub norm: RmsNorm<H, T>
    sub qkv: Linear<H, 3 * H, T>
    sub out: Linear<H, H, T>

    pub entry forward<B: Dim, S: Dim>(x: Tensor[B, S, H; T]) -> Tensor[B, S, H; T] {
        let projected = qkv.forward(norm.forward(x))
        let q = heads<B, S, Heads, H / Heads, T>(projected[:, :, 0:H])
        let k = heads<B, S, Heads, H / Heads, T>(projected[:, :, H:2 * H])
        let v = heads<B, S, Heads, H / Heads, T>(projected[:, :, 2 * H:3 * H])
        let mixed = attention(q, k, v, rsqrt(cast<f32>(H / Heads)), some(causal_mask<S, S>()))
        return x + out.forward(reshape(permute(mixed, [0, 2, 1, 3]), [B, S, H]))
    }
}

// `[B, S, N * D]` -> `[B, N, S, D]`.
fn heads<B: Dim, S: Dim, N: Dim, D: Dim, T: Float>(
    x: Tensor[B, S, N * D; T],
) -> Tensor[B, N, S, D; T] {
    return permute(reshape(x, [B, S, N, D]), [0, 2, 1, 3])
}
```

The slice bounds, the `reshape`, and the head width `H / Heads` follow from
`H % Heads == 0`. Change `3 * H` to `2 * H` and `linnet check` points at the
slice that no longer fits.

```python
from linnet.torch import load          # PyTorch
model = load("tiny.linnet", generics={"H": 512, "Heads": 8}, weights="weights/")

from linnet.jax import load            # JAX / XLA
step = load("tiny.linnet", generics={"H": 512, "Heads": 8}, weights="weights/")
```

```bash
linnet check tiny.linnet               # shapes, dtypes, index domains — nothing executes
linnet stablehlo --bind H=512 --bind Heads=8 --bind B=1 --bind S=128 tiny.linnet
linnet onnx      --bind H=512 --bind Heads=8 --bind B=1 --bind S=128 tiny.linnet
```

## Interpret, generate, or compile

```bash
linnet check model.linnet        # shapes, dtypes, index domains — nothing runs
```

| Path | Call | Use |
| --- | --- | --- |
| Interpreted | `linnet.torch.load(...)` | inspecting a model op by op |
| Generated code | `load(..., compile="inductor")` or `linnet.jax.load_source(...)` | training and serving inside a framework |
| Compiled graph | `linnet stablehlo`, `linnet onnx` | XLA, ONNX Runtime, other consumers |

On one H100, Llama 3.1 8B decodes one request at 170 tokens per second
through XLA (vLLM: 158, transformers compiled: 103), and BERT's forward pass
takes 0.85 ms against 3.65 in transformers; serving many requests at once,
vLLM is still well ahead. Every model in the zoo, and where Linnet loses, is
on the [benchmarks](/benchmarks) page.

## At a glance

| | |
| --- | --- |
| Compiler | C++23, about 25 k lines, no external dependencies |
| Tests | 57 unit and CLI tests, 46 spec cases, 49 adapter tests (PyTorch, JAX, ONNX) |
| Standard library | 12 modules, 42 operations and blocks, all written in Linnet |
| Backends | PyTorch, JAX and Flax NNX, StableHLO, ONNX |
| Importers | `torch.export`, `jax.export`, StableHLO text, ONNX |

## Scope

Linnet describes the tensor program: shapes, dtypes, parameters, and the
operations between them. Blocks, generics over dimensions and dtypes, index
notation, `static for` and `while`, `state` for caches, and a standard
library in the language itself.

Not in the language: Python inside the model, data-dependent shapes, hidden
mutation, tensor data in source. Tokenizers, data loading, optimizers, and
serving stay in the framework; the model is a file `linnet check` can verify
and every backend can run.
