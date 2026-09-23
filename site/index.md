---
layout: home

hero:
  name: Linnet
  text: A typed tensor language
  tagline: Write a model once as checked, weight-free source. Run it in PyTorch, JAX, XLA, or ONNX Runtime — and bring models back from all of them.
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
  - title: Shapes checked before anything runs
    details: Every dimension is a symbol the compiler reasons about. A reshape that does not preserve the element count, an attention head split that does not divide, a mismatched dtype — all reported at check time with the shapes involved, never as a runtime error.
  - title: Source without payload
    details: A .linnet file declares parameters and never contains tensor data. Checking or inspecting a package executes nothing, and weights bind by path from SafeTensors — no pickle, no model classes to trust.
  - title: One model, every framework
    details: The same source materializes as a torch.nn.Module, a jax function or Flax NNX module, a StableHLO module for XLA, or an ONNX model. Exporters recover softmax, norms, and activations from framework graphs back into library calls.
  - title: Index notation
    details: Contractions and reductions are written as indexed expressions with compile-time domains — sum[k] a[i, k] * b[k, j] — and lowered to broadcasts, gathers, and reductions, or matched to native kernels.
  - title: A standard library in Linnet
    details: linear, embedding, softmax, rms_norm, layer_norm, rope, attention, swiglu are ordinary source the checker verifies like any other. The compiler knows no model names.
  - title: Tooling from day one
    details: Formatter, linter, language server (VS Code, Neovim), executable spec tests, an optimizer with an e-graph superoptimizer, and linnet explain to see what it did.
---

## In one screen

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

Every slice bound, every `reshape`, and the `H / Heads` head width are proved
from `H % Heads == 0`; change `3 * H` to `2 * H` and `linnet check` names the
slice that no longer fits.

```python
from linnet_torch import load          # PyTorch
model = load("tiny.linnet", generics={"H": 512, "Heads": 8}, weights="weights/")

from linnet_jax import load            # JAX / XLA
step = load("tiny.linnet", generics={"H": 512, "Heads": 8}, weights="weights/")
```

```bash
linnet check tiny.linnet               # shapes, dtypes, index domains — nothing executes
linnet stablehlo --bind H=512 --bind Heads=8 --bind B=1 --bind S=128 tiny.linnet
linnet onnx      --bind H=512 --bind Heads=8 --bind B=1 --bind S=128 tiny.linnet
```

## Three ways to run the same source

```bash
linnet check model.linnet                       # shapes, dtypes, index domains; nothing executes
```

| | How | When |
| --- | --- | --- |
| Interpret | `linnet_torch.load(...)` walks the Core IR on tensors | debugging, tiny models, every op visible |
| Generate | `load(..., compile="inductor")`, `load_source(...)` in JAX | training and serving in a framework |
| Compile | `linnet stablehlo`, `linnet onnx` | XLA, ONNX Runtime, or another consumer |

On an H100 the generated path under `torch.compile` runs a small Llama forward
in 2.1 ms against 2.7 ms for hand-written compiled PyTorch, and the XLA path
in 0.66 ms — see [Benchmarks](/benchmarks).

## By the numbers

| | |
| --- | --- |
| Compiler | 25 k lines of C++23, no dependencies beyond the standard library |
| Tests | 57 unit and CLI tests, 46 executable spec cases, 49 Python adapter tests across PyTorch, JAX, and ONNX |
| Standard library | 12 modules, 42 operations and blocks — all Linnet source |
| Backends | PyTorch (interpreted, generated), JAX (compiled, generated, Flax NNX), StableHLO, ONNX |
| Importers | `torch.export`, `jax.export`, StableHLO text, ONNX |

## What it is, and is not

Linnet is a language for the tensor program itself: the shapes, the dtypes,
the parameters, and the operations that connect them. It has blocks, generics
over dimensions and dtypes, index notation, `static for` and `while`, `state`
for caches, and a standard library written in itself.

It has no Python inside the model, no data-dependent shapes, no hidden
mutation, and no tensor payloads in source. Tokenizers, data loading,
optimizers, and serving stay in the framework; the model becomes a file that
`linnet check` can vouch for and any of the backends can run.
