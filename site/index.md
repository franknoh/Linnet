---
layout: home

hero:
  name: Linnet
  text: The checked source format for neural network architectures
  tagline: SafeTensors for model structure. Architecture is source, weights are data, and the architecture is checked and run without importing the model's Python.
  image:
    src: /logo.svg
    alt: Linnet
  actions:
    - theme: brand
      text: Get started
      link: /guide/installation
    - theme: alt
      text: Benchmarks
      link: /benchmarks
    - theme: alt
      text: Nest registry
      link: https://nest.franknoh.dev

features:
  - title: Git understands the architecture
    details: A model is text in one canonical format. A new head count, RoPE base, or block is a diff you review, blame, and tag like any other code.
  - title: Run the model, not its repository
    details: Checking and loading read declarative source with the compiler. No modeling code from the model's author is imported, and weights come from SafeTensors.
  - title: Portable, not interpreted
    details: Each backend gets its own native path — generated PyTorch under torch.compile or CUDA graphs, XLA, ONNX Runtime and TensorRT.
  - title: One checked source, many runtimes
    details: The same file loads in PyTorch, JAX, and ONNX Runtime, deploys to Triton, and exports to vLLM and llama.cpp for the Llama, Qwen2, Qwen3, Phi-3, and GPT-2 families.
---

## Weights got SafeTensors. Structure deserves the same.

SafeTensors made weights data: a header of names, shapes, and dtypes, then
bytes, loaded without running pickle. The structure those weights belong to
is still usually code — a `modeling_*.py` and a `config.json` that mean
whatever the Python does when it runs.

```text
Usual distribution
  structure   config.json + modeling_*.py    Python, executed to find out
  weights     model.safetensors

Linnet
  structure   model.linnet                   checked source, read by the compiler
  weights     model.safetensors              bound by parameter path
```

A `.linnet` file declares every parameter with its shape and dtype, so the
checkpoint a model needs is known before any weights are read:

```console
$ linnet inspect --parameters model.linnet
examples.model::Model<H, Inner, Layers, Vocab, T>
  param embedding: Tensor[Vocab, H; T]
  param layers[*].norm_weight: Tensor[H; T] x Layers
  ...
```

Loading binds SafeTensors to those paths. The PyTorch and ONNX loaders check
each tensor's shape and dtype against the declaration before anything runs,
and [Nest](#nest) checks every published checkpoint the same way.

## Git should understand your model architecture

Two of the differences between the Llama 3.1 8B and Mistral 7B v0.3 cards in
[Nest](https://nest.franknoh.dev), as `git diff` shows them:

```diff
--- a/src/rope.linnet
+++ b/src/rope.linnet
-pub const THETA: f32 = 500000.0
+pub const THETA: f32 = 1000000.0
--- a/nest.toml
+++ b/nest.toml
 [generics]
-Vocab = 128256
+Vocab = 32768
 H = 4096
 Heads = 32
 KvHeads = 8
 Inner = 14336
 Layers = 32
 Batch = 1
-MaxSeq = 8192
+MaxSeq = 32768
```

`linnet fmt` has one style and no options, so a diff shows what changed in
the model rather than in its formatting. A head count, a norm, or a block
swap goes through review like any other change; `git blame` says who moved
the RoPE base; a tag names an architecture the way it names a release. In CI,
`linnet check` fails a change that breaks a shape anywhere downstream, and
the message names the shapes.

## Run the model, not its repository

```text
A custom architecture on the Hub         A Linnet model
  config.json                              model.linnet
  modeling_custom.py                       model.safetensors
  requirements.txt
  model.safetensors
  trust_remote_code=True
```

A custom architecture on the Hub loads by importing Python the repository
ships (`trust_remote_code=True`); the architecture is whatever that code does.
A `.linnet` file is declarative. `linnet check` and every loader read it with
the compiler, which checks shapes, dtypes, parameters, and every operation
before anything runs, without importing code from the model's author. What
runs afterwards is the framework you chose, executing what the compiler
generated from the checked source.

This is not a sandbox: the compiler, the frameworks, and the runtimes are
ordinary software, and a checked model can still compute the wrong thing. The
claim is narrower — the structure is known before anything runs, from a file
that contains nothing else. Importing a model into Linnet from PyTorch or JAX
traces its Python once, when the source is written; loading the result does
not.

## Portable does not mean interpreted

```text
model.linnet
  └─ linnet check           shapes, dtypes, parameters, operations
      └─ plan               one checked, typed program
          ├─ PyTorch         generated source; torch.compile or CUDA graphs
          ├─ JAX / XLA       StableHLO, or generated jax.numpy
          ├─ ONNX            ONNX Runtime, TensorRT, Triton
          └─ exports         transformers checkpoint (vLLM), GGUF (llama.cpp)
```

The plan is lowered to each backend's own kernels: `F.scaled_dot_product_attention`,
`F.conv2d`, and `index_put_` in PyTorch, `dot_general` and `convolution` in
StableHLO, `MatMul` and `Conv` in ONNX. On one H100, in `bf16`:

| | Linnet | Reference stacks |
| --- | --- | --- |
| Llama 3.1 8B, decode one request | 170 tok/s (XLA), 168 (CUDA graphs) | vLLM 157, transformers compiled 110 |
| BERT base, forward at batch 1 | 0.74 ms (CUDA graphs) | transformers 3.60, compiled 1.67 |
| SD VAE decoder, 512 px | 7.4 ms (XLA) | diffusers 22.0, compiled 10.4 |
| Llama 3.1 8B, 256 requests served | 4858 tok/s (CUDA graphs) | vLLM 5658 |
| gpt-oss 20B, decode one request | 299 tok/s (XLA) | vLLM 303, transformers 45 |

Where Linnet loses — vLLM's paged serving, tensor parallelism across GPUs —
is on the [benchmarks](/benchmarks) page with every other row: 24 models, 528
measurements.

## One source, many runtimes

The same checked source is what every target starts from. Not every target
takes every model:

| Target | How | Scope |
| --- | --- | --- |
| PyTorch | `linnet.torch.load`: interpreted, generated source, `torch.compile`, CUDA graphs; training, device placement, tensor parallelism | |
| JAX | `linnet.jax.load` (XLA), `load_source` (`jax.numpy`, `jax.grad`), `load_nnx` (Flax NNX) | |
| StableHLO | `linnet stablehlo` | static shapes |
| ONNX Runtime | `linnet onnx`, `linnet.onnx.export_model` and `load_model`, CUDA or TensorRT | static shapes |
| Triton Inference Server | `linnet.triton`: a model repository over ONNX or a Python backend | ONNX: entries without state |
| vLLM and other transformers-checkpoint servers | `linnet.hf`: a transformers checkpoint | Llama, Qwen2, Qwen3, Phi-3, and GPT-2 families |
| llama.cpp, Ollama | `linnet.gguf`: GGUF and a Modelfile | the same families |
| ComfyUI | [linnet-comfyui](https://github.com/franknoh/linnet-comfyui) custom nodes | PyTorch |
| Serving | `linnet.serve`: continuous batching over PyTorch, JAX, or ONNX, with sampling and an OpenAI-compatible HTTP server | decoders with `prefill_slots` and `decode_rows` |

Models also come the other way: `torch.export`, `jax.export`, StableHLO text,
and ONNX graphs import into `.linnet` source. The
[compatibility matrix](/compatibility) lists what each path supports.

## Nest

[Nest](https://nest.franknoh.dev) is a verified registry of checked model
architectures and SafeTensors checkpoints. A card is `.linnet` source and a
`nest.toml` naming its generics, license, links, and the Hub repository its
SafeTensors live in. CI checks every card:

- the source compiles and the card's generics bind;
- the published checkpoint's SafeTensors headers, read with two range
  requests and nothing downloaded, match every parameter's shape and dtype;
- the model exports to StableHLO, ONNX, PyTorch, and JAX;
- the card has its README, license, and links.

```python
from linnet import nest
model = nest.load("gpt2")     # source, generics, and weights from the registry
```

It holds 24 models, from MiniLM to gpt-oss 20B, each measured against the
stacks it usually runs on.

## How it works

Underneath is a small typed tensor language. Every dimension is a symbol,
shapes and dtypes are checked before anything runs, and `where` clauses carry
the constraints a block relies on:

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

| | |
| --- | --- |
| Language | blocks, generics over dimensions and dtypes, `where` constraints, index notation, `static for` and `while`, `state` for caches |
| Standard library | linear, attention, norms, RoPE, KV caches, convolution, pooling, quantized linears, a counter-based PRNG — all written in Linnet |
| Compiler | C++23, no external dependencies; Core IR, an optimizer, and a JSON [plan](/docs/plan-format) for materializers |
| Tooling | `linnet fmt`, a language server for VS Code and Neovim, architecture diagrams, `linnet explain` for kernel choices |
| Specification | a normative [spec](/spec/00-overview) and grammar, with executable spec tests |

Not in the language: Python inside the model, data-dependent shapes, hidden
mutation, tensor data in source. Tokenizers, data loading, optimizers, and
serving policy stay in the framework.

## Getting started

```bash
linnet check model.linnet        # shapes, dtypes, parameters — nothing runs
```

```python
from linnet.torch import load

model = load("model.linnet", generics={"H": 512, "Heads": 8}, weights="model.safetensors")
```

[Installation](/guide/installation) sets up the compiler and the Python
package, and [Coming from PyTorch](/guide/from-pytorch) maps what you already
write onto Linnet.
