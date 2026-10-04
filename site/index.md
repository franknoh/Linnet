---
layout: home

hero:
  name: Linnet
  text: The checked source format for neural network architectures
  tagline: SafeTensors for model structure. Architecture is source, weights are data, and the architecture is checked and run without importing the model's Python.
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
    details: A model is text in one canonical format. A new head count, RoPE base or block is a diff you review, blame and tag like code.
  - title: Run the model, not its repository
    details: The compiler reads declarative source and imports no code from the model's author. It is not a sandbox, but the structure is known before anything runs.
  - title: Portable, not interpreted
    details: "Each backend gets its native path: generated PyTorch under torch.compile or CUDA graphs, XLA, ONNX Runtime and TensorRT."
  - title: One checked source, many runtimes
    details: The same file loads in PyTorch, JAX and ONNX Runtime, deploys to Triton, and exports to vLLM and llama.cpp for the Llama, Qwen2, Qwen3, Phi-3 and GPT-2 families.
---

## Git understands it

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

`linnet fmt` has one style and no options, so a diff shows changes to the
model, not its formatting. In CI, `linnet check` fails a change that breaks a
shape anywhere downstream and names the shapes.

## Benchmarks

Each backend runs the checked program on its own kernels. On one H100, in
`bf16`:

| | Linnet | Reference stacks |
| --- | --- | --- |
| Llama 3.1 8B, decode one request | 167 tok/s (XLA), 161 (CUDA graphs) | vLLM 152, transformers compiled 110 |
| BERT base, forward at batch 1 | 0.74 ms (CUDA graphs) | transformers 3.60, compiled 1.67 |
| SD VAE decoder, 512 px | 7.4 ms (XLA) | diffusers 22.0, compiled 10.4 |
| Llama 3.1 8B, 256 requests served | 5600 tok/s (CUDA graphs) | vLLM 5449 |
| gpt-oss 20B, decode one request | 368 tok/s (CUDA graphs) | vLLM 299, transformers 45 |
| gpt-oss 20B, 256 requests served | 6031 tok/s (CUDA graphs) | vLLM 4313 |
| Llama 3.1 8B, LoRA fine-tuning step | 1.04 s (PyTorch) | TRL 1.83 s |
| Llama 3.1 8B, GRPO step | 1.18 s (PyTorch) | TRL with vLLM 3.14 s |

Linnet loses on MiniLM and ResNet-50 under ONNX Runtime, and by 4%
splitting Qwen3 8B across two GPUs. The
[benchmarks](/benchmarks) page has every row: 24 models, 528 measurements.

## Targets

- Load in PyTorch, JAX (XLA, `jax.numpy`, Flax NNX) and ONNX Runtime (CUDA,
  TensorRT).
- Export StableHLO, ONNX, a Triton Inference Server model, a transformers
  checkpoint for vLLM, and GGUF for llama.cpp and Ollama.
- Serve with `linnet.serve`: continuous batching and an OpenAI-compatible
  HTTP server.
- [Train](/docs/training) in PyTorch or JAX: supervised fine-tuning, DPO and
  GRPO, with LoRA or fully sharded across GPUs. Or hand the model to
  transformers' `Trainer` and TRL.
- Import from PyTorch, JAX, StableHLO and ONNX.
- Run in ComfyUI with [linnet-comfyui](https://github.com/franknoh/linnet-comfyui).

The [compatibility matrix](/compatibility) lists the entry points and limits
of each target.

## Nest

[Nest](https://nest.franknoh.dev) is a verified registry of checked model
architectures and SafeTensors checkpoints, 24 models from MiniLM to gpt-oss
20B. CI checks that every card compiles, that its published checkpoint
matches every parameter's shape and dtype, and that it exports to StableHLO,
ONNX, PyTorch and JAX. `linnet.nest.load` loads a card by name.

## The language

Linnet is a small typed tensor language. Every dimension is a symbol, and
`where` clauses state the constraints a block relies on:

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

The slice bounds, the `reshape` and the head width `H / Heads` follow from
`H % Heads == 0`. Change `3 * H` to `2 * H` and `linnet check` points at the
slice that no longer fits.

The compiler is C++23 with no external dependencies, and the standard
library is written in Linnet. Not in the language: Python inside the model,
data-dependent shapes, hidden mutation, tensor data in source.

## Documentation

- [Installation](/guide/installation) and the
  [Quickstart](/docs/getting-started)
- [Coming from PyTorch](/guide/from-pytorch)
- [Language tour](/docs/language-tour) and
  [specification](/spec/00-overview)
- [Plan format](/docs/plan-format) for materializers
- [Benchmarks](/benchmarks) and [compatibility](/compatibility)
