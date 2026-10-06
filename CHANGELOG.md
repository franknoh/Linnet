# Changelog

## 0.1.0 (unreleased)

The first release: the language, its compiler and tools, and the Python
package that runs Linnet models in PyTorch, JAX and ONNX Runtime.

### Language

- Typed tensor source: shapes as types, dimensions as compile-time
  expressions, and `where` clauses the checker proves at every call.
- Index notation with explicit reductions and accumulation dtypes; no
  implicit summation, broadcasting only where it is provable, no implicit
  dtype promotion.
- Blocks with `param`, `buffer`, `state` and `sub` members, optional
  parameters and children, block arrays with `static for`, and runtime
  `while` loops.
- A standard library written in Linnet: linear layers, embeddings,
  normalizations, attention (grouped, masked, with sinks), rotary
  positions, KV caches, convolutions, pooling, losses, mixture of experts,
  tensor-parallel collectives, counter-based randomness and quantized
  weights (int8, int4, MXFP4).
- The specification in `spec/`, with an executable suite in `spec-tests/`.

### Compiler and tools

- `linnet check`, `lint`, `fmt`, `inspect`, `explain` and `init`, with
  stable diagnostic codes and JSON output.
- Exports of one entry to StableHLO, ONNX, PyTorch source and JAX source,
  under three numerics policies.
- A language server (`linnet lsp`), a VS Code extension, and Vim and
  Neovim support.
- `linnet serve`, which serves a model with OpenAI's API.
- `linnet memory` and `linnet fit`: static memory analysis of inference and
  training (weights, KV caches, activation liveness, gradients, optimizer
  states, checkpointing, sharding, tensor and pipeline parallelism), each
  number marked exact, backend-modeled, estimated or unknown; the largest
  batch, context or cache that fits a device; and the layout over several
  devices with the highest roofline throughput bound.

### Python package (`linnet-lang`)

- Loaders for PyTorch (as a `torch.nn.Module`, compiled or replayed as CUDA
  graphs), JAX (XLA, `jax.numpy`, Flax NNX) and ONNX Runtime (CUDA,
  TensorRT).
- PyTorch across GPUs: block placement and offloading, tensor parallelism,
  and pipelines of stages over processes under GPipe or 1F1B
  (`linnet.torch.pipeline`).
- Imports from PyTorch, JAX, StableHLO and ONNX; exports to a Triton
  Inference Server model, a transformers checkpoint for vLLM, SGLang and
  TGI, and GGUF for llama.cpp and Ollama.
- `linnet.serve`: continuous batching with packed prompts and a CUDA graph
  per step, and an OpenAI-compatible HTTP server.
- `linnet.train`: supervised fine-tuning, DPO and GRPO in PyTorch and JAX,
  with LoRA, fully sharded data parallelism and checkpoints; the models
  also train under transformers' `Trainer` and TRL.
- `linnet.nest`: models by Nest name, from any Hugging Face Hub repo with a
  card, or from a directory; `transformers` checkpoints of the Llama,
  Mistral, Qwen2, Qwen3, Phi-3 and GPT-2 families convert on the way.
- Platform wheels for Linux (x86_64, aarch64), macOS (arm64) and Windows
  (x64) that carry the compiler and its standard library.
