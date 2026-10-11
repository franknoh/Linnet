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
  parameters and children, block arrays with `static for`, runtime `while`
  and counted `for` loops, scans (`yield`), and struct values.
- Custom backward passes (`grad` after an op's body), and compiler autodiff
  (`--grad` on every export).
- Tile kernels (`kernel`): Triton in generated PyTorch, Pallas in generated
  JAX, launched in place of an op's body; `cumsum` everywhere.
- A standard library written in Linnet: linear layers, embeddings,
  normalizations, attention (grouped, masked, with sinks, over pages),
  rotary positions, KV caches, convolutions, pooling, losses, mixture of
  experts, tensor-parallel collectives, counter-based randomness and
  quantized weights (int8 and FP8 scaled per row, group or block; int4;
  MXFP4).
- The specification in `spec/`, with an executable suite in `spec-tests/`.
- Packages with path and git dependencies (GitHub and the Hugging Face Hub),
  pinned in `linnet.lock` by `linnet fetch` and `linnet update`.

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
  devices with the highest throughput, predicted operation by operation
  from rates measured on the device (`--device local` measures the GPU it
  runs on, once).

### Python package (`linnet-lang`)

- Loaders for PyTorch (as a `torch.nn.Module`, compiled or replayed as CUDA
  graphs), JAX (XLA, `jax.numpy`, Flax NNX) and ONNX Runtime (CUDA,
  TensorRT).
- PyTorch across GPUs: block placement and offloading, tensor parallelism,
  and pipelines of stages over processes under GPipe or 1F1B
  (`linnet.torch.pipeline`), each stage split or sharded across processes
  of its own; a sharded stage gathers its weights once a step.
- Under `torch.compile`, each kind of repeated block compiles once: the
  first step compiles in a fraction of the time.
- Imports from PyTorch, JAX, StableHLO and ONNX; exports to a Triton
  Inference Server model, a transformers checkpoint for vLLM, SGLang and
  TGI, and GGUF for llama.cpp and Ollama.
- `linnet.serve`: continuous batching with packed prompts and a CUDA graph
  per step, serving from pages of one cache pool, and an OpenAI-compatible
  HTTP server.
- 8-bit weights on GPUs: Linnet's kernels multiply int8 and FP8 weights as
  they are stored; under `numerics="fast"` a prompt's rows round to 8 bits
  too (W8A8). `linnet.quant` quantizes checkpoints to int4 and int8 groups
  and imports GPTQ and AWQ.
- `linnet.train`: supervised fine-tuning, DPO and GRPO in PyTorch and JAX,
  with LoRA, fully sharded data parallelism, tensor parallelism (with
  `train`, and in pipelines) and checkpoints; the models
  also train under transformers' `Trainer` and TRL.
- `linnet.nest`: models by Nest name, from any Hugging Face Hub repo with a
  card, or from a directory; `transformers` checkpoints of the Llama,
  Mistral, Qwen2, Qwen3, Phi-3 and GPT-2 families convert on the way.
- Platform wheels for Linux (x86_64, aarch64), macOS (arm64) and Windows
  (x64) that carry the compiler and its standard library.
