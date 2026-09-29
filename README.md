# Linnet

**Linnet is the checked source format for neural network architectures.**

Think SafeTensors for model structure: the architecture lives in a readable,
typed `.linnet` file, the weights stay in SafeTensors, and the architecture is
checked and loaded without importing the model author's Python.

```text
model.linnet          architecture
model.safetensors     weights
```

A Linnet model can be checked, diffed, reviewed, and versioned with Git, and
materialized into PyTorch, JAX and XLA, ONNX Runtime, and the serving and
deployment paths listed below.

```bash
linnet check model.linnet                    # shapes, dtypes, parameters; nothing runs
linnet inspect --parameters model.linnet     # every tensor a checkpoint must supply
```

```python
from linnet.torch import load

model = load("model.linnet", generics={"H": 512, "Heads": 8}, weights="model.safetensors")
```

Documentation: [linnet.franknoh.dev](https://linnet.franknoh.dev). Model
registry: [Nest](https://nest.franknoh.dev).

## Why Linnet

**Architecture is source, weights are data.** A `.linnet` file declares every
parameter with its shape and dtype. Loading binds SafeTensors to those paths;
the PyTorch and ONNX loaders check each tensor's shape and dtype before
anything runs.

**Git understands it.** `linnet fmt` has one style and no options, so a diff
shows what changed in the model: a head count, a RoPE base, a block. Review,
blame, branches, and tags work as they do for code, and `linnet check` in CI
fails a change that breaks a shape downstream.

**Run the model, not its repository.** A custom architecture on the Hub loads
by importing the Python its repository ships (`trust_remote_code=True`).
Checking and loading a `.linnet` file run the compiler over declarative
source; no code from the model's author is imported. This is not a sandbox —
the compiler and the frameworks are ordinary software — but the structure is
known before anything runs. Importing from PyTorch or JAX traces the model's
Python once, when the source is written.

**Portable does not mean interpreted.** The checked program is lowered to each
backend's native path: generated PyTorch under `torch.compile` or CUDA graphs,
XLA, ONNX Runtime and TensorRT. On one H100 in `bf16`, Llama 3.1 8B decodes one
request at 169 tokens per second under XLA (vLLM 157, transformers compiled
110) and BERT base runs a batch-1 forward pass in 0.74 ms as CUDA graphs
(transformers 3.60). vLLM stays ahead when serving many requests at once
(5658 against 4476 tokens per second for Llama 3.1 8B); every row, including
where Linnet loses, is on the
[benchmarks](https://linnet.franknoh.dev/benchmarks) page.

## Integrations

| Target | Entry point | Scope |
| --- | --- | --- |
| PyTorch | `linnet.torch.load`: interpreted, generated source, `torch.compile`, CUDA graphs; training, device placement, tensor parallelism | |
| JAX | `linnet.jax.load` (XLA), `load_source` (`jax.numpy`, `jax.grad`), `load_model`, `load_nnx` (Flax NNX) | |
| StableHLO | `linnet stablehlo` | static shapes |
| ONNX Runtime | `linnet onnx`, `linnet.onnx.export_model`, `load_model` (CUDA, TensorRT) | static shapes |
| Triton Inference Server | `python -m linnet.triton export`: ONNX or Python backend | ONNX backend: entries without state |
| vLLM and other transformers-checkpoint servers | `linnet.hf.export` | Llama and GPT-2 families |
| llama.cpp, Ollama | `linnet.gguf.export`: GGUF and a Modelfile | Llama and GPT-2 families |
| ComfyUI | [linnet-comfyui](https://github.com/franknoh/linnet-comfyui) | PyTorch |
| Serving | `linnet.serve`: continuous batching with sampling, and `python -m linnet.serve`, an OpenAI-compatible HTTP server | decoders with `prefill_slots` and `decode_rows` |
| Importers | `linnet.torch.export_linnet`, `linnet.jax.export_linnet`, `linnet.jax.import_stablehlo`, `linnet.onnx.import_onnx` | |

[Nest](https://github.com/franknoh/nest) is a registry of checked
architectures and SafeTensors checkpoints: CI compiles every card, checks the
published checkpoint's headers against every parameter's shape and dtype, and
exports each model to StableHLO, ONNX, PyTorch, and JAX. `linnet.nest.load`
loads a card by name.

## The language

A small typed tensor language: every dimension is a symbol, and shapes and
dtypes are checked before anything runs.

```linnet
pub block Linear<In: Dim, Out: Dim, T: Float = bf16> {
    param weight: Tensor[Out, In; T]
    param bias: Tensor[Out; T]? = none

    pub fn forward<*S: Shape>(x: Tensor[*S, In; T]) -> Tensor[*S, Out; T] {
        return linear(x, weight, bias)
    }
}
```

Blocks, generics over dimensions and dtypes, `where` constraints, index
notation, `static for` and `while`, and `state` for caches, with a standard
library written in Linnet itself. The [language tour](docs/language-tour.md)
and the [specification](spec/) cover the rest.

## Repository

| Directory | |
| --- | --- |
| `src/`, `include/` | the compiler: parser, checker, Core IR, optimizer, emitter, exporters, language server (C++23, no dependencies) |
| `stdlib/` | the standard library in Linnet: `std.linalg`, `std.nn` (linear, embedding, activations, softmax, norms, rope, attention, MLPs, decoding, KV caches, convolution, pooling, resizing), `std.random`, `std.quant` |
| `spec/` | the normative specification and grammar; `spec-tests/` the executable cases |
| `examples/` | a linear layer up to Llama, GPT-2, ViT, and CLIP, each with a guide |
| `python/linnet` | the `linnet-lang` package: plans, checkpoints, the typed program, diagrams, the Nest client, `linnet.serve`, `linnet.triton`, `linnet.hf`, `linnet.gguf`, and the `linnet.torch`, `linnet.jax`, and `linnet.onnx` backends |
| `editors/` | VS Code extension, Vim runtime files, TextMate grammar |
| `bench/` | the benchmark harness and published results |
| `site/` | the documentation site |

## Commands

```bash
linnet check [--strict] [--json] <path>...     # check files and their imports
linnet fmt [--check] <path>...                 # one style, no options
linnet inspect --parameters|--emit|--core-ir <file>
linnet plan [--root <Block>] <file>            # JSON plan for a materializer
linnet stablehlo|onnx|torch|jax --bind <G>=<v>... <file>
linnet emit plan.json                          # Linnet source from a plan
linnet explain <file>                          # which kernel each library operation gets
linnet init <dir>                              # new package
linnet lsp --stdio                             # language server
linnet spec-test <dir>                         # run specification tests
```

`--std <dir>` or `LINNET_STD` names the standard library directory. Exit
status is 0 on success, 1 on errors, 2 for a bad command line. See
[docs/tooling.md](docs/tooling.md).

## Building

CMake 3.25, Ninja, and a C++23 compiler (GCC 13, Clang 19, MSVC 2022).

```bash
cmake --preset release && cmake --build --preset release && ctest --preset release
```

Presets: `debug`, `release`, `sanitize`, `tidy`, `fuzz` (Clang), `msvc`.
`scripts/check.sh [preset...]` runs the format check, build, and tests;
`scripts/check-format.sh --fix` reformats. The Python package's tests run
with `pytest` from `python/linnet`.

## Conventions

- Language changes update the `spec/` chapter, `spec/grammar.ebnf`, and a
  `spec-tests/` case together with the implementation.
- Diagnostic codes are stable and never reused
  (`include/linnet/diagnostic/codes.hpp`).
- The compiler depends on no tensor framework and knows no model; high-level
  operations live in `stdlib/`, and backends are consumers of the plan.
- Every `.linnet` file in the repository is formatter-clean; C++ warnings are
  errors in all presets.

## Documentation

- [Installation](https://linnet.franknoh.dev/guide/installation) and the
  [quickstart](docs/getting-started.md)
- [Coming from PyTorch](https://linnet.franknoh.dev/guide/from-pytorch)
- [Language tour](docs/language-tour.md) and [specification](spec/)
- [PyTorch](docs/torch.md), [JAX](docs/jax.md), [ONNX](docs/onnx.md),
  [Nest](docs/nest.md), and [integrations](docs/integrations.md)
- [Benchmarks](https://linnet.franknoh.dev/benchmarks)

## License

MIT; see [LICENSE](LICENSE).
