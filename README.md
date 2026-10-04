# Linnet

**Linnet is the checked source format for neural network architectures.**

Think SafeTensors for model structure: the architecture lives in a typed
`.linnet` file and the weights in SafeTensors. Linnet checks and loads the
architecture without importing the model author's Python.

```bash
uv add "linnet-lang[torch]"                  # the compiler, its standard library and linnet.torch
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

- **Architecture is source, weights are data.** A `.linnet` file declares
  every parameter's shape and dtype. The PyTorch and ONNX loaders check each
  SafeTensors tensor against it before anything runs.
- **Git understands it.** `linnet fmt` has one style, so a diff shows only
  model changes. `linnet check` in CI fails a change that breaks a shape
  downstream.
- **Run the model, not its repository.** Loading reads declarative source
  and imports no code from the model's author, unlike
  `trust_remote_code=True`. It is not a sandbox, but the structure is known
  before anything runs.
- **Portable, not interpreted.** Each backend gets its native path:
  generated PyTorch under `torch.compile` or CUDA graphs, XLA, ONNX Runtime
  and TensorRT.

## Benchmarks

On one H100 in `bf16`:

| | Linnet | Reference stacks |
| --- | --- | --- |
| Llama 3.1 8B, decode one request | 167 tok/s (XLA), 161 (CUDA graphs) | vLLM 152, transformers compiled 110 |
| BERT base, forward at batch 1 | 0.74 ms (CUDA graphs) | transformers 3.60 |
| Llama 3.1 8B, many requests at once | 5600 tok/s | vLLM 5449 |
| Llama 3.1 8B, LoRA fine-tuning step | 1.04 s | TRL 1.83 s |
| Llama 3.1 8B, GRPO step | 1.18 s | TRL with vLLM 3.14 s |

Every row, including where Linnet loses, is on the
[benchmarks](https://linnet.franknoh.dev/benchmarks) page.

## Targets

- Load in PyTorch, JAX (XLA, `jax.numpy`, Flax NNX) and ONNX Runtime (CUDA,
  TensorRT).
- Export StableHLO, ONNX, a Triton Inference Server model, a transformers
  checkpoint for vLLM, and GGUF for llama.cpp and Ollama.
- Serve with `linnet.serve`: continuous batching and an OpenAI-compatible
  HTTP server.
- Train in PyTorch or JAX: supervised fine-tuning, DPO and GRPO, with LoRA
  or fully sharded across GPUs. Or hand the model to transformers' `Trainer`
  and TRL.
- Import from PyTorch, JAX, StableHLO and ONNX.
- Run in ComfyUI with [linnet-comfyui](https://github.com/franknoh/linnet-comfyui).

The [compatibility matrix](https://linnet.franknoh.dev/compatibility) lists
the entry points and limits of each target.

## Nest

[Nest](https://nest.franknoh.dev) is a registry of checked architectures and
SafeTensors checkpoints ([source](https://github.com/franknoh/nest)). Its CI
compiles every card, checks the published checkpoint against each
parameter's shape and dtype, and exports the model to StableHLO, ONNX,
PyTorch and JAX. `linnet.nest.load` loads a model by its Nest name, from any
Hugging Face Hub repo with a card at its root, or from a directory.

## Documentation

- [Installation](https://linnet.franknoh.dev/guide/installation) and the
  [quickstart](docs/getting-started.md)
- [Coming from PyTorch](https://linnet.franknoh.dev/guide/from-pytorch)
- [Language tour](docs/language-tour.md) and [specification](spec/)
- [Command line](docs/tooling.md)
- [PyTorch](docs/torch.md), [JAX](docs/jax.md), [ONNX](docs/onnx.md),
  [Nest](docs/nest.md), and [integrations](docs/integrations.md)
- [Training](docs/training.md)
- [Benchmarks](https://linnet.franknoh.dev/benchmarks)

## Contributing

Building needs CMake 3.25, Ninja, and a C++23 compiler (GCC 13, Clang 19,
MSVC 2022):

```bash
cmake --preset release && cmake --build --preset release && ctest --preset release
```

Presets: `debug`, `release`, `sanitize`, `tidy`, `fuzz` (Clang), `msvc`.
`scripts/check.sh [preset...]` runs the format check, build and tests, and
`scripts/check-format.sh --fix` reformats. Run the Python tests with
`pytest` in `python/linnet`.

- A language change updates its `spec/` chapter, `spec/grammar.ebnf` and a
  `spec-tests/` case with the implementation.
- Diagnostic codes are stable and never reused
  (`include/linnet/diagnostic/codes.hpp`).
- The compiler depends on no tensor framework and knows no model.
  High-level operations live in `stdlib/`.
- Every `.linnet` file in the repository is formatter-clean, and C++
  warnings are errors in all presets.

## License

MIT; see [LICENSE](LICENSE).
