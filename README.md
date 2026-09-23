# Linnet

A typed tensor language. A model is a `.linnet` file: its shapes and dtypes
are checked before anything runs, it carries no weights, and the same file
runs in PyTorch, JAX, XLA, and ONNX Runtime.

Documentation: [linnet.franknoh.dev](https://linnet.franknoh.dev) — installation,
a guide for PyTorch users, the language tour, examples, benchmarks, and the
specification. The site is built from `docs/`, `spec/`, and `examples/`.

```linnet
pub block Linear<In: Dim, Out: Dim, T: Float = bf16> {
    param weight: Tensor[Out, In; T]
    param bias: Tensor[Out; T]? = none

    pub fn forward<*S: Shape>(x: Tensor[*S, In; T]) -> Tensor[*S, Out; T] {
        return linear(x, weight, bias)
    }
}
```

```bash
linnet check src/model.linnet              # shapes, dtypes, index notation; nothing executes
linnet inspect --parameters src/model.linnet
linnet stablehlo --bind H=64 --bind B=1 --bind S=128 src/model.linnet
```

```python
from linnet.torch import load              # also linnet.jax.load / load_source / load_nnx
model = load("src/model.linnet", generics={"H": 64}, weights="weights/")
```

## What is here

| Directory | |
| --- | --- |
| `src/`, `include/` | the compiler: parser, checker, Core IR, optimizer, emitter, exporters, language server (C++23, no dependencies) |
| `stdlib/` | the standard library in Linnet: `std.linalg`, `std.nn` (linear, embedding, activations, softmax, norms, rope, attention, swiglu, argmax), `std.random`, `std.quant` |
| `spec/` | the normative specification and grammar; `spec-tests/` the executable cases |
| `examples/` | a linear layer up to Llama, GPT-2, ViT, and CLIP, each with a guide |
| `python/linnet` | the `linnet-lang` package: `linnet` (plans, checkpoints, the typed program, diagrams, the [Nest](https://github.com/franknoh/nest) zoo client, `linnet.triton` model repositories), `linnet.torch` (`load` interpreted or generated, training, `export_linnet`), `linnet.jax` (`load` on XLA, `load_source`, `load_nnx`, `export_linnet`, `import_stablehlo`), `linnet.onnx` (`import_onnx`) |
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
`scripts/check-format.sh --fix` reformats.

## Conventions

- Language changes update the `spec/` chapter, `spec/grammar.ebnf`, and a
  `spec-tests/` case together with the implementation.
- Diagnostic codes are stable and never reused
  (`include/linnet/diagnostic/codes.hpp`).
- The compiler depends on no tensor framework and knows no model; high-level
  operations live in `stdlib/`, and backends are consumers of the plan.
- Every `.linnet` file in the repository is formatter-clean; C++ warnings are
  errors in all presets.

## License

MIT; see [LICENSE](LICENSE).
