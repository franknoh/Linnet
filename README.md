# Linnet

Linnet is a typed tensor language. Models are written as `.linnet` source files,
checked statically for shape and dtype correctness, and kept separate from their
weights. Inspecting or checking a Linnet package never executes package code.

The project is in early development. The toolchain currently provides the
frontend (parsing, formatting, static checking of names, dtypes, symbolic
shapes, and tensor index notation, packages, a language server, Core Tensor
IR, an optimizer) and two backends: `python/linnet_torch` materializes a
checked model as a `torch.nn.Module` with SafeTensors weights and exports a
`torch.nn.Module` back to Linnet source (see [docs/torch.md](docs/torch.md)),
`linnet stablehlo` prints an entry as a StableHLO module for XLA, and
`python/linnet_jax` runs models in JAX and exports JAX functions to Linnet
(see [docs/jax.md](docs/jax.md)).

- `spec/` is the normative language specification; `spec/grammar.ebnf` is the
  consolidated grammar.
- `spec-tests/` is the executable specification: programs that must be accepted
  or must be rejected with a specific diagnostic code.
- `stdlib/` is the standard library, written in Linnet: `std.linalg` and
  `std.nn` (`linear`, `embedding`, activations, `softmax`, `rms_norm`,
  `layer_norm`, `rope`, `attention`, `swiglu`). Every high-level operation is
  ordinary source that `linnet check` verifies like any other.
- `examples/` contains sample Linnet sources, from a linear layer to a small
  transformer built from the standard library (`09-tiny-transformer`) and
  models in the shape of Llama, GPT-2, and a Vision Transformer (`10`–`12`);
  see [examples/README.md](examples/README.md) for what each one shows.

[docs/getting-started.md](docs/getting-started.md) builds the toolchain, writes
a first model, and runs it in PyTorch; [docs/language-tour.md](docs/language-tour.md)
walks through the language.

## Usage

```bash
linnet check src/model.linnet   # check files and every module they import
linnet check .                  # check every .linnet file below a directory
linnet check --strict --json .  # for CI and editors; see docs/tooling.md
linnet fmt src/                 # format files in place (directories recurse)
linnet fmt --check .            # exit 1 if anything would change; for CI
linnet fmt - < in.linnet        # format stdin to stdout; for editors
linnet inspect --ast file.linnet
linnet plan --root Model model.linnet   # JSON plan for a materializer
linnet stablehlo --bind H=64 --bind B=1 --bind S=128 model.linnet   # StableHLO text
linnet emit plan.json                   # Linnet source back from a plan
linnet inspect --tokens file.linnet
```

`linnet init` creates a package (`linnet.toml` plus `src/lib.linnet`). Imports
are logical paths: `crate.a.b` is `src/a/b.linnet` of the package, `std.a.b`
lives in the standard library directory (`--std <dir>` or `LINNET_STD`), and
`dep.a.b` is found through the `[dependencies]` table. See
[docs/modules-and-packages.md](docs/modules-and-packages.md).

Exit status is 0 on success, 1 when the input has errors (or `--check` finds
unformatted files), and 2 for command-line mistakes. Files with syntax errors
are never rewritten.

The formatter has one style and no options: 4-space indentation, 100 columns,
one statement per line. A list that ends with a trailing comma stays one element
per line; otherwise lists are joined when they fit. Comments are always kept;
a comment written in the middle of an expression moves to the nearest line
boundary.

## Editors

`linnet lsp --stdio` is the language server. `editors/vscode` is a VS Code
extension that launches it and ships the syntax highlighting; `editors/vim` has
Vim and Neovim runtime files and the client setup; `editors/textmate` is the
standalone grammar other tools can reuse.

## Building

Requirements: CMake 3.25+, Ninja, and a C++23 compiler (GCC 13+, Clang 17+, or
MSVC 2022).

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Other presets: `release`, `sanitize` (ASan + UBSan), `tidy` (clang-tidy),
`fuzz` (libFuzzer targets, Clang only), and `msvc` (Windows).

## Development

`scripts/check.sh [preset...]` runs the format check and then configures,
builds, and tests each preset (default `debug`). `scripts/check-format.sh --fix`
reformats the C++ sources.

Conventions:

- C++23, formatted with the repository `.clang-format`; warnings are errors in
  all presets.
- Public headers live under `include/linnet/<component>/`, implementations under
  `src/<component>/`, tests under `tests/`.
- Language changes update the relevant `spec/` chapter, `spec/grammar.ebnf`, and
  at least one case in `spec-tests/` together with the implementation.
- Every `.linnet` file in the repository that parses must be formatter-clean.
- Diagnostic codes are stable; new ones are added to
  `include/linnet/diagnostic/codes.hpp` and never reused.
- The compiler core is framework-independent: it must not depend on tensor
  frameworks, and it must not contain model-specific logic.
- The CLI, language server, formatter, and backends all share the one frontend
  library.

## License

MIT; see [LICENSE](LICENSE).
