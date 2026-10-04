# Contributing

Bug reports, models, fixes and features are welcome. For a larger change,
open an issue first so the design can be settled before the code.

## Build

The compiler needs CMake 3.25, Ninja, and a C++23 compiler (GCC 13, Clang
19, MSVC 2022):

```bash
cmake --preset release && cmake --build --preset release && ctest --preset release
```

Presets: `debug`, `release`, `sanitize`, `tidy`, `fuzz` (Clang), `msvc`.
The Python package lives in `python/linnet` and uses
[uv](https://docs.astral.sh/uv/):

```bash
cd python/linnet
uv sync --all-extras
LINNET_BIN=../../build/release/linnet uv run pytest
```

## Checks

CI runs all of these; run the ones a change touches before opening a pull
request.

| Change | Check |
| --- | --- |
| C++ | `scripts/check.sh [preset...]` (format, build, tests); `scripts/check-format.sh --fix` reformats; the `tidy` preset runs clang-tidy |
| Python | `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run pyright`, `uv run pytest` |
| `.linnet` | `linnet fmt --check` and `linnet lint` |
| Docs and site | `npm run build` in `site/` fails on a dead link |
| VS Code extension | `npm run check` in `editors/vscode/` |

C++ warnings are errors in every preset, and every `.linnet` file in the
repository is formatter-clean.

## Conventions

- A language change updates its `spec/` chapter, `spec/grammar.ebnf` and a
  `spec-tests/` case along with the implementation.
- Diagnostic codes are stable and never reused
  (`include/linnet/diagnostic/codes.hpp`).
- The compiler depends on no tensor framework and knows no model.
  High-level operations live in `stdlib/`, written in Linnet.
- Documentation is short and concrete: what a thing does and how to use
  it, with numbers where there are numbers.

## Pull requests

- One change per pull request, with tests for what it adds or fixes.
- Commit subjects are one imperative line of at most 72 characters, such
  as "Bind optional biases when converting checkpoints".
- A pull request merges once every check passes.

## Models

Models go to [Nest](https://github.com/franknoh/nest), whose README lists
what a card needs; `python -m linnet.nest check` runs the same checks as
its CI. A model can also live in any Hugging Face Hub repo with a card at
its root (`python -m linnet.nest push`).
