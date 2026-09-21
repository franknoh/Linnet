# Linnet

Linnet is a typed tensor language. Models are written as `.linnet` source files,
checked statically for shape and dtype correctness, and kept separate from their
weights. Inspecting or checking a Linnet package never executes package code.

The project is in early development; the toolchain currently provides the
foundations of the compiler frontend.

## Building

Requirements: CMake 3.25+, Ninja, and a C++23 compiler (GCC 13+, Clang 17+, or
MSVC 2022).

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Other presets: `release`, `sanitize` (ASan + UBSan), `tidy` (clang-tidy), and
`msvc` (Windows).

## Development

`scripts/check.sh [preset...]` runs the format check and then configures,
builds, and tests each preset (default `debug`). `scripts/check-format.sh --fix`
reformats the C++ sources.

Conventions:

- C++23, formatted with the repository `.clang-format`; warnings are errors in
  all presets.
- Public headers live under `include/linnet/<component>/`, implementations under
  `src/<component>/`, tests under `tests/`.
- The compiler core is framework-independent: it must not depend on tensor
  frameworks, and it must not contain model-specific logic.
- The CLI, language server, formatter, and backends all share the one frontend
  library.
