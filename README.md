# Linnet

Linnet is a typed tensor language. Models are written as `.linnet` source files,
checked statically for shape and dtype correctness, and kept separate from their
weights. Inspecting or checking a Linnet package never executes package code.

The project is in early development. The toolchain currently provides the
frontend: parsing, canonical formatting, and static checking of names, dtypes,
symbolic shapes, and tensor index notation. There is no backend yet.

- `spec/` is the normative language specification; `spec/grammar.ebnf` is the
  consolidated grammar.
- `spec-tests/` is the executable specification: programs that must be accepted
  or must be rejected with a specific diagnostic code.
- `examples/` contains sample Linnet sources.

## Usage

```bash
linnet check src/model.linnet   # check files and every module they import
linnet check .                  # check every .linnet file below a directory
linnet fmt src/                 # format files in place (directories recurse)
linnet fmt --check .            # exit 1 if anything would change; for CI
linnet fmt - < in.linnet        # format stdin to stdout; for editors
linnet inspect --ast file.linnet
linnet inspect --tokens file.linnet
```

`check` finds imported modules by fixed rules: `crate.a.b` is `src/a/b.linnet` of
the package containing `linnet.toml`, and `std.a.b` is `a/b.linnet` in the
standard library directory given by `--std <dir>` or `LINNET_STD`.

Exit status is 0 on success, 1 when the input has errors (or `--check` finds
unformatted files), and 2 for command-line mistakes. Files with syntax errors
are never rewritten.

The formatter has one style and no options: 4-space indentation, 100 columns,
one statement per line. A list that ends with a trailing comma stays one element
per line; otherwise lists are joined when they fit. Comments are always kept;
a comment written in the middle of an expression moves to the nearest line
boundary.

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
