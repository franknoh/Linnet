# Command-line tooling

## `linnet check`

```bash
linnet check [--strict] [--json] [--std <dir>] <path>...
```

Checks the given files — directories are searched recursively for `.linnet`
files — together with every module they import. Checking never executes
anything from a package.

| Exit status | Meaning |
| --- | --- |
| 0 | no errors (warnings are allowed unless `--strict`) |
| 1 | errors were reported, or warnings under `--strict` |
| 2 | the command line was wrong |

`--strict` changes only which findings fail the command. It never changes what
a program means.

### Warnings

| Code | Meaning |
| --- | --- |
| W1001 | an import that is never used |
| W1002 | a local binding that is never used; prefix its name with `_` to keep it |
| W1003 | a `param`, `buffer`, or `sub` that its block never uses |

Warnings are reported only when a program has no errors. The full list of
diagnostic codes is in [diagnostics/](diagnostics/README.md).

### JSON output

`--json` prints one JSON document on standard output instead of text on
standard error:

```json
{
  "version": 1,
  "diagnostics": [
    {
      "code": "E2103",
      "severity": "error",
      "message": "tensor dtypes do not match",
      "label": "",
      "location": {
        "file": "model.linnet",
        "start": { "line": 7, "column": 12, "offset": 141 },
        "end": { "line": 7, "column": 17, "offset": 146 }
      },
      "related": [],
      "notes": ["left operand has dtype bf16", "right operand has dtype f32"],
      "help": ["Linnet does not implicitly promote tensor dtypes; use an explicit cast"]
    }
  ],
  "summary": { "errors": 1, "warnings": 0 }
}
```

- `severity` is `error`, `warning`, or `note`.
- `location` is `null` for findings that do not belong to a source position.
- Lines and columns are 1-based; columns count Unicode code points; `offset`
  is a 0-based byte offset; `end` is exclusive.
- `label` annotates the primary location; `related` lists secondary locations
  with their own messages.
- Fields may be added in later versions; `version` changes only when existing
  fields change meaning.

## `linnet lint`

`linnet lint` is `linnet check --strict`: the same analysis, with warnings
failing the command. It exists so that scripts and editors can name the intent.

## `linnet spec-test`

```bash
linnet spec-test [--std <dir>] spec-tests/
```

Runs the executable specification: every `[[case]]` in `manifest.toml` must be
accepted (`expect = "ok"`) or rejected with exactly its `code`, and each
snapshot in `diagnostics/` must match the rendered errors byte for byte. This
is a compiler developer's command; `ctest` runs it too.

## `linnet lsp`

```bash
linnet lsp --stdio [--std <dir>]
```

The language server, speaking JSON-RPC over standard input and output. It
provides diagnostics (published on open and on every change), hover, go to
definition, references, rename, document and workspace symbols, completion,
whole-document formatting, semantic tokens, and inlay hints for inferred
binding types. Each open document is analyzed with the modules it imports;
unsaved contents of other open documents are used in place of the files on
disk. The VS Code extension in `editors/vscode` and the Neovim snippet in
`editors/vim/README.md` launch it.

## `linnet fmt`

```bash
linnet fmt <path>...        # rewrite files in place
linnet fmt --check <path>...
linnet fmt - < in.linnet    # standard input to standard output
```

There is one style and no options. Files with syntax errors are never
rewritten. A list that ends with a trailing comma stays one element per line.

## `linnet inspect`

```bash
linnet inspect --tokens file.linnet
linnet inspect --ast file.linnet
linnet inspect --core-ir [--std <dir>] file.linnet
linnet inspect --parameters [--json] file.linnet
```

`--tokens`, `--ast`, and `--core-ir` are compiler-internal views for
debugging; their format is not stable. `--core-ir` prints the Core Tensor IR
of every function, op, entry, and block method of the file and the modules it
imports, after running the IR verifier.

`--parameters` prints the parameter manifest of every block declared in the
file: one line per `param` or `buffer`, with its path from the block, its
tensor type, and the array lengths along the path. The block's own generic
parameters stay symbolic; sub-block arguments are substituted.

```text
examples.model::Model<H, Inner, Layers, Vocab>
  param embedding: Tensor[Vocab, H; bf16]
  param layers[*].up.weight: Tensor[Inner, H; bf16] x Layers
  param head.bias: Tensor[Vocab; bf16]?
```

`--json` gives the same data as `{"version": 1, "blocks": [...]}`, where each
block has `name`, `module`, `generics`, and `entries` with `path`, `kind`,
`dtype`, `shape`, `repeat`, and `optional`. `[*]` in a path stands for every
index of a sub-block array and `repeat` lists the array lengths, outermost
first. The manifest contains no tensor data.
