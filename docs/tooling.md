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

## `linnet plan`

```bash
linnet plan [--root <Block>] [--std <dir>] file.linnet
```

Checks the file and everything it imports, lowers it to Core IR, and prints
the plan of the root block (the only block with entries, or `--root`) as JSON
on standard output; see [plan-format.md](plan-format.md). The plan is what
`linnet_torch` consumes. It contains no tensor data.

## `linnet emit`

```bash
linnet emit plan.json
linnet emit - < plan.json
```

The other direction: reads a plan document and prints the Linnet source of
its module, formatted. Only the declarations of the plan's own module are
printed; functions and blocks it references from other modules become `use`
lines, and calls to them keep their explicit generic arguments, so a plan
that carries only its own functions (as a framework adapter writes) emits a
module that checks against the libraries it uses. Values nothing uses are
dropped first; otherwise emitting the plan of a file prints the same text as
`linnet inspect --emit` on that file. The command reads JSON only; it never
executes anything.

## `linnet stablehlo`

```bash
linnet stablehlo [--root <Block>] [--entry <name>] [--bind <G>=<value>]...
                 [--optionals present|absent] [--std <dir>] file.linnet
```

Prints one entry of the root block as a StableHLO module (MLIR text) on
standard output. The function is `@main`; its arguments are the entry's
inputs followed by every parameter and buffer of the block hierarchy in
manifest order, each carrying a `linnet.path` attribute with its parameter
path, so a runtime binds weights by name and the module itself holds no
tensor data. Shapes are static: every generic of the root block and of the
entry needs `--bind` (defaults apply), and optional parameters are all absent
unless `--optionals present`. Calls are inlined, `static for` is unrolled, and
index notation becomes broadcasts, gathers, and reductions over the output
grid. What StableHLO cannot express as captured is reported as an error, never
approximated. `python/linnet_torch/tests/test_stablehlo.py` compiles the output
with XLA and checks it against the PyTorch materializer.

Both exporters take `--numerics exact|equivalent` (default `equivalent`):
library operations whose selected implementation the format has an operator
for are spelled with it — contractions (`std.linalg::matmul`, `linear`) as
`dot_general` / `MatMul`, attention as two contractions around a softmax,
and in ONNX `Softmax`, `LayerNormalization`, `Gelu`, `Sigmoid`, `Relu` —
while `exact` keeps every canonical body. `linnet explain` lists the
choices.

## `linnet onnx`

```bash
linnet onnx [same options as stablehlo] file.linnet
```

The same export in the ONNX text format: a `main` graph whose inputs are the
entry's inputs followed by the parameters as `param<N>`, with `metadata_props`
mapping each `linnet.path.param<N>` to its parameter path. `python/linnet_onnx`
imports the result back (see [onnx.md](onnx.md)).

## `linnet torch`

```bash
linnet torch --std stdlib --bind H=8 --bind Heads=2 --bind B=2 --bind S=5 --bind T=f32 model.linnet
```

The same options as `linnet stablehlo`, printing the entry as a Python module
of straight-line PyTorch code: `main(*inputs, *parameters, *states)` returns
the entry's results followed by the assigned `state` members, with the
argument and result order given by the module's `PARAMETERS`, `STATES`, and
`NEXT_STATES` lists (parameter paths) and `RESULTS`. Semantic calls whose
selected candidate is a PyTorch kernel (`--numerics equivalent`, the
default; `exact` keeps every canonical body) become that library call —
`torch.softmax`, `F.scaled_dot_product_attention`, `torch.rms_norm`, ... —
and everything else is the static-shape lowering the other exporters use.
`linnet_torch.load(..., compile=True)` runs this per input shape and
executes the result, which `torch.compile` can then trace whole.

## `linnet jax`

The same options as `linnet torch`, printing the entry as a module of
straight-line `jax.numpy` code with the same `main`/`PARAMETERS`/`STATES`/
`NEXT_STATES`/`RESULTS` contract. The result is ordinary JAX: `jax.jit`,
`jax.grad`, and `jax.vmap` apply, `while` becomes `jax.lax.while_loop`, and
library operations with a selected candidate become `jax.nn` calls or
`jnp.matmul`. `linnet_jax.load_source(...)` runs it per input shape and is
the way to train a Linnet model in JAX. The generated module enables 64-bit
integers (`jax_enable_x64`) because Linnet's `i64` needs them.

## `linnet explain`

```bash
linnet explain [--std <dir>] file.linnet
```

Lists every semantic operation the program calls, where it is called from,
the implementations a backend could use for it, which one is selected, and
why. `--numerics exact` (the default) selects each operation's canonical
decomposition — its own `.linnet` body — for every call. `--numerics
equivalent` also allows PyTorch library calls that agree with the
decomposition up to floating-point rounding, and reports the strongest one
allowed. The same policy switch exists on `linnet plan`. Optimization is meant
to be inspectable, and this is where it is inspected.

## Optimization

`linnet plan`, `linnet explain`, `linnet stablehlo`, and `linnet inspect
--core-ir -O` run the canonical passes — identity `reshape`/`broadcast_to`/
`cast`/`permute` removal, composition of nested permutations, double
negation, integer constant folding, common subexpression elimination, and
dead-code elimination — followed by equality saturation. Saturation builds an
e-graph of the region-free operations of each block, applies rewrite rules
until nothing new appears (or a budget is spent), and extracts the cheapest
term of every value under a shape-based cost model; the block is then rebuilt
in dependency order. The IR verifier runs after each pass and a failure aborts
the compilation. `--no-optimize` disables all of it.

Every rule declares how far it may change floating-point results, and the
`--numerics` policy decides which rules run:

| policy | rules |
| --- | --- |
| `exact` (default) | view composition, double negation, commutativity, `x * 1`, `select(c, x, x)` |
| `equivalent` | also `x + 0` (sign of zero), reassociation of `+` and `*`, factoring `a*c + b*c` |

Rewrites never look inside comprehensions, reductions, or matches, and never
merge values of different types.

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
linnet inspect --core-ir [-O] [--std <dir>] file.linnet
linnet inspect --emit [-O] [--std <dir>] file.linnet
linnet inspect --parameters [--json] file.linnet
```

`--tokens`, `--ast`, and `--core-ir` are compiler-internal views for
debugging; their format is not stable. `--core-ir` prints the Core Tensor IR
of every function, op, entry, and block method of the file and the modules it
imports, after running the IR verifier; `-O` runs the canonical optimizer
passes first.

`--emit` reconstructs Linnet source for the file's declarations from that IR:
functions, ops, entries, blocks with their members and methods, constants,
enums, and structs; imports of the same package are spelled `crate.`. The
result is a complete module that checks and lints clean and that emits itself
again unchanged, so it is the canonical form a program takes after lowering.
SSA values become `let` bindings named after their source names; values used
once are written inline; index notation is rebuilt from comprehension and
reduction regions; calls spell their generic arguments explicitly and pass
every argument, since defaults are resolved before lowering. The same emitter
is how programs imported from other frameworks become `.linnet` files.

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
