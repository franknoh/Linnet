# Command line

One executable, `linnet`, with one command per job. Every command reads
source and writes text; none executes a model.

| Command | Job |
| --- | --- |
| `check`, `lint` | check files and the modules they import |
| `fmt` | format |
| `inspect` | parameter manifest, tokens, AST, Core IR, emitted source |
| `plan` | JSON plan for a materializer |
| `stablehlo`, `onnx`, `torch`, `jax` | export one entry with bound shapes |
| `emit` | Linnet source from a plan |
| `explain` | which implementation each library operation gets |
| `init` | create a package |
| `lsp` | language server |
| `spec-test` | run the executable specification |

Exit status is 0 on success, 1 when the input has errors, 2 for a bad
command line. `--std <dir>` names the standard library directory everywhere
it is needed; `LINNET_STD` does the same.

## check and lint

```bash
linnet check [--strict] [--json] [--std <dir>] <path>...
linnet lint <path>...            # check --strict
```

Directories are searched for `.linnet` files. Warnings are reported only when
there are no errors, and fail the command only under `--strict`.

| Warning | |
| --- | --- |
| W1001 | unused import |
| W1002 | unused local; prefix the name with `_` to keep it |
| W1003 | a `param`, `buffer`, or `sub` the block never uses |

`--json` prints one document instead of text:

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

Lines and columns are 1-based (columns count code points), `offset` is a
0-based byte offset, `end` is exclusive, and `location` is `null` for
findings without a position. Fields may be added; `version` changes only when
existing fields change meaning. All codes: [Diagnostics](diagnostics/README.md).

## fmt

```bash
linnet fmt <path>...          # rewrite in place
linnet fmt --check <path>...  # exit 1 if anything would change
linnet fmt - < in.linnet      # stdin to stdout
```

One style, no options: 4-space indentation, 100 columns, one statement per
line. A list that ends with a trailing comma stays one element per line.
Files with syntax errors are never rewritten.

## inspect

```bash
linnet inspect --parameters [--json] file.linnet
linnet inspect --emit [-O] file.linnet
linnet inspect --core-ir [-O] file.linnet
linnet inspect --tokens file.linnet
linnet inspect --ast file.linnet
```

`--parameters` prints the manifest of every block in the file: one line per
`param`, `buffer`, or `state`, with its path, type, and the array lengths
along the path (`x Layers`). Generics of the block stay symbolic.

```text
examples.model::Model<H, Inner, Layers, Vocab>
  param embedding: Tensor[Vocab, H; bf16]
  param layers[*].up.weight: Tensor[Inner, H; bf16] x Layers
  param head.bias: Tensor[Vocab; bf16]?
```

`--json` gives `{"version": 1, "blocks": [...]}` with `name`, `module`,
`generics`, and `entries` (`path`, `kind`, `dtype`, `shape`, `repeat`,
`optional`).

`--emit` prints the file's declarations back from Core IR: a complete module
that checks, lints, and emits itself again unchanged. This is the canonical
form after lowering, and the same emitter turns imported frameworks' graphs
into `.linnet` files.

`--core-ir`, `--tokens`, and `--ast` are compiler views for debugging; their
format is not stable. `-O` runs the optimizer first.

## plan and emit

```bash
linnet plan [--root <Block>] [--numerics exact|equivalent|fast] file.linnet
linnet emit plan.json        # or `linnet emit -` from stdin
```

`plan` prints the root block (the only block with entries, or `--root`) as a
JSON document: structure, parameter manifest, and the Core IR of every
function. See [Plan format](plan-format.md). `emit` is the other direction:
it prints the plan's own module as formatted source, turning references to
other modules into `use` lines. Both read and write text only.

## Exporting an entry

```bash
linnet stablehlo [--root <Block>] [--entry <name>] [--bind <G>=<value>]...
                 [--optionals present|absent] [--numerics exact|equivalent|fast] file.linnet
linnet onnx  ...same options...
linnet torch ...same options...
linnet jax   ...same options...
```

All four export one entry with every generic bound (`--bind`; defaults
apply). Calls are inlined, `static for` is unrolled, index notation becomes
broadcasts, gathers, and reductions, and `while` becomes the format's loop.
Anything the format cannot express is an error, never an approximation.

| Format | Output |
| --- | --- |
| `stablehlo` | an MLIR module with `@main`; parameters carry `linnet.path`, state inputs `linnet.state`, assigned states are listed in `linnet.states` |
| `onnx` | ONNX text (`onnx.parser.parse_model`); parameters are `param<N>` inputs with `linnet.path.param<N>` metadata, states `state<N>` in and `next_state<N>` out |
| `torch` | a Python module: `main(*inputs, *parameters, *states, *constants)` with `PARAMETERS`, `STATES`, `NEXT_STATES`, `RESULTS`, and `constants(device)`, the input-independent tensors (rotary tables, masks) computed once per shape |
| `jax` | the same module in `jax.numpy`, differentiable with `jax.grad` |

`--numerics equivalent` (the default) spells library operations with the
format's own operator when one exists: contractions as `dot_general` or
`MatMul`, attention as two contractions around a softmax, `torch.softmax`,
`F.scaled_dot_product_attention`, `jax.nn.silu`, and so on. `exact` keeps
every canonical body except the two that are the same numbers however they
are computed: `std.nn.embedding::embedding` (a gather) and
`std.nn.attention::causal_mask` (a boolean mask), which every policy takes
natively. A square causal mask reaches attention as `is_causal=True`, and
`std.nn.attention::grouped_attention` reaches it with the key/value heads
unrepeated. `fast` additionally lets layer normalization and attention
accumulate in the input dtype instead of the f32 the canonical bodies
specify; for `bf16` and `f16` this is what framework reference models do, and
results differ by rounding only. Softmax and RMS normalization are not in
that tier for PyTorch: those kernels accumulate in f32 whatever their input
dtype, so `equivalent` already selects them without casts. Optional parameters are absent
unless `--optionals present`.

| Tier | Selected implementations | Agreement with the canonical body |
| --- | --- | --- |
| `exact` | the canonical `.linnet` bodies | bit-exact |
| `equivalent` | library kernels with f32 accumulation | up to floating-point rounding |
| `fast` | layer normalization and attention in the input dtype | rounding of the input dtype |

## explain

```bash
linnet explain [--numerics exact|equivalent|fast] file.linnet
```

Lists every library operation the program calls, the implementations a
backend could use, which one is selected, and why. `exact` (the default here)
selects each operation's own body; `equivalent` allows kernels that agree up
to floating-point rounding; `fast` also allows the input-dtype variants.

## Optimization

`plan`, `explain`, the exporters, and `inspect -O` run the canonical passes
(identity view removal, permutation composition, double negation, integer
constant folding, common subexpression and dead code elimination) followed by
equality saturation: an e-graph of each block's region-free operations,
rewrite rules applied to saturation or a budget, and the cheapest term
extracted under a shape-based cost model. The IR verifier runs after each
pass. `--no-optimize` turns it all off.

| Policy | Rules |
| --- | --- |
| `exact` | view composition, double negation, commutativity, `x * 1`, `select(c, x, x)` |
| `equivalent` | also `x + 0`, reassociation of `+` and `*`, factoring `a*c + b*c` |

Rewrites never look inside comprehensions, reductions, or matches.

## lsp

```bash
linnet lsp --stdio [--std <dir>]
```

JSON-RPC over stdio: diagnostics on open and change, hover, go to
definition, references, rename, symbols, completion, formatting, semantic
tokens, and inlay hints. Unsaved contents of open documents take precedence
over files on disk. `editors/vscode` and `editors/vim` launch it.

## spec-test

```bash
linnet spec-test [--std <dir>] spec-tests/
```

Runs the executable specification: each `[[case]]` in `manifest.toml` must be
accepted or rejected with exactly its code, and each snapshot in
`diagnostics/` must match byte for byte. `ctest` runs it too.
