# Command line

`linnet` has one command per job. Every command but `serve` reads source
and writes text, and runs no model.

| Command | Job |
| --- | --- |
| `check`, `lint` | check files and the modules they import |
| `fmt` | format |
| `inspect` | parameter manifest, tokens, AST, Core IR, emitted source |
| `plan` | JSON plan for a materializer |
| `stablehlo`, `onnx`, `torch`, `jax` | export one entry with bound shapes |
| `emit` | Linnet source from a plan |
| `explain` | which implementation each library operation gets |
| `init` | create a package ([Modules and packages](modules-and-packages.md)) |
| `lsp` | language server |
| `serve` | serve a model over HTTP: `python -m linnet.serve` ([Serving](integrations.md#over-http)) |
| `memory`, `fit` | the memory a configuration needs, and the largest that fits a device ([Memory planning](memory.md)) |
| `spec-test` | run the executable specification |

Exit status is 0 on success, 1 for errors in the input, 2 for a bad command
line. `--std <dir>` or `LINNET_STD` sets the standard library directory.
`plan`, `explain`, and the exporters optimize the program first;
`--no-optimize` skips that.

## Files Linnet keeps

What Linnet downloads or measures goes under `LINNET_HOME`, `~/.linnet` by
default:

| Directory | Holds |
| --- | --- |
| `nest/` | Nest model directories fetched from the registry |
| `converted/` | Hub checkpoints converted to cards, by commit |
| `devices/` | device profiles measured on this machine (`--device local`) |
| `compiled/` | the Python package's compiler outputs, by everything they were compiled from, and the generated modules it imports |

Checkpoints stay in Hugging Face's cache (`HF_HOME`), shared with other
tools. Everything here can be deleted; it is downloaded or computed again
when needed.

## check and lint

```bash
linnet check [--strict] [--json] [--std <dir>] <path>...
linnet lint <path>...            # check --strict
```

Directories are searched for `.linnet` files. Warnings (W1001, W1002,
W1003) show only when there are no errors and fail only under `--strict`.
See [Diagnostics](diagnostics/README.md).

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

Lines and columns (code points) are 1-based, `offset` is a 0-based byte
offset, `end` is exclusive, and `location` is `null` without a position.
`version` changes only when an existing field changes meaning.

## fmt

```bash
linnet fmt <path>...          # rewrite in place
linnet fmt --check <path>...  # exit 1 if anything would change
linnet fmt - < in.linnet      # stdin to stdout
```

One style, no options: 4-space indentation, 100 columns, one statement per
line; a trailing comma keeps a list one element per line. Files with syntax
errors are never rewritten.

## inspect

```bash
linnet inspect --parameters [--json] file.linnet
linnet inspect --emit [-O] file.linnet
linnet inspect --core-ir [-O] file.linnet
linnet inspect --tokens file.linnet
linnet inspect --ast file.linnet
```

`--parameters` prints each block's `param`, `buffer`, and `state` paths and
types, with array lengths (`x Layers`):

```text
examples.model::Model<H, Inner, Layers, Vocab>
  param embedding: Tensor[Vocab, H; bf16]
  param layers[*].up.weight: Tensor[Inner, H; bf16] x Layers
  param head.bias: Tensor[Vocab; bf16]?
```

With `--json`: `{"version": 1, "blocks": [...]}` with `name`, `module`,
`generics`, and `entries` (`path`, `kind`, `dtype`, `shape`, `repeat`,
`optional`).

`--emit` prints the file back from Core IR as a complete module.
`--core-ir`, `--tokens`, and `--ast` are debugging views with no stable
format. `-O` optimizes first.

## plan and emit

```bash
linnet plan [--root <Block>] [--numerics exact|equivalent|fast] file.linnet
linnet emit plan.json        # or `linnet emit -` from stdin
```

`plan` prints the root block (the only block with entries, or `--root`) as
JSON: structure, parameter manifest, and Core IR. `--functions` prints the
module-level entries instead. See [Plan format](plan-format.md). `emit`
turns a plan back into formatted source.

## Exporting an entry

```bash
linnet stablehlo [--root <Block>] [--entry <name>] [--bind <G>=<value>]...
                 [--optionals present|absent] [--numerics exact|equivalent|fast] file.linnet
linnet onnx  ...same options...
linnet torch ...same options...
linnet jax   ...same options...
```

| Option | Meaning |
| --- | --- |
| `--root <Block>` | the root block; default: the only block with entries |
| `--entry <name>` | the entry to export |
| `--bind <G>=<value>` | a generic's value; defaults apply. Shape packs: `--bind S=2,3`, or `--bind S=` for none |
| `--optionals present\|absent` | optional parameters; default `absent` |
| `--numerics exact\|equivalent\|fast` | the numerics policy; default `equivalent` |
| `--grad` | the entry's gradient instead (see below) |

Every generic must be bound, and the values must satisfy the `where`
clauses (`H=5 Heads=2` fails `H % Heads == 0`). Anything the format cannot
express is an error, never an approximation.

A module-level entry exports the same way; name it with `--entry`. A file
with one such function and no block entries needs no `--entry`. Without
`--root`, an `--entry` name that both the module and a block declare is an
error.

| Format | Output |
| --- | --- |
| `stablehlo` | an MLIR module with `@main`; parameters carry `linnet.path`, state inputs `linnet.state`, assigned states are listed in `linnet.states` |
| `onnx` | ONNX text (`onnx.parser.parse_model`); parameters are `param<N>` inputs with `linnet.path.param<N>` metadata, states `state<N>` in and `next_state<N>` out |
| `torch` | a Python module: `main(*inputs, *parameters, *states, *constants)` with `PARAMETERS`, `STATES`, `NEXT_STATES`, `RESULTS`, and `constants(device)`, the input-independent tensors computed once per shape |
| `jax` | the same module in `jax.numpy`, differentiable with `jax.grad` |

### Gradients

`--grad` exports a loss and its gradient. The entry must return one
floating scalar. The export returns that loss, then its gradient with
respect to every floating parameter in the order the export takes them,
or, for a module-level entry, every floating input.

```bash
linnet onnx --grad --entry loss --bind N=32 model.linnet
```

The compiler writes the backward pass from the exported operations, so
ONNX and StableHLO models can be trained by a runtime with no autograd.
Each format lists the paths: `linnet.gradients` on `@main`,
`linnet.gradient.output<N>` metadata, or `GRADIENTS` in Python. Library
calls run as their canonical bodies. A `for` loop keeps each iteration's
starting values and runs its iterations backward; `while` loops have no
gradient yet. An entry that assigns `state` returns the states' new values
after the gradients, as its forward export does.

### Generated Python

`torch` and `jax` take further options, which the runtimes set when they
load a model. A block is named by its path: `layers.3`, `lm_head`.

| Option | Formats | Meaning |
| --- | --- | --- |
| `--prepare` | `torch`, `jax` | weight-only work moves into a `prepare` function, run once per loaded model |
| `--no-fuse` | `torch` | with `--prepare`, sibling linear layers stay separate products |
| `--place <block>=<slot>` | `torch` | the block runs on a device slot; a value crosses slots once |
| `--offload <block>` | `torch` | the block's parameters stay on the host and move to its slot when it runs |
| `--fully-shard <block>` | `torch`, `jax` | each process or device holds part of the block's parameters, gathered whole where the block runs (FSDP) |
| `--remat <block>` | `jax` | the backward pass computes each call of the block again instead of keeping its values (`jax.checkpoint`) |
| `--lora <pattern>` | `torch`, `jax` | a low-rank adapter beside every weight whose path matches the glob; takes `--lora-rank <r>` and scales by `--lora-alpha <a>` / r |
| `--absent <path>`, `--absent-file <file>` | all | optional parameters the weights lack, with `--optionals present`; `-` reads the file from stdin |

### Numerics policy

| Policy | Uses | Agrees with the canonical body |
| --- | --- | --- |
| `exact` | the canonical `.linnet` bodies | bit-exact |
| `equivalent` | the format's own operators (`MatMul`, `F.scaled_dot_product_attention`, ...) with f32 accumulation | up to floating-point rounding |
| `fast` | also softmax, normalization, and attention in the input dtype, as framework reference models do in `bf16` and `f16` | rounding of the input dtype |

Ops with a [kernel](kernels.md) launch it under `equivalent` and `fast`;
`exact` runs their bodies.

## explain

```bash
linnet explain [--numerics exact|equivalent|fast] file.linnet
```

Lists every library operation the program calls, its candidate
implementations, the one selected, and why. The default policy here is
`exact`.

## lsp

```bash
linnet lsp --stdio [--std <dir>]
```

JSON-RPC over stdio: diagnostics, hover, go to definition, references,
rename, symbols, completion, formatting, semantic tokens, and inlay hints.
Unsaved editor contents take precedence over files on disk.
`editors/vscode` and `editors/vim` launch it.

## spec-test

```bash
linnet spec-test [--std <dir>] spec-tests/
```

Runs the executable specification: each `[[case]]` in `manifest.toml` must
pass or fail with its code, and each snapshot in `diagnostics/` must match
byte for byte. `ctest` runs it too.
