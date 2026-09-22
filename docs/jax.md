# Running Linnet models in JAX

`python/linnet_jax` is the JAX adapter: it runs a Linnet entry as a JAX
function and exports a JAX function as Linnet source. Like the PyTorch
adapter, it contains no model-specific code; everything it knows about a model
comes from the compiler.

```bash
cd python/linnet_jax
uv sync --all-extras
```

## Linnet to JAX

```python
import jax
from linnet_jax import load

model = load(
    "examples/04-tiny-transformer/src/lib.linnet",
    generics={"Vocab": 100, "H": 64, "Heads": 4, "Inner": 256, "Layers": 2, "T": "f32"},
    weights="weights/",       # .safetensors file or directory, or {path: array}
    std_root="stdlib",
)
logits = jax.jit(model)(tokens, cos_table, sin_table)
```

`load` reads the plan of the root block to learn the entry's signature and
parameter paths. Each time the function is called with a new combination of
input shapes it asks the compiler for the StableHLO of the entry with those
dimensions bound (`linnet stablehlo`), checks that every parameter the module
names is present in the weights, and wraps the module as a
`jax.export.Exported`. Calls then run under XLA and compose with `jax.jit`.
Weights are plain array arguments of the wrapped module, bound by the
`linnet.path` names the compiler attaches. Optional parameters must be all
present or all absent. The wrapped module has no VJP: this is inference.

## JAX to Linnet

```python
from linnet_jax import export_linnet

export_linnet(
    forward,                 # forward(params, *inputs)
    params,                  # pytree of arrays: nested dicts, lists of alike subtrees
    (tokens,),               # example inputs fix shapes and dtypes
    output="src/model.linnet",
    weights="weights/",      # optional: SafeTensors under the parameter paths
    std_root="stdlib",
)
```

`export_linnet` captures `forward` with `jax.export`, which produces StableHLO
with static shapes, and translates that module operation by operation into a
Core IR plan: the parameter pytree becomes the block hierarchy (`param`
leaves, `sub` members for nested dicts, a sub array for a list of identical
subtrees, so parameter paths such as `layers.0.q` are the pytree paths),
`forward` becomes the root block's `entry`, and each StableHLO operation
becomes the Linnet primitive with the same meaning — elementwise arithmetic,
`reshape`/`permute`/`broadcast_to`, slices and concatenation — or index
notation: `dot_general` is a comprehension with a `sum` over the contracted
axes, `reduce` a comprehension over the kept axes, and the row-lookup form of
`gather` (`table[ids]`) an element lookup. Private helper functions JAX emits
are inlined. `linnet emit` prints the plan as formatted source, which is
checked before it is written.

The translation never guesses. An operation without a mapping (a custom call,
a `chlo` composite such as `erf`, a general `gather`) stops the export with a
message naming every such operation. When JAX's lowering carries something
Linnet has no need for — the NaN out-of-bounds guard of `jnp.take`, since
Linnet indices are checked — it is dropped and the fact is reported in
`ExportResult.notes`. Non-splat constants folded into the graph become
`buffer` members saved with the weights.

## Round trip

`tests/test_round_trip.py` exports a small transformer written with `jnp`,
lints and formats the result, loads it back with the exported weights, and
compares against the original under `jax.jit`; a second export is
byte-identical. It also runs the hand-written transformer example through
`load` and compares with a `jnp` reference of the same architecture, so the
same `.linnet` file now has PyTorch, XLA, and JAX executions that agree.

## Tests

```bash
uv run pytest
uv run pyright
uv run ruff check src tests && uv run ruff format --check src tests
```
