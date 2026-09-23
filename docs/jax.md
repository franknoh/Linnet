# JAX and Flax

`linnet.jax` runs a Linnet entry in JAX three ways and exports JAX
functions as Linnet. Like the PyTorch adapter, it has no model-specific
code.

```bash
cd python/linnet && uv sync --extra flax    # or: pip install "linnet-lang[flax]"
```

| Function | Runs the entry as | Differentiable |
| --- | --- | --- |
| `load` | a StableHLO module compiled by XLA (`jax.export.Exported`) | no |
| `load_source` | generated `jax.numpy` code under `jax.jit` | yes |
| `load_nnx` / `to_nnx` | a Flax NNX module over either of the above | with `load_source` |

## load

```python
import jax
from linnet.jax import load

f = load("src/model.linnet", generics={...}, weights="weights/", std_root="stdlib")
logits = jax.jit(f)(tokens)
```

Each new combination of input shapes asks the compiler for the entry's
StableHLO with those dimensions bound (`linnet stablehlo`), checks that every
parameter it names is in the weights, and wraps the module as an `Exported`
that runs on any XLA backend. Weights are device arrays passed as arguments,
bound by the `linnet.path` names. Optional parameters must be all present or
all absent.

An entry that touches `state` is called with `state=` and returns the new
state: `out, state = f(x, state=state)`, both mappings by parameter path,
missing inputs starting at zeros.

`numerics` defaults to `"equivalent"`: library operations become the
StableHLO spellings XLA fuses well. `"exact"` keeps every canonical body;
`"fast"` lets softmax, normalization, and attention accumulate in the input
dtype, as Flax reference models do on `bf16`. All three loaders take it.

## load_source and training

```python
from linnet.jax import load_source

f = load_source("src/model.linnet", generics={...}, weights="weights/", std_root="stdlib")
logits = f(tokens)                                   # same call as `load`

def loss(params):
    return cross_entropy(f.apply(params, tokens), labels)

grads = jax.grad(loss)(f.parameters)                 # ordinary JAX autodiff
```

`load_source` compiles each entry with `linnet jax` instead: a module of
straight-line `jax.numpy` (see `f.generated_source()`), run under `jax.jit`.
Because it is plain JAX, `jax.grad`, `jax.vmap`, and any optimizer over the
`f.parameters` dict work. `while` becomes `jax.lax.while_loop`, library
operations become `jax.nn` calls, and state is threaded as with `load`. The
generated module turns on `jax_enable_x64`, which Linnet's `i64` needs.

## Flax NNX

```python
from flax import nnx
from linnet.jax import load_nnx

model = load_nnx("src/model.linnet", generics={...}, weights="weights/", std_root="stdlib")
nnx.display(model)
```

```text
Model( # Param: 38,570,496 (77.1 MB)
  embedding=Embedding( # Param: 16,384,000 (32.8 MB)
    weight=Param( # 16,384,000 (32.8 MB)
      value=Array(shape=(32000, 512), dtype=dtype(bfloat16))
    )
  ),
  layers=List([
    DecoderLayer( # Param: 2,900,992 (5.8 MB)
      attention_norm=RmsNorm( # Param: 512 (1.0 KB)
        weight=Param( # 512 (1.0 KB)
          value=Array(shape=(512,), dtype=dtype(bfloat16))
        )
      ),
      attention=GroupedQueryAttention( # Param: 786,432 (1.6 MB)
        q_proj=Linear( # Param: 262,144 (524.3 KB)
          weight=Param( # 262,144 (524.3 KB)
            value=Array(shape=(512, 512), dtype=dtype(bfloat16))
          ),
          bias=None
        ),
        ...
```

The block hierarchy becomes nested `nnx.Module`s named after the blocks:
`param` is an `nnx.Param`, `buffer` an `nnx.Variable`, `sub` a child module
or an `nnx.List`. State paths are the parameter paths (`layers/0/attention/
q_proj/weight`), so `nnx.split`, checkpoints, and sharding see the same
names as every other backend. Absent optional parameters are `None`. Calling
the module runs the entry on the arrays it currently holds; built on
`load_source` (`to_nnx(load_source(...))`), `nnx.grad` trains it.

## Exporting a JAX function

```python
from linnet.jax import export_linnet

export_linnet(
    forward,                 # forward(params, *inputs)
    params,                  # pytree: nested dicts, lists of alike subtrees
    (tokens,),               # example inputs fix shapes and dtypes
    output="src/model.linnet",
    weights="weights/",      # optional: SafeTensors under the parameter paths
    std_root="stdlib",
)
```

`export_linnet` captures `forward` with `jax.export` and translates the
StableHLO into a plan: the parameter pytree becomes the block hierarchy
(dict keys become members, lists and `0..n-1` dicts become sub arrays, so
`layers.0.q` is the pytree path), `forward` becomes the root `entry`, and
each operation becomes the primitive or index notation with the same meaning
(`dot_general` is a `sum` comprehension, `reduce` a comprehension over the
kept axes, row `gather` an element lookup). `import_stablehlo(text, params)`
does the same for StableHLO text produced elsewhere.

Decompositions JAX produces are recognized and emitted as library calls:
`jax.nn.softmax`, `sigmoid`, `silu`, the tanh `gelu`, and `x * rsqrt(mean(x
* x) + eps) * w` as `rms_norm`. The match is structural; anything else, and
anything dropped (the NaN guard of `jnp.take`, which Linnet does not need),
is listed in `ExportResult.notes`. Unmapped operations stop the export by
name.

## Tests

`tests/` export a `jnp` transformer, load it back, and compare under
`jax.jit`; run the hand-written examples through `load` and `load_source`
against `jnp` references and the StableHLO path; thread state and `while`
loops; and train an MLP with `jax.grad` and an NNX module with `nnx.grad`.
