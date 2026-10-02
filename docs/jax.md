# JAX and Flax

`linnet.jax` runs a Linnet entry in JAX three ways and exports JAX
functions as Linnet. Like the PyTorch adapter, it has no model-specific
code.

```bash
cd python/linnet && uv sync --extra flax    # or: pip install ".[flax]"
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
bound by the `linnet.path` names. Each optional parameter is compiled in
when the weights have it and out when they do not, one parameter at a time.

An entry that touches `state` is called with `state=` and returns the new
state: `out, state = f(x, state=state)`, both mappings by parameter path,
missing inputs starting at zeros.

`numerics` defaults to `"fast"`: layer normalization and attention
accumulate in the input dtype, as Flax reference models do on `bf16`.
`"equivalent"` keeps the f32 accumulation the canonical bodies specify, and
`"exact"` keeps every canonical body. Both also ask XLA for f32 products
(matrix products, convolutions, attention) at full precision, where its
default on an NVIDIA GPU since Ampere is TF32, with errors near 1e-3;
`"fast"` leaves the default. All three loaders take it.

## load_model

```python
from linnet.jax import load_model

model = load_model("model.linnet", generics={..., "Batch": 1, "MaxSeq": 1024},
                   weights="model.safetensors")
logits = model.run_entry("prefill", [tokens, jnp.int32(0)])
logits = model.run_entry("decode", [token, jnp.int32(position)])
```

`load` is one entry; `load_model` is every entry of the root block over one
copy of the weights on the device, with the block's `state` (a decoder's KV
caches) kept there between calls in `model.state`. Each state an entry
replaces is donated to it, so XLA writes the new cache into the old one's
memory. The entries run as generated JAX source (`generated=False` runs
the StableHLO export instead). `linnet.serve.Engine` takes such a model for
continuous batching.

`load_model(..., mesh=2)` (or a one-axis `jax.sharding.Mesh`) runs the model
tensor-parallel: each weight is placed split over the devices as
`linnet.parallel` says -- projections into heads and feed-forward widths by
output, projections back by input, everything else copied -- and XLA
partitions every entry, adding the collectives the split needs. KV caches
are split by heads. `rules={"*.lm_head.weight": 0}` adds or overrides rules by
path pattern.

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

Mixed precision takes f32 master parameters and computes in bf16:
`load_source(..., generics={..., "T": "bf16"}, cast_dtype=True)` casts the
parameters given to `apply` to `bf16` on every call, so `jax.grad` returns
gradients in f32 for an f32 optimizer state. `numerics="equivalent"` keeps
softmax, normalization, and attention accumulating in f32.
## load_function

```python
from linnet.jax import load_function

cross_entropy = load_function("functions.linnet", "cross_entropy")

def loss(params):
    return cross_entropy(f.apply(params, tokens), labels)

grads = jax.grad(loss)(f.parameters)
```

An `entry` declared at module level, outside any block, is a function of
its inputs alone: a loss, a preprocessing step, a reward
(`examples/09-functions`). `load_function` runs one as `jax.numpy` code
that `linnet jax` generates for each binding of its generics, under
`jax.jit`. The generics are bound from the inputs' shapes and dtypes, also
inside `jax.grad`, `jax.jit`, and `jax.vmap`, which hand the function the
abstract arrays they trace; one the inputs leave open is given by name
(`positions(offset, N=8)`). `load_function` turns on `jax_enable_x64`, so
`int64` inputs made before the first call stay 64-bit.

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
