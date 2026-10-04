# JAX and Flax

`linnet.jax` runs Linnet entries in JAX and Flax and exports JAX functions
as Linnet source.

## Install

```bash
cd python/linnet && uv sync --extra flax    # or: pip install ".[flax]"
```

## Loaders

| Function | Runs the entry as | Differentiable |
| --- | --- | --- |
| `load` | a StableHLO module compiled by XLA (`jax.export.Exported`) | no |
| `load_source` | generated `jax.numpy` code under `jax.jit` | yes |
| `load_nnx` / `to_nnx` | a Flax NNX module over either of the above | with `load_source` |

## Load an entry

```python
import jax
from linnet.jax import load

f = load("src/model.linnet", generics={...}, weights="weights/", std_root="stdlib")
logits = jax.jit(f)(tokens)
```

`load` compiles the entry to StableHLO per combination of input shapes, for
any XLA backend. Weights bind by parameter path; an
optional parameter is compiled in only if the weights have it.

An entry that touches `state` takes `state=` and returns the new state:
`out, state = f(x, state=state)`, both mappings by parameter path. Missing
state starts at zeros.

## Load a model

```python
from linnet.jax import load_model

model = load_model("model.linnet", generics={..., "Batch": 1, "MaxSeq": 1024},
                   weights="model.safetensors")
logits = model.run_entry("prefill", [tokens, jnp.int32(0)])
logits = model.run_entry("decode", [token, jnp.int32(position)])
```

`load_model` loads every entry of the root block over one copy of the
weights. The block's `state`, such as KV caches, stays on the device in
`model.state` and is updated in place. Entries run as generated JAX
source; `generated=False` runs StableHLO instead. `linnet.serve.Engine`
takes such a model for continuous batching.

## Numerics policy

| `numerics=` | Effect |
| --- | --- |
| `"fast"` (default) | layer normalization and attention accumulate in the input dtype, as Flax reference models do on `bf16` |
| `"equivalent"` | f32 accumulation as the canonical bodies specify, and f32 products at full precision |
| `"exact"` | every canonical body, and f32 products at full precision |

`"fast"` keeps XLA's default for f32 products, which on an NVIDIA GPU
since Ampere is TF32, with errors near 1e-3. Every loader takes
`numerics=`.

## Multiple devices

`load_model(..., mesh=2)`, or a one-axis `jax.sharding.Mesh`, runs the
model tensor-parallel, splitting weights and KV caches by the
[`linnet.parallel` rules](torch.md#tensor-parallelism).
`rules={"*.lm_head.weight": 0}` adds or overrides rules by path pattern.

## Training

```python
from linnet.jax import load_source

f = load_source("src/model.linnet", generics={...}, weights="weights/", std_root="stdlib")
logits = f(tokens)                                   # same call as `load`

def loss(params):
    return cross_entropy(f.apply(params, tokens), labels)

grads = jax.grad(loss)(f.parameters)                 # ordinary JAX autodiff
```

`load_source` runs each entry as generated `jax.numpy` under `jax.jit`
(see `f.generated_source()`), so `jax.grad`, `jax.vmap` and any optimizer
over `f.parameters` work. It turns on `jax_enable_x64` for Linnet's `i64`.

For mixed precision, `load_source(..., generics={..., "T": "bf16"}, cast_dtype=True)`
keeps f32 parameters and casts them to `bf16` on every call, so gradients
come back in f32.

`linnet.jax.train`, `linnet.jax.dpo` and `linnet.jax.grpo` train a model's
packed entries for you, on one device or fully sharded over a mesh: see
[Training](training.md#jax).

## Functions

```python
from linnet.jax import load_function

cross_entropy = load_function("functions.linnet", "cross_entropy")

def loss(params):
    return cross_entropy(f.apply(params, tokens), labels)

grads = jax.grad(loss)(f.parameters)
```

`load_function` runs a module-level `entry`, a function of its inputs
alone such as a loss (`examples/09-functions`), under `jax.jit`. Generics bind from the inputs' shapes and dtypes, also under
`jax.grad`, `jax.jit` and `jax.vmap`. Name any the inputs leave open:
`positions(offset, N=8)`. It turns on `jax_enable_x64`, so `int64` inputs
made before the first call stay 64-bit.

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

Blocks become nested `nnx.Module`s: `param` is an `nnx.Param`, `buffer` an
`nnx.Variable`, `sub` a child module or an `nnx.List`, and an absent
optional parameter `None`. State paths are the parameter paths
(`layers/0/attention/q_proj/weight`). To train with `nnx.grad`, build it
with `to_nnx(load_source(...))`.

## JAX to Linnet

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

`export_linnet` captures `forward` with `jax.export` and writes Linnet
source. `import_stablehlo(text, params)` does the same for StableHLO text
from elsewhere.

| JAX | Linnet |
| --- | --- |
| parameter pytree | blocks: dict keys become members, lists and `0..n-1` dicts become sub arrays (`layers.0.q` is the pytree path) |
| `forward` | the root `entry` |
| operations | primitives or index notation |
| `jax.nn.softmax`, `sigmoid`, `silu`, tanh `gelu`, `x * rsqrt(mean(x * x) + eps) * w` | library calls (`rms_norm` for the last) |

`ExportResult.notes` lists what was recovered or dropped. An unmapped
operation stops the export with its name.

## Tests

The tests round-trip a `jnp` transformer, check `load` and `load_source`
against `jnp` references, thread state and `while` loops, and train with
`jax.grad` and `nnx.grad`.
