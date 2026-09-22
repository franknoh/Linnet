# linnet-jax

Runs a Linnet model as a JAX function and exports a JAX function as Linnet
source.

```python
from linnet_jax import load, export_linnet

model = load("src/model.linnet", generics={"H": 64, "Layers": 2, "Vocab": 100}, weights="weights/")
logits = jax.jit(model)(tokens)  # inference; composes with jit

export_linnet(forward, params, (tokens,), output="src/model.linnet")
```

`load` asks the compiler for the StableHLO of the entry (`linnet stablehlo`)
with the input shapes it is called with, binds SafeTensors weights by
parameter path, and wraps the module as a `jax.export.Exported` so it runs
under XLA like any exported JAX function. `export_linnet` captures a function
with `jax.export`, translates the StableHLO it produces into a Core IR plan,
and has `linnet emit` print the source; see `docs/jax.md`.
