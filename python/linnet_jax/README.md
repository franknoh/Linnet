# linnet-jax

Runs a Linnet model as a JAX function and exports a JAX function as Linnet
source.

```python
from linnet_jax import load, export_linnet

model = load("src/model.linnet", generics={"H": 64, "Layers": 2, "Vocab": 100}, weights="weights/")
logits = jax.jit(model)(tokens)  # inference; composes with jit

export_linnet(forward, params, (tokens,), output="src/model.linnet")

module = load_nnx("src/model.linnet", generics={...}, weights="weights/")  # a Flax NNX module
```

`load` asks the compiler for the StableHLO of the entry (`linnet stablehlo`)
with the input shapes it is called with, binds SafeTensors weights by
parameter path, and wraps the module as a `jax.export.Exported` so it runs
under XLA like any exported JAX function. `load_nnx` mirrors the block
hierarchy as a Flax NNX module whose state paths are the parameter paths;
`load_source` runs entries as generated `jax.numpy` code that `jax.grad`
differentiates, for training.
`export_linnet` captures a function with `jax.export`, translates the
StableHLO it produces into a Core IR plan (recognizing the decompositions of
`softmax`, `rms_norm`, and the activations along the way), and has
`linnet emit` print the source; `import_stablehlo` does the same for a
StableHLO module produced elsewhere. See `docs/jax.md`.
