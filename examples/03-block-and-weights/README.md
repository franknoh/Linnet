# Block and weights

A small stack of blocks — `Linear`, `Mlp`, a `Model` with a sub-array of
layers — and the parameter manifest that a checkpoint must satisfy. It shows
how structure in source becomes paths in a SafeTensors file.

## Blocks own parameters, sub-blocks compose

`Model<H, Inner, Layers, Vocab, T>` declares `sub layers: [Mlp<H, Inner, T>;
Layers]`; each `Mlp` declares two `Linear` sub-blocks; each `Linear`
declares `weight` and an optional `bias`. `static for layer in layers`
applies them in order at compile time.

## Paths, not classes

```text
layers.0.up.weight
layers.0.down.weight
...
```

`linnet inspect --parameters` prints the manifest: every leaf with its shape
after the generics are substituted, and whether it is optional. Weights bind
by these names, so there is no model class to keep in sync with the file.

## Run it

```bash
linnet inspect --parameters examples/03-block-and-weights/model.linnet
```

```python
from linnet_torch import load
model = load("examples/03-block-and-weights/model.linnet",
             generics={"H": 8, "Inner": 16, "Layers": 2, "Vocab": 11, "T": "f32"},
             weights="weights/")          # SafeTensors named by the paths above
```
