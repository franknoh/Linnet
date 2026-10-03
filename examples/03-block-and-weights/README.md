# Block and weights

Nested blocks and the parameter manifest a checkpoint must satisfy. Block
structure in source becomes the tensor names in a SafeTensors file.

## Commands

```bash
linnet inspect --parameters examples/03-block-and-weights/model.linnet
```

The manifest lists every leaf by path, with its shape after generic
substitution and whether it is optional:

```text
layers.0.up.weight
layers.0.down.weight
...
```

```python
from linnet.torch import load
model = load("examples/03-block-and-weights/model.linnet",
             generics={"H": 8, "Inner": 16, "Layers": 2, "Vocab": 11, "T": "f32"},
             weights="weights/")          # SafeTensors named by the paths above
```

## Key ideas

### Sub-blocks and arrays

`Model` declares `sub layers: [Mlp<H, Inner, T>; Layers]`. Each `Mlp` has
two `Linear` sub-blocks, and each `Linear` has a `weight` and an optional
`bias`. In `Model.forward`, `static for layer in layers` applies the layers
in order, unrolled at compile time.

### Parameter paths

A path follows `sub` names down to a `param`. Weights bind by path, so no
model class has to stay in sync with the file.
