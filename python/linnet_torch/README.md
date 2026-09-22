# linnet-torch

Materializes a Linnet model as a `torch.nn.Module`.

```python
from linnet_torch import load

model = load(
    "src/model.linnet",
    root="Model",
    generics={"H": 64, "Inner": 256, "Layers": 2, "Vocab": 100},
    weights="weights/",
)
logits = model(tokens)
```

`load` runs `linnet plan` to obtain the compiled plan of the root block, builds
a module hierarchy whose parameter names follow the Linnet parameter paths
(`layers.0.up.weight`), binds SafeTensors weights after checking every name,
shape, and dtype against the plan, and evaluates the model's entries by
interpreting Core IR with PyTorch operations. Semantic operations run through
their canonical decompositions; there are no model-specific classes here.
