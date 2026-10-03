# Quickstart

Write a small model, check it, and run it in PyTorch. You need a built
compiler: see [Installation](https://linnet.franknoh.dev/guide/installation).

## 1. Create a package

```bash
linnet init hello-model
cd hello-model
```

Replace the generated `src/lib.linnet` with:

```linnet
module hello_model

use std.nn.activations::{relu}
use std.nn.linear::{Linear}

pub block Mlp<In: Dim, Hidden: Dim, Out: Dim, T: Float = f32> {
    sub up: Linear<In, Hidden, T>
    sub down: Linear<Hidden, Out, T>

    pub entry forward<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        return down.forward(relu(up.forward(x)))
    }
}
```

A `block` owns parameters and sub-blocks, and an `entry` is what a backend
calls. The compiler checks every shape in terms of `In`, `Hidden`, `Out` and
`B`.

## 2. Check it

```bash
linnet check .
linnet fmt --check .
linnet inspect --parameters src/lib.linnet
```

```text
hello_model::Mlp<In, Hidden, Out, T>
  param up.weight: Tensor[Hidden, In; T]
  param up.bias: Tensor[Hidden; T]?
  param down.weight: Tensor[Out, Hidden; T]
  param down.bias: Tensor[Out; T]?
```

These are the tensors a checkpoint must provide; `?` marks an optional one.
Change the result type to `Tensor[B, In; T]` and `linnet check .` reports
the mismatch.

## 3. Run it in PyTorch

```bash
cd python/linnet && uv sync --extra torch
export LINNET_BIN=/path/to/Linnet/build/release/linnet
```

```python
import torch
from linnet.torch import load

model = load("hello-model/src/lib.linnet", generics={"In": 4, "Hidden": 8, "Out": 2})
print(model(torch.randn(3, 4)).shape)   # torch.Size([3, 2])
```

The model now runs as a PyTorch module. Pass `weights="weights/"` to load
SafeTensors named by parameter path; see [PyTorch](torch.md).

## Next

- [Language tour](language-tour.md)
- [Command line](tooling.md)
- [Editor setup](https://linnet.franknoh.dev/guide/installation#editors)
