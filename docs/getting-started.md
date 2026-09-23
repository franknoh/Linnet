# Quickstart

Write a model, check it, and run it in PyTorch. This assumes a built
compiler; see [Installation](https://linnet.franknoh.dev/guide/installation).

## Create a package

```bash
linnet init hello-model
cd hello-model
```

This writes `linnet.toml` and `src/lib.linnet`. Replace `src/lib.linnet` with:

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

A `block` owns parameters and sub-blocks. `entry` marks what a backend can
call. `In`, `Hidden`, `Out`, and `B` are dimensions the caller binds; the
compiler checks every shape in terms of them.

## Check it

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

The last command prints the parameter manifest: the paths and shapes a
checkpoint must provide. `?` marks an optional parameter.

To see the checker at work, change the result type to `Tensor[B, In; T]` and
run `linnet check .` again. It reports the shape that does not match.
Checking never executes anything.

## Run it in PyTorch

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

Weights come from SafeTensors files whose tensor names are the parameter
paths (`up.weight`, `down.weight`, ...): `load(..., weights="weights/")`.
Details in [PyTorch](torch.md).

## Editors

- VS Code: the extension in `editors/vscode` highlights `.linnet` files and
  runs `linnet lsp` for diagnostics, hover, navigation, rename, and
  formatting.
- Vim and Neovim: `editors/vim`, with a language server snippet in its README.

## Next

- [Language tour](language-tour.md): the whole language on one page.
- [Command line](tooling.md): every command.
- `spec/` is the normative specification; `stdlib/` is the standard library,
  written in Linnet.
