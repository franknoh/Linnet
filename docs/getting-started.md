# Getting started

## Build the toolchain

Requirements: CMake 3.25+, Ninja, and a C++23 compiler (GCC 13+, Clang 19+, or
MSVC 2022).

```bash
git clone https://github.com/franknoh/Linnet.git
cd Linnet
cmake --preset release
cmake --build --preset release
ctest --preset release
```

The executable is `build/release/linnet`. It finds the standard library in
`stdlib/` next to the checkout; anywhere else, pass `--std <dir>` or set
`LINNET_STD`.

## A first model

Create a package:

```bash
build/release/linnet init hello-model
cd hello-model
```

Replace `src/lib.linnet` with:

```text
module hello_model

use std.nn.linear::{Linear}
use std.nn.activations::{relu}

pub block Mlp<In: Dim, Hidden: Dim, Out: Dim, T: Float = f32> {
    sub up: Linear<In, Hidden, T>
    sub down: Linear<Hidden, Out, T>

    pub entry forward<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        return down.forward(relu(up.forward(x)))
    }
}
```

Check it, format it, and see what a checkpoint must contain:

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

Try a mistake: change the entry's result type to `Tensor[B, In; T]` and run
`linnet check .` again. The checker reports the shape that does not match and
nothing runs.

## Run it in PyTorch

```bash
cd python/linnet_torch && uv sync
export LINNET_BIN=/path/to/Linnet/build/release/linnet
```

```python
import torch
from linnet_torch import load

model = load("hello-model/src/lib.linnet", generics={"In": 4, "Hidden": 8, "Out": 2})
print(model(torch.randn(3, 4)).shape)   # torch.Size([3, 2])
```

Weights come from SafeTensors files whose tensor names are the parameter paths
above (`up.weight`, ...); see [torch.md](torch.md).

## Editor support

- VS Code: the extension in `editors/vscode` provides highlighting and, through
  `linnet lsp`, diagnostics, hover, navigation, rename, and formatting.
- Vim and Neovim: `editors/vim`, with a language-server snippet in its README.

## Next

- [language-tour.md](language-tour.md) walks through the language.
- [tooling.md](tooling.md) documents every command.
- `spec/` is the normative specification; `stdlib/` shows the operations in
  ordinary Linnet.
