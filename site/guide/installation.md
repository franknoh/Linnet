# Installation

One package, `linnet-lang`, holds the `linnet` compiler, its standard
library and the Python package `linnet`. Install it with
[uv](https://docs.astral.sh/uv/):

```bash
uv add "linnet-lang[torch]"
uv run linnet --version
```

Pick the extras for the frameworks you use:

| Extra | Adds |
| --- | --- |
| `torch` | `linnet.torch`: load, train and export in PyTorch, and `linnet.serve` |
| `jax` | `linnet.jax`: load, train and export in JAX |
| `flax` | `linnet.jax.load_nnx` |
| `onnx` | `linnet.onnx`; add `onnxruntime`, or `onnxruntime-gpu` for CUDA and TensorRT, to run models |
| `nest` | `linnet.nest`: models from [Nest](https://nest.franknoh.dev) and the Hugging Face Hub |
| `all` | all of them |

Outside a uv project, `uv pip install "linnet-lang[torch]"` installs into
the active environment, and `uv run --with "linnet-lang[torch]" python
script.py` runs one script. `pip install` works the same way.

The wheels for Linux (x86_64, aarch64), macOS (arm64, 14 or newer) and
Windows (x64) carry the compiler. On another platform, build the compiler
from source and point `LINNET_BIN` at it.

## From source

You need CMake 3.25 or newer, Ninja, and a C++23 compiler (GCC 13, Clang 19
or MSVC 2022).

```bash
git clone https://github.com/franknoh/Linnet.git
cd Linnet
cmake --preset release
cmake --build --preset release
ctest --preset release          # optional
```

The executable is `build/release/linnet` and finds `stdlib/` in the
checkout. Elsewhere, pass `--std <dir>` or set `LINNET_STD`. To use it from
the Python package, set `LINNET_BIN=build/release/linnet`. In
`python/linnet`, `uv sync --all-extras` installs the package with every
backend and test dependency.

## Editors

The editors run the language server, [`linnet lsp`](/docs/tooling#lsp).

### VS Code

```bash
cd editors/vscode && npm install && npm run check      # from a checkout
npx @vscode/vsce package
code --install-extension linnet-*.vsix
```

Set `linnet.path` if `linnet` is not on `PATH` (in a uv project,
`.venv/bin/linnet`), and `linnet.stdRoot` to pass `--std`. Run `Linnet: Restart Language Server` after rebuilding the compiler.

### Neovim and Vim

```bash
mkdir -p ~/.local/share/nvim/site/pack/linnet/start
ln -s "$PWD/editors/vim" ~/.local/share/nvim/site/pack/linnet/start/linnet
```

```lua
vim.api.nvim_create_autocmd("FileType", {
    pattern = "linnet",
    callback = function(args)
        vim.lsp.start({
            name = "linnet",
            cmd = { "linnet", "lsp", "--stdio" },
            root_dir = vim.fs.root(args.buf, { "linnet.toml", ".git" }),
        })
    end,
})
```

`gq` formats through `linnet fmt -`.

### Other editors

Run `linnet lsp --stdio` from any Language Server Protocol client. For
highlighting, use the TextMate grammar
`editors/textmate/linnet.tmLanguage.json`.

Next: the [Quickstart](/docs/getting-started), or
[Coming from PyTorch](/guide/from-pytorch) if you already have models.
