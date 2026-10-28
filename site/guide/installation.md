# Installation

One package, `linnet-lang`, holds the `linnet` compiler, its standard
library and the Python package `linnet`. It is not on PyPI yet: install it
from the repository with [uv](https://docs.astral.sh/uv/), which builds the
compiler as it builds the package.

The `linnet` command (`check`, `fmt`, `lsp`, and the editors that run it)
goes on `PATH` as a uv tool:

```bash
uv tool install "linnet-lang @ git+https://github.com/franknoh/Linnet#subdirectory=python/linnet"
uv tool update-shell      # once, if uv says its bin directory is not on PATH
linnet --version
```

For `linnet serve`, install `"linnet-lang[serve] @ ..."` instead.

A project that imports `linnet` adds the package with the extras it uses;
the project's environment gets its own compiler:

```bash
uv add "linnet-lang[torch] @ git+https://github.com/franknoh/Linnet#subdirectory=python/linnet"
```

The build needs a C++23 compiler (GCC 13, Clang 19 or MSVC 2022); CMake and
Ninja come from PyPI. It takes a minute or two, once: uv keeps the built
wheel for that commit. Put `@<commit>` after `Linnet` to pin one.

Pick the extras for the frameworks you use:

| Extra | Adds |
| --- | --- |
| `torch` | `linnet.torch`: load, train and export in PyTorch, and `linnet.serve` |
| `jax` | `linnet.jax`: load, train and export in JAX |
| `flax` | `linnet.jax.load_nnx` |
| `onnx` | `linnet.onnx`; add `onnxruntime`, or `onnxruntime-gpu` for CUDA and TensorRT, to run models |
| `nest` | `linnet.nest`: models from [Nest](https://nest.franknoh.dev) and the Hugging Face Hub |
| `serve` | `linnet serve`: PyTorch, the Hub and a tokenizer, to serve a model with OpenAI's API |
| `all` | all of them |

Outside a uv project, `uv pip install "linnet-lang[torch] @ git+https://github.com/franknoh/Linnet#subdirectory=python/linnet"`
installs into the active environment, and `uv run --with "linnet-lang[torch] @ git+https://github.com/franknoh/Linnet#subdirectory=python/linnet"
python script.py` runs one script.

Without a C++23 compiler (Ubuntu 22.04 and Colab ship GCC 11), install a
wheel from the [releases page](https://github.com/franknoh/Linnet/releases)
instead: the wheels for Linux (x86_64, aarch64), macOS (arm64, 14 or newer)
and Windows (x64) carry a built compiler.

```bash
uv add "linnet-lang[torch] @ https://github.com/franknoh/Linnet/releases/download/<tag>/<wheel>"
```

With `LINNET_BUILD_COMPILER=0` the package installs without the compiler and
runs the one `LINNET_BIN` names, or the one on `PATH`; the releases page has
the compiler alone too.

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

The extension is not on the Visual Studio Marketplace or Open VSX yet.
Download the `.vsix` for your platform from the
[releases page](https://github.com/franknoh/Linnet/releases) and install it
with `code --install-extension linnet-<version>-<platform>.vsix` (in
VSCodium or Cursor, **Extensions: Install from VSIX...**). The builds for
Linux, macOS (Apple silicon) and Windows carry the compiler and need nothing
else; the one without a platform uses `linnet` on `PATH`.

`linnet.path` picks another compiler (in a uv project,
`.venv/bin/linnet`), and `linnet.stdRoot` passes `--std`. Run
`Linnet: Restart Language Server` after changing the compiler. From a
checkout:

```bash
cd editors/vscode && npm install && npm run check
npx @vscode/vsce package
code --install-extension linnet-*.vsix
```

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
