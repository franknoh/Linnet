# Installation

Build the `linnet` compiler, then add editor support and the Python adapters
as needed.

## Compiler

No version is on the [releases page](https://github.com/franknoh/Linnet/releases)
yet, so build from source. You need CMake 3.25 or newer, Ninja, and a C++23
compiler (GCC 13, Clang 19 or MSVC 2022).

```bash
git clone https://github.com/franknoh/Linnet.git
cd Linnet
cmake --preset release
cmake --build --preset release
ctest --preset release          # optional
```

The executable is `build/release/linnet` and finds `stdlib/` in the
checkout. Elsewhere, pass `--std <dir>` or set `LINNET_STD`.

## Editors

The editors run the language server, [`linnet lsp`](/docs/tooling#lsp).

### VS Code

```bash
cd editors/vscode && npm install && npm run check      # from a checkout
npx @vscode/vsce package
code --install-extension linnet-*.vsix
```

Set `linnet.path` if `linnet` is not on `PATH`, and `linnet.stdRoot` to pass
`--std`. Run `Linnet: Restart Language Server` after rebuilding the compiler.

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

## Python adapters

The adapters are one package, `linnet-lang`, imported as `linnet`. It is not
on PyPI yet; install it from a checkout with [uv](https://docs.astral.sh/uv/):

```bash
export LINNET_BIN=/path/to/linnet      # or put `linnet` on PATH

cd python/linnet && uv sync --extra torch    # linnet.torch: load, bind_weights, export_linnet
cd python/linnet && uv sync --extra jax      # linnet.jax: load, load_source, export_linnet, import_stablehlo
cd python/linnet && uv sync --extra flax     # linnet.jax.load_nnx
cd python/linnet && uv sync --extra onnx     # linnet.onnx: export_model, load_model, import_onnx
```

Or with pip:

```bash
pip install "linnet-lang[torch] @ git+https://github.com/franknoh/Linnet#subdirectory=python/linnet"
```

`linnet.onnx.load_model` also needs `onnxruntime`, or `onnxruntime-gpu` for
CUDA and TensorRT.

Next: the [Quickstart](/docs/getting-started), or
[Coming from PyTorch](/guide/from-pytorch) if you already have models.
