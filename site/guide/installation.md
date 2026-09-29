# Installation

Linnet is one executable, `linnet`, a standard library directory, and
optional Python adapters for PyTorch, JAX, and ONNX.

## Compiler

### Release archives

Tagging a version builds archives for Linux x86-64, macOS arm64, and Windows
x86-64 (`bin/linnet` and `share/linnet/stdlib`) on the
[releases page](https://github.com/franknoh/Linnet/releases). No version has
been tagged yet; until one is, build from source.

### From source

Requirements: CMake 3.25 or newer, Ninja, and a C++23 compiler (GCC 13,
Clang 19, or MSVC 2022). There are no other dependencies.

```bash
git clone https://github.com/franknoh/Linnet.git
cd Linnet
cmake --preset release
cmake --build --preset release
ctest --preset release          # optional
```

The executable is `build/release/linnet`; it finds `stdlib/` in the
checkout. Elsewhere, pass `--std <dir>` or set `LINNET_STD`.

## Editors

### VS Code

The extension highlights `.linnet` files and runs `linnet lsp` for
diagnostics as you type, hover with types and shapes, go to definition,
references, rename, symbols, completion, formatting, semantic tokens, and
inlay hints.

```bash
cd editors/vscode && npm install && npm run check      # from a checkout
npx @vscode/vsce package
code --install-extension linnet-*.vsix
```

Settings: `linnet.path` (the executable, default `linnet` on `PATH`) and
`linnet.stdRoot` (passed as `--std`). Run `Linnet: Restart Language Server`
after rebuilding the compiler.

### Neovim and Vim

`editors/vim` provides filetype detection, highlighting, indentation, and
comments:

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

Any LSP client can run `linnet lsp --stdio`. The TextMate grammar in
`editors/textmate/linnet.tmLanguage.json` gives highlighting to editors that
read TextMate grammars; this site uses it too.

## Python adapters

The adapters are one package, `linnet-lang` (imported as `linnet`), a
[uv](https://docs.astral.sh/uv/) project under `python/linnet`. It calls the
compiler as a subprocess and reads its JSON; nothing links against C++. It is
not on PyPI yet: install it from a checkout, or with pip from the repository
(`pip install "linnet-lang[torch] @ git+https://github.com/franknoh/Linnet#subdirectory=python/linnet"`).

```bash
export LINNET_BIN=/path/to/linnet      # or put `linnet` on PATH

cd python/linnet && uv sync --extra torch    # linnet.torch: load, bind_weights, export_linnet
cd python/linnet && uv sync --extra jax      # linnet.jax: load, load_source, export_linnet, import_stablehlo
cd python/linnet && uv sync --extra flax     # linnet.jax.load_nnx
cd python/linnet && uv sync --extra onnx     # linnet.onnx: export_model, load_model, import_onnx
```

`linnet.onnx.load_model` runs on ONNX Runtime, which the `onnx` extra does not
install: add `onnxruntime`, or `onnxruntime-gpu` for CUDA and TensorRT.

Next: the [Quickstart](/docs/getting-started), or [Coming from
PyTorch](/guide/from-pytorch) if you already have models.
