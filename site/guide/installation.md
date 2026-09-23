# Installation

Linnet is one executable, `linnet`, a standard library directory, and
optional Python adapters for PyTorch, JAX, and ONNX.

## Compiler

### Release archive

The [releases page](https://github.com/franknoh/Linnet/releases) has
archives for Linux x86-64, macOS arm64, and Windows x86-64. Each contains
`bin/linnet` and `share/linnet/stdlib`; the executable finds the standard
library next to itself.

```bash
tar -xzf linnet-<version>-linux-x86_64.tar.gz
export PATH="$PWD/linnet-<version>-linux-x86_64/bin:$PATH"
linnet --version
```

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
code --install-extension linnet-<version>.vsix        # from a release
```

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

Each adapter is a [uv](https://docs.astral.sh/uv/) project under `python/`.
They call the compiler as a subprocess and read its JSON; none links against
C++.

```bash
export LINNET_BIN=/path/to/linnet      # or put `linnet` on PATH

cd python/linnet_torch && uv sync      # load, bind_weights, export_linnet
cd python/linnet_jax   && uv sync      # load, load_source, load_nnx, export_linnet, import_stablehlo
cd python/linnet_onnx  && uv sync      # import_onnx
```

Next: the [Quickstart](/docs/getting-started), or [Coming from
PyTorch](/guide/from-pytorch) if you already have models.
