# Installation

Linnet is one executable, `linnet`, plus a standard library directory and
optional Python adapters for PyTorch, JAX, and ONNX.

## The compiler

### From a release

Tagged releases publish archives for Linux x86-64, macOS arm64, and Windows
x86-64 on the [releases page](https://github.com/franknoh/Linnet/releases).
Each contains `bin/linnet` and `share/linnet/stdlib`; unpack it and put `bin`
on your `PATH`:

```bash
tar -xzf linnet-<version>-linux-x86_64.tar.gz
export PATH="$PWD/linnet-<version>-linux-x86_64/bin:$PATH"
linnet --version
```

The executable looks for the standard library next to itself; anywhere else,
pass `--std <dir>` or set `LINNET_STD`.

### From source

Requirements: CMake 3.25+, Ninja, and a C++23 compiler (GCC 13+, Clang 19+,
or MSVC 2022). No other dependencies.

```bash
git clone https://github.com/franknoh/Linnet.git
cd Linnet
cmake --preset release
cmake --build --preset release
ctest --preset release          # optional: the unit and CLI tests
```

The executable is `build/release/linnet` and finds `stdlib/` in the checkout.

## Editor support

### Visual Studio Code

The extension provides highlighting and, through `linnet lsp`, diagnostics as
you type, hover with inferred types and shapes, go to definition, references,
rename, symbols, completion, formatting, semantic tokens, and inlay hints.

- From a release: download `linnet-<version>.vsix` and run
  `code --install-extension linnet-<version>.vsix`.
- From a checkout:

  ```bash
  cd editors/vscode
  npm install
  npm run check
  npx @vscode/vsce package        # writes linnet-<version>.vsix
  code --install-extension linnet-*.vsix
  ```

Settings: `linnet.path` (the executable; default `linnet` on `PATH`) and
`linnet.stdRoot` (passed as `--std`). `Linnet: Restart Language Server`
restarts the server after you rebuild the compiler.

### Neovim and Vim

`editors/vim` has filetype detection, syntax highlighting, indentation, and
comment settings:

```bash
mkdir -p ~/.local/share/nvim/site/pack/linnet/start
ln -s "$PWD/editors/vim" ~/.local/share/nvim/site/pack/linnet/start/linnet
```

With the built-in LSP client:

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

Any LSP client works: the server is `linnet lsp --stdio`. The TextMate grammar
in `editors/textmate/linnet.tmLanguage.json` gives highlighting to editors
that read TextMate grammars (Zed, Sublime Text, this site).

## Python adapters

Each adapter is a [uv](https://docs.astral.sh/uv/) project under `python/`;
they talk to the compiler as a subprocess and read its JSON, so none of them
links against C++.

```bash
export LINNET_BIN=/path/to/linnet      # or put `linnet` on PATH

cd python/linnet_torch && uv sync      # PyTorch: load, bind_weights, export_linnet
cd python/linnet_jax   && uv sync      # JAX: load, load_nnx, export_linnet, import_stablehlo
cd python/linnet_onnx  && uv sync      # ONNX: import_onnx
```

Then continue with [Getting started](/docs/getting-started), or with
[Coming from PyTorch](/guide/from-pytorch) if you have models already.
