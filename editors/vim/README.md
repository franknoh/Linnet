# Linnet for Vim and Neovim

Filetype detection, syntax highlighting, indentation, and comment settings for
`.linnet` files.

## Installing

Copy or link this directory into a package path, for example:

```bash
mkdir -p ~/.vim/pack/linnet/start
ln -s "$PWD/editors/vim" ~/.vim/pack/linnet/start/linnet
```

For Neovim use `~/.local/share/nvim/site/pack/linnet/start` instead.

When the `linnet` executable is on `PATH`, `gq` formats through
`linnet fmt -`; `:%!linnet fmt -` formats the whole buffer.

## Language server

With Neovim's built-in client:

```lua
vim.api.nvim_create_autocmd("FileType", {
    pattern = "linnet",
    callback = function(args)
        vim.lsp.start({
            name = "linnet",
            cmd = { "linnet", "lsp", "--stdio" },
            root_dir = vim.fs.root(args.buf, { "linnet.toml" }),
        })
    end,
})
```

Pass `"--std", "/path/to/stdlib"` after `"--stdio"` when the standard library
is not next to the executable. Any LSP client for Vim works the same way; the
server speaks JSON-RPC over stdio and needs no other configuration.
