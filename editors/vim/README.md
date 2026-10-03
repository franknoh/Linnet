# Linnet for Vim and Neovim

Filetype detection, highlighting, indentation and comment settings for
`.linnet` files.

## Install

Link this directory into a package path. For Neovim, use
`~/.local/share/nvim/site/pack/linnet/start` instead.

```bash
mkdir -p ~/.vim/pack/linnet/start
ln -s "$PWD/editors/vim" ~/.vim/pack/linnet/start/linnet
```

With `linnet` on `PATH`, `gq` formats through `linnet fmt -`, and
`:%!linnet fmt -` formats the whole buffer.

## Language server

In Neovim, start the built-in client:

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

If the standard library is not next to the executable, add
`"--std", "/path/to/stdlib"` after `"--stdio"`. Other Vim LSP clients take
the same command.
