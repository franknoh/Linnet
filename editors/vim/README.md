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
