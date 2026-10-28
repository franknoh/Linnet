# linnet-cli

The `linnet` command: the [Linnet](https://linnet.franknoh.dev) compiler
(`check`, `fmt`, `inspect`, ...), its language server, and `linnet serve`,
for installing as a uv tool:

```bash
curl -LsSf https://linnet.franknoh.dev/install.sh | sh
# or
uv tool install "linnet-cli @ git+https://github.com/franknoh/Linnet#subdirectory=python/cli"
uv tool update-shell
```

To import `linnet` from Python, add `linnet-lang` to the project instead.
