# Linnet for Visual Studio Code

Language support for [Linnet](https://linnet.franknoh.dev), the checked
source format for neural network architectures: highlighting for `.linnet`
files, and the Linnet language server for everything else.

- Shape, dtype and type errors as you type, with the fix in the message
- Hover types, go to definition, references, rename and symbols
- Completion, inlay hints and semantic highlighting
- Formatting in Linnet's one style

The extension for Linux (x64, arm64), macOS (Apple silicon) and Windows
(x64) carries the `linnet` compiler and its standard library, so nothing
else needs installing. On other platforms, install the compiler with
`pip install linnet-lang` or
[build it](https://linnet.franknoh.dev/guide/installation#from-source).

## Settings

- `linnet.path`: the `linnet` executable. Empty (the default) uses the
  bundled one, else `linnet` on `PATH`. Set it to use another build, such
  as the one in a project's `.venv/bin`.
- `linnet.stdRoot`: a standard library directory, passed as `--std`.

Run `Linnet: Restart Language Server` after changing the compiler.

## From a checkout

```bash
cd editors/vscode
npm install
npm run check          # grammar copy, compile, lint, format check
code --extensionDevelopmentPath="$PWD"
```
