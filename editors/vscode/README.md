# Linnet for Visual Studio Code

Highlighting for `.linnet` files, and the features of the Linnet language
server ([`linnet lsp`](../../docs/tooling.md#lsp)).

## Run from a checkout

```bash
cd editors/vscode
npm install
npm run check          # grammar copy, compile, lint, format check
code --extensionDevelopmentPath="$PWD"
```

To package and install it, see
[Installation](https://linnet.franknoh.dev/guide/installation#vs-code).

## Settings

- `linnet.path`: the `linnet` executable (default: `linnet` on `PATH`).
- `linnet.stdRoot`: the standard library directory, passed as `--std`.

Run `Linnet: Restart Language Server` after rebuilding the compiler.
