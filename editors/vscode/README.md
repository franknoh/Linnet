# Linnet for Visual Studio Code

Syntax highlighting, and every language feature of the Linnet language server:
diagnostics, hover, go to definition, references, rename, document and
workspace symbols, completion, formatting, semantic tokens, and inlay hints.

The extension only launches `linnet lsp --stdio`. Type and shape information
comes from the compiler, never from the extension.

## Settings

- `linnet.path` — the `linnet` executable (default: `linnet` on `PATH`).
- `linnet.stdRoot` — the standard library directory, passed as `--std`.

`Linnet: Restart Language Server` restarts the server after rebuilding it.

## Trying it from a checkout

```bash
cd editors/vscode
npm install
npm run check          # grammar copy, compile, lint, format check
code --extensionDevelopmentPath="$PWD"
```

The grammar lives in `editors/textmate`; `npm run prepare-grammar` copies it in.
