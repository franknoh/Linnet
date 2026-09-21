# Linnet for Visual Studio Code

Syntax highlighting, comment toggling, bracket matching, and indentation for
`.linnet` files.

## Trying it from a checkout

```bash
cd editors/vscode
npm run prepare-grammar
code --extensionDevelopmentPath="$PWD"
```

The grammar lives in `editors/textmate`; `prepare-grammar` copies it into the
extension. Type and shape information will come from the language server and
is never computed in the extension itself.
