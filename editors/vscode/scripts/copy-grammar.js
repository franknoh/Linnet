// The TextMate grammar has one source of truth, editors/textmate. The
// extension ships a copy, made here before packaging.
const fs = require("fs");
const path = require("path");

const source = path.join(__dirname, "..", "..", "textmate", "linnet.tmLanguage.json");
const target = path.join(__dirname, "..", "syntaxes", "linnet.tmLanguage.json");

fs.mkdirSync(path.dirname(target), { recursive: true });
fs.copyFileSync(source, target);
