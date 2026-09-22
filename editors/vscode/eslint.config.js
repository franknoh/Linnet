const js = require("@eslint/js");
const tseslint = require("typescript-eslint");

module.exports = tseslint.config(
    js.configs.recommended,
    ...tseslint.configs.strictTypeChecked,
    {
        languageOptions: {
            parserOptions: { project: "./tsconfig.json", tsconfigRootDir: __dirname },
        },
    },
    { ignores: ["out/", "eslint.config.js", "scripts/"] },
);
