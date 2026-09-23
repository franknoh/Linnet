// Copies the repository's documentation into the site so one source serves
// both: `docs/*.md` become `/docs/*`, `spec/*.md` become `/spec/*`, and every
// example becomes a page showing its sources. Generated directories are
// ignored by git; run before `vitepress dev` or `vitepress build`.
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, "..");
const repo = resolve(site, "..");

function fresh(dir) {
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
}

// ---- docs/ and spec/

const docs = join(site, "docs");
fresh(docs);
for (const file of readdirSync(join(repo, "docs"))) {
  if (file.endsWith(".md")) {
    // The diagnostics catalog moves from a directory README to a page.
    const text = readFileSync(join(repo, "docs", file), "utf8").replaceAll(
      "diagnostics/README.md",
      "diagnostics.md",
    );
    writeFileSync(join(docs, file), text);
  }
}
// The diagnostics catalog is a README in its own directory.
writeFileSync(
  join(docs, "diagnostics.md"),
  readFileSync(join(repo, "docs/diagnostics/README.md"), "utf8"),
);

const spec = join(site, "spec");
fresh(spec);
for (const file of readdirSync(join(repo, "spec"))) {
  if (file.endsWith(".md")) {
    cpSync(join(repo, "spec", file), join(spec, file));
  }
}
// The grammar is a page too, as a code block.
writeFileSync(
  join(spec, "grammar.md"),
  "# Grammar\n\nThe consolidated EBNF grammar (`spec/grammar.ebnf`).\n\n```text\n" +
    readFileSync(join(repo, "spec/grammar.ebnf"), "utf8") +
    "\n```\n",
);

// ---- examples/

const examples = join(site, "examples");
fresh(examples);
const readme = readFileSync(join(repo, "examples/README.md"), "utf8");
const entries = readdirSync(join(repo, "examples"))
  .filter((name) => statSync(join(repo, "examples", name)).isDirectory())
  .sort();

function sources(dir) {
  const out = [];
  for (const name of readdirSync(dir).sort()) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) {
      out.push(...sources(path));
    } else if (name.endsWith(".linnet") || name === "linnet.toml") {
      out.push(path);
    }
  }
  return out;
}

const links = entries
  .map((name) => `- [${name}](/examples/${name})`)
  .join("\n");
writeFileSync(
  join(examples, "index.md"),
  readme.replace(/^# .*$/m, "# Examples") +
    "\n\n## Sources\n\nEach example's complete source, highlighted:\n\n" +
    links +
    "\n",
);

for (const name of entries) {
  const dir = join(repo, "examples", name);
  let page = "";
  const guide = join(dir, "README.md");
  if (existsSync(guide)) {
    // The example's own guide: intro, what it shows, how to run it.
    page += readFileSync(guide, "utf8").trimEnd() + "\n\n## Source\n\n";
  } else {
    page += `# ${name}\n\n`;
    const row = readme.split("\n").find((line) => line.startsWith(`| \`${name}\``));
    if (row) {
      page += row.split("|")[2].trim() + "\n\n";
    }
  }
  for (const file of sources(dir)) {
    const rel = relative(dir, file);
    const lang = file.endsWith(".toml") ? "toml" : "linnet";
    page += `## \`${rel}\`\n\n\`\`\`${lang}\n${readFileSync(file, "utf8")}\`\`\`\n\n`;
  }
  page += `[Browse on GitHub](https://github.com/franknoh/Linnet/tree/main/examples/${name})\n`;
  writeFileSync(join(examples, `${name}.md`), page);
}

console.log(`synced ${readdirSync(docs).length} docs, ${readdirSync(spec).length} spec pages, ${entries.length} examples`);
