// Copies the repository's documentation into the site so one source serves
// both: `docs/*.md` become `/docs/*`, `spec/*.md` become `/spec/*`, every
// example becomes a page showing its sources, and the Nest registry (a
// checkout next to this repository, or `NEST_DIR`) becomes `/nest/`.
// Generated directories are ignored by git; run before `vitepress dev` or
// `vitepress build`.
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

// ---- nest/ (the model zoo registry)

const nest = join(site, "nest");
fresh(nest);
const nestPublic = join(site, "public/nest");
fresh(nestPublic);
const nestDir = process.env.NEST_DIR ?? resolve(repo, "..", "nest");
const indexPath = join(nestDir, "index.json");
let nestCount = 0;

function parameters(count) {
  if (count == null) return "?";
  for (const [unit, size] of [["B", 1e9], ["M", 1e6], ["K", 1e3]]) {
    if (count >= size) return `${(count / size).toFixed(1).replace(/\.0$/, "")}${unit}`;
  }
  return String(count);
}

function linkList(links) {
  const names = { huggingface: "Hugging Face", github: "GitHub", arxiv: "arXiv", homepage: "Homepage" };
  return Object.entries(links)
    .map(([key, url]) => `[${names[key] ?? key}](${url})`)
    .join(" · ");
}

if (existsSync(indexPath)) {
  const registry = JSON.parse(readFileSync(indexPath, "utf8"));
  const models = registry.models;
  nestCount = models.length;
  let page =
    "# Nest\n\nThe Linnet model zoo. Each model is a checked `.linnet` source with a SafeTensors " +
    "checkpoint on the Hugging Face Hub, loadable in PyTorch, JAX, XLA, and ONNX Runtime from the " +
    "same file. [How Nest works](/docs/nest); [the registry on GitHub](https://github.com/franknoh/nest).\n\n" +
    "```python\nfrom linnet import nest\n\nmodel = nest.load(\"" +
    (models[0]?.name ?? "gpt2") +
    "\", backend=\"torch\")\n```\n\n" +
    "| Model | Family | Parameters | License | Links |\n| --- | --- | --- | --- | --- |\n";
  for (const model of models) {
    page += `| [${model.title}](/nest/${model.name}) | ${model.family ?? ""} | ${parameters(model.parameters)} | ${model.license} | ${linkList(model.links)} |\n`;
  }
  writeFileSync(join(nest, "index.md"), page);

  for (const model of models) {
    const dir = join(nestDir, "models", model.name);
    const previewDir = join(nestPublic, model.name);
    mkdirSync(previewDir, { recursive: true });
    for (const file of ["preview.svg", "preview-dark.svg"]) {
      if (existsSync(join(dir, file))) cpSync(join(dir, file), join(previewDir, file));
    }
    let body = `# ${model.title}\n\n`;
    body += `${model.summary}\n\n`;
    body += `| | |\n| --- | --- |\n`;
    body += `| Family | ${model.family ?? ""} |\n| Parameters | ${parameters(model.parameters)} |\n| License | ${model.license} |\n`;
    body += `| Links | ${linkList(model.links)} |\n`;
    if (model.weights) {
      body += `| Weights | [${model.weights.repo}](https://huggingface.co/${model.weights.repo}) (${model.weights.files.join(", ")}) |\n`;
    }
    body += `| Load | \`nest.load("${model.name}")\` |\n\n`;
    if (existsSync(join(dir, "preview.svg"))) {
      body += `## Architecture\n\n`;
      body += `The \`${model.entry}\` entry, one level of blocks expanded; edges carry the tensor types the compiler inferred.\n\n`;
      body += `<img class="nest-preview nest-preview-light" src="/nest/${model.name}/preview.svg" alt="${model.title} architecture">\n`;
      if (existsSync(join(dir, "preview-dark.svg"))) {
        body += `<img class="nest-preview nest-preview-dark" src="/nest/${model.name}/preview-dark.svg" alt="${model.title} architecture">\n`;
      }
      body += "\n";
    }
    body += `## Entries\n\n| Entry | Signature |\n| --- | --- |\n`;
    for (const entry of model.entries) {
      body += `| \`${entry.name}\` | \`${entry.signature.replace(/^pub entry /, "")}\` |\n`;
    }
    body += `\n## Generics\n\n| | |\n| --- | --- |\n`;
    for (const [name, value] of Object.entries(model.generics)) {
      body += `| \`${name}\` | \`${value}\` |\n`;
    }
    const readme = join(dir, "README.md");
    if (existsSync(readme)) {
      // The card's own README, its title dropped since the page has one.
      body += "\n## Card\n\n" + readFileSync(readme, "utf8").replace(/^# .*\n+/, "").replace(/^## /gm, "### ").trimEnd() + "\n";
    }
    body += "\n## Source\n\n";
    for (const file of model.files) {
      if (file.endsWith(".linnet") || file === "linnet.toml" || file === "nest.toml" || file === "bindings.json") {
        const lang = file.endsWith(".toml") ? "toml" : file.endsWith(".json") ? "json" : "linnet";
        body += `### \`${file}\`\n\n\`\`\`${lang}\n${readFileSync(join(dir, file), "utf8")}\`\`\`\n\n`;
      }
    }
    body += `[Browse on GitHub](https://github.com/franknoh/nest/tree/main/models/${model.name})\n`;
    writeFileSync(join(nest, `${model.name}.md`), body);
  }
} else {
  writeFileSync(
    join(nest, "index.md"),
    "# Nest\n\nThe model zoo registry was not available when this site was built. " +
      "See [github.com/franknoh/nest](https://github.com/franknoh/nest) and [how Nest works](/docs/nest).\n",
  );
}

console.log(
  `synced ${readdirSync(docs).length} docs, ${readdirSync(spec).length} spec pages, ${entries.length} examples, ${nestCount} nest models`,
);
