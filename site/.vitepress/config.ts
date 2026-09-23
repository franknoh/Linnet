import { readFileSync, readdirSync, existsSync } from "node:fs";
import { dirname, resolve, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitepress";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../..");

// The editors' TextMate grammar is the one source of highlighting: the site
// registers it with Shiki, so pages highlight exactly like VS Code does.
const grammar = JSON.parse(
  readFileSync(resolve(repo, "editors/textmate/linnet.tmLanguage.json"), "utf8"),
);
grammar.name = "linnet";

// Pages the sync script copies from the repository, for the sidebars.
function pages(dir: string, prefix: string): { text: string; link: string }[] {
  const root = resolve(here, "..", dir);
  if (!existsSync(root)) {
    return [];
  }
  return readdirSync(root)
    .filter((file) => file.endsWith(".md") && file !== "index.md")
    .sort()
    .map((file) => {
      const text = readFileSync(resolve(root, file), "utf8");
      const heading = text.match(/^#\s+(.+)$/m);
      const name = basename(file, ".md");
      return { text: heading ? heading[1].replace(/`/g, "") : name, link: `${prefix}/${name}` };
    });
}

export default defineConfig({
  title: "Linnet",
  description:
    "A typed tensor language: models as checked, weight-free source that runs in PyTorch, JAX, XLA, and ONNX Runtime.",
  lang: "en-US",
  cleanUrls: true,
  lastUpdated: true,
  sitemap: { hostname: "https://linnet.franknoh.dev" },
  head: [
    ["link", { rel: "icon", type: "image/svg+xml", href: "/logo.svg" }],
    ["meta", { name: "theme-color", content: "#3c6e71" }],
  ],
  markdown: {
    languages: [grammar],
    theme: { light: "github-light", dark: "github-dark" },
    config(md) {
      // The repository's Markdown marks Linnet snippets as `text`; those
      // that read like Linnet are highlighted as Linnet here.
      const fence = md.renderer.rules.fence!;
      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx];
        const looksLikeLinnet =
          /^(module |pub (block|fn|op|entry|const|struct|enum)|block |fn |op |let |state |param |sub )|Tensor\[/m.test(
            token.content,
          );
        if (token.info.trim() === "text" && looksLikeLinnet) {
          token.info = "linnet";
        }
        return fence(tokens, idx, options, env, self);
      };
    },
  },
  themeConfig: {
    logo: "/logo.svg",
    nav: [
      { text: "Guide", link: "/guide/installation", activeMatch: "/guide/" },
      { text: "Docs", link: "/docs/getting-started", activeMatch: "/docs/" },
      { text: "Examples", link: "/examples/", activeMatch: "/examples/" },
      { text: "Benchmarks", link: "/benchmarks" },
      { text: "Spec", link: "/spec/00-overview", activeMatch: "/spec/" },
    ],
    sidebar: {
      "/guide/": [
        {
          text: "Guide",
          items: [
            { text: "Installation", link: "/guide/installation" },
            { text: "Coming from PyTorch", link: "/guide/from-pytorch" },
            { text: "Getting started", link: "/docs/getting-started" },
            { text: "Language tour", link: "/docs/language-tour" },
            { text: "Compatibility", link: "/compatibility" },
            { text: "Benchmarks", link: "/benchmarks" },
          ],
        },
      ],
      "/docs/": [
        {
          text: "Documentation",
          items: [
            { text: "Getting started", link: "/docs/getting-started" },
            { text: "Language tour", link: "/docs/language-tour" },
            { text: "Modules and packages", link: "/docs/modules-and-packages" },
            { text: "Command-line tooling", link: "/docs/tooling" },
            { text: "Randomness", link: "/docs/random" },
            { text: "Diagnostics", link: "/docs/diagnostics" },
          ],
        },
        {
          text: "Backends",
          items: [
            { text: "PyTorch", link: "/docs/torch" },
            { text: "JAX and Flax", link: "/docs/jax" },
            { text: "ONNX", link: "/docs/onnx" },
            { text: "Plan format", link: "/docs/plan-format" },
          ],
        },
        { text: "Guide", items: [{ text: "Compatibility", link: "/compatibility" }] },
      ],
      "/examples/": [
        { text: "Examples", items: [{ text: "Overview", link: "/examples/" }, ...pages("examples", "/examples")] },
      ],
      "/spec/": [{ text: "Specification", items: pages("spec", "/spec") }],
    },
    socialLinks: [{ icon: "github", link: "https://github.com/franknoh/Linnet" }],
    editLink: {
      pattern: ({ filePath }) => {
        // Copied pages point back at their source in the repository.
        if (filePath.startsWith("docs/")) {
          return `https://github.com/franknoh/Linnet/edit/main/${filePath}`;
        }
        if (filePath.startsWith("spec/")) {
          return `https://github.com/franknoh/Linnet/edit/main/${filePath}`;
        }
        return `https://github.com/franknoh/Linnet/edit/main/site/${filePath}`;
      },
      text: "Edit this page on GitHub",
    },
    search: { provider: "local" },
    footer: {
      message: "Released under the MIT License.",
      copyright: "Copyright © Frank Noh",
    },
  },
});
