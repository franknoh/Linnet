import DefaultTheme from "vitepress/theme";
import type { Theme } from "vitepress";
import BenchTable from "./BenchTable.vue";
import "./custom.css";

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component("BenchTable", BenchTable);
  },
} satisfies Theme;
