import { h } from "vue";
import DefaultTheme from "vitepress/theme";
import type { Theme } from "vitepress";
import BenchChart from "./BenchChart.vue";
import BenchTable from "./BenchTable.vue";
import HeroFiles from "./HeroFiles.vue";
import ZooBench from "./ZooBench.vue";
import "./custom.css";

export default {
  extends: DefaultTheme,
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      "home-hero-info-after": () => h(HeroFiles),
    }),
  enhanceApp({ app }) {
    app.component("BenchChart", BenchChart);
    app.component("BenchTable", BenchTable);
    app.component("ZooBench", ZooBench);
  },
} satisfies Theme;
