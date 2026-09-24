<script setup lang="ts">
// Latency bar charts from bench/results/latest.json: one chart per run
// (configuration × entry), one bar per variant, drawn as plain SVG so the
// page needs no chart library and follows the site's colours.
import { computed, ref } from "vue";
import results from "../../../bench/results/latest.json";

interface Variant {
  name: string;
  latency_ms: number | null;
  throughput: number | null;
  max_abs_diff: number | null;
  note?: string;
}

interface Run {
  model: string;
  config: string;
  entry: string;
  unit: string;
  variants: Variant[];
}

const data = results as { runs: Run[] };
// Bars show speed-up against the first variant (eager PyTorch); the toggle
// switches to raw latency.
const latency = ref(false);

type Family = "reference" | "torch" | "xla";

function family(name: string): Family {
  if (name.startsWith("PyTorch reference")) return "reference";
  if (name.includes("XLA")) return "xla";
  return "torch";
}

// Short labels for the axis; the table below keeps the full names.
function label(name: string): string {
  const fast = name.includes("numerics=fast");
  if (name.startsWith("PyTorch reference (eager")) return "PyTorch eager";
  if (name.startsWith("PyTorch reference (torch.compile")) return "PyTorch torch.compile";
  if (name.includes("numerics=equivalent")) return "Linnet interpreted";
  if (name.includes("CUDA graphs")) return "Linnet generated + CUDA graphs, fast";
  if (name.includes("generated source + torch.compile")) return fast ? "Linnet generated + compile, fast" : "Linnet generated + compile";
  if (name.includes("generated source")) return "Linnet generated";
  if (name.includes("XLA")) return fast ? "Linnet XLA, fast" : "Linnet XLA";
  return name;
}

function title(run: Run): string {
  const size = /H=(\d+) L=(\d+)/.exec(run.config);
  const shape = size ? `H=${size[1]}, ${size[2]} layers` : run.config;
  return `${run.entry} · ${shape}`;
}

const WIDTH = 680;
const LABEL = 230;
const BAR = 20;
const GAP = 7;
const TOP = 6;

interface Bar {
  label: string;
  family: Family;
  fast: boolean;
  value: number;
  x: number;
  y: number;
  width: number;
  text: string;
}

const charts = computed(() =>
  data.runs.map((run) => {
    const variants = run.variants.filter((v) => v.latency_ms !== null);
    const base = variants[0]?.latency_ms ?? 1;
    const measure = (v: Variant) => (latency.value ? (v.latency_ms as number) : base / (v.latency_ms as number));
    const values = variants.map(measure);
    const max = Math.max(...values);
    const plot = WIDTH - LABEL - 64;
    const scale = (value: number) => (plot * value) / max;
    const bars: Bar[] = variants.map((v, i) => {
      const value = values[i];
      return {
        label: label(v.name),
        family: family(v.name),
        fast: v.name.includes("numerics=fast"),
        value,
        x: LABEL,
        y: TOP + i * (BAR + GAP),
        width: Math.max(2, scale(value)),
        text: latency.value
          ? `${value >= 100 ? value.toFixed(0) : value.toFixed(2)} ms`
          : `${value.toFixed(2)}×`,
      };
    });
    const step = max > 8 ? 2 : max > 4 ? 1 : 0.5;
    const ticks = latency.value
      ? [0, 0.25, 0.5, 0.75, 1].map((f) => max * f)
      : Array.from({ length: Math.floor(max / step) + 1 }, (_, i) => i * step);
    return {
      key: `${run.config}-${run.entry}`,
      title: title(run),
      height: TOP + variants.length * (BAR + GAP) + 22,
      bars,
      baseline: LABEL + scale(1),
      ticks: ticks.map((t) => ({
        x: LABEL + (t === 0 ? 0 : scale(t)),
        text: latency.value ? (t >= 10 ? t.toFixed(0) : t >= 1 ? t.toFixed(1) : t.toFixed(2)) : `${t}×`,
      })),
      axisY: TOP + variants.length * (BAR + GAP) + 2,
    };
  }),
);
</script>

<template>
  <div v-if="data.runs.length" class="bench-charts">
    <svg width="0" height="0" aria-hidden="true" class="bench-defs">
      <defs>
        <pattern id="bench-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" fill="var(--vp-c-bg-alt)" />
          <rect width="3" height="6" fill="var(--linnet-red)" fill-opacity="0.75" />
        </pattern>
      </defs>
    </svg>
    <div class="bench-charts-bar">
      <span class="bench-legend">
        <i class="swatch swatch-reference"></i> PyTorch reference
        <i class="swatch swatch-torch"></i> Linnet in PyTorch
        <i class="swatch swatch-xla"></i> Linnet in XLA
        <i class="swatch swatch-fast"></i> numerics=fast
      </span>
      <label class="bench-toggle"><input v-model="latency" type="checkbox" /> show latency instead of speed-up</label>
    </div>
    <div class="bench-grid">
      <figure v-for="chart in charts" :key="chart.key" class="bench-chart">
        <figcaption>{{ chart.title }}</figcaption>
        <svg :viewBox="`0 0 ${WIDTH} ${chart.height}`" role="img" :aria-label="chart.title">
          <g v-for="tick in chart.ticks" :key="tick.text + tick.x" class="bench-tick">
            <line :x1="tick.x" :x2="tick.x" :y1="TOP - 2" :y2="chart.axisY" />
            <text :x="tick.x" :y="chart.axisY + 14" text-anchor="middle">{{ tick.text }}</text>
          </g>
          <line v-if="!latency" class="bench-baseline" :x1="chart.baseline" :x2="chart.baseline" :y1="TOP - 2" :y2="chart.axisY" />
          <g v-for="bar in chart.bars" :key="bar.label" :class="['bench-bar', `bench-bar-${bar.family}`, { 'bench-bar-fast': bar.fast }]">
            <text class="bench-label" :x="LABEL - 8" :y="bar.y + BAR / 2 + 4" text-anchor="end">{{ bar.label }}</text>
            <rect :x="bar.x" :y="bar.y" :width="bar.width" :height="BAR" rx="2" />
            <text class="bench-value" :x="bar.x + bar.width + 6" :y="bar.y + BAR / 2 + 4">{{ bar.text }}</text>
          </g>
        </svg>
      </figure>
    </div>
  </div>
</template>

<style scoped>
.bench-charts {
  margin: 1.5rem 0 2rem;
}
.bench-defs {
  position: absolute;
}
.bench-charts-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.75rem;
  font-size: 0.82rem;
  color: var(--vp-c-text-2);
  margin-bottom: 0.75rem;
}
.bench-legend {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}
.swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  margin-left: 0.6rem;
}
.swatch:first-child {
  margin-left: 0;
}
.swatch-reference {
  background: var(--vp-c-text-3);
}
.swatch-torch {
  background: var(--linnet-red);
}
.swatch-xla {
  background: hsl(358 60% 34%);
}
.swatch-fast {
  background: repeating-linear-gradient(135deg, var(--linnet-red) 0 3px, transparent 3px 6px);
  border: 1px solid var(--linnet-red);
}
.bench-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  cursor: pointer;
}
.bench-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
}
.bench-chart {
  margin: 0;
  padding: 0.75rem 0.9rem 0.5rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 0.25rem;
  background: var(--vp-c-bg-alt);
}
.bench-chart figcaption {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--vp-c-text-2);
  margin-bottom: 0.4rem;
  font-family: var(--vp-font-family-mono);
}
.bench-chart svg {
  width: 100%;
  height: auto;
  display: block;
  font-family: var(--vp-font-family-base);
}
.bench-tick line {
  stroke: var(--vp-c-divider);
  stroke-width: 1;
}
.bench-baseline {
  stroke: var(--vp-c-text-3);
  stroke-width: 1;
  stroke-dasharray: 3 3;
}
.bench-tick text {
  font-size: 10px;
  fill: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
}
.bench-label {
  font-size: 12px;
  fill: var(--vp-c-text-1);
}
.bench-value {
  font-size: 11px;
  fill: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
}
.bench-bar-reference rect {
  fill: var(--vp-c-text-3);
}
.bench-bar-torch rect {
  fill: var(--linnet-red);
}
.bench-bar-xla rect {
  fill: hsl(358 60% 34%);
}
.bench-bar-fast rect {
  fill: url(#bench-hatch);
  stroke: var(--linnet-red);
  stroke-width: 1;
}
.bench-bar-fast.bench-bar-xla rect {
  stroke: hsl(358 60% 34%);
}
</style>
