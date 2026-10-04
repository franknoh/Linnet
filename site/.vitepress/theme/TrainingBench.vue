<script setup lang="ts">
// Training step times from bench/results/training.json: one chart per run,
// one bar per stack (lower is better, the fastest in the accent), and a
// table of what each run measured besides.
import results from "../../../bench/results/training.json";

interface Row {
  stack: string;
  family: "linnet" | "reference";
  step_seconds: number;
  tokens_per_second?: number;
  peak_gib?: number;
  quality: string;
  note?: string;
}

interface Run {
  id: string;
  title: string;
  quality: string;
  memory?: boolean;
  rows: Row[];
}

const data = results as { date: string; device: string; model: string; note: string; runs: Run[] };

const WIDTH = 680;
const LABEL = 170;
const BAR = 20;
const GAP = 7;
const TOP = 6;

const charts = data.runs.map((run) => {
  const max = Math.max(...run.rows.map((r) => r.step_seconds));
  const best = Math.min(...run.rows.map((r) => r.step_seconds));
  const plot = WIDTH - LABEL - 64;
  const scale = (value: number) => (plot * value) / max;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => max * f);
  return {
    run,
    height: TOP + run.rows.length * (BAR + GAP) + 22,
    axisY: TOP + run.rows.length * (BAR + GAP) + 2,
    bars: run.rows.map((row, i) => ({
      row,
      best: row.step_seconds === best,
      y: TOP + i * (BAR + GAP),
      width: Math.max(2, scale(row.step_seconds)),
    })),
    ticks: ticks.map((t) => ({ x: LABEL + scale(t), text: t >= 1 ? t.toFixed(1) : t.toFixed(2) })),
  };
});

function thousands(value?: number): string {
  return value === undefined ? "" : `${(value / 1000).toFixed(1)}K`;
}
</script>

<template>
  <div class="training-bench">
    <p class="training-meta">
      {{ data.model }} on {{ data.device }}, {{ data.date }}. {{ data.note }}
    </p>
    <div class="bench-charts-bar">
      <span class="bench-legend">
        <i class="swatch swatch-best"></i> fastest step
        <i class="swatch swatch-linnet"></i> Linnet
        <i class="swatch swatch-reference"></i> TRL
      </span>
      <span>seconds a step, lower is better</span>
    </div>
    <div v-for="chart in charts" :key="chart.run.id" class="training-run">
      <figure class="bench-chart">
        <figcaption>{{ chart.run.title }}</figcaption>
        <svg :viewBox="`0 0 ${WIDTH} ${chart.height}`" role="img" :aria-label="chart.run.title">
          <g v-for="tick in chart.ticks" :key="tick.text" class="bench-tick">
            <line :x1="tick.x" :x2="tick.x" :y1="TOP - 2" :y2="chart.axisY" />
            <text :x="tick.x" :y="chart.axisY + 14" text-anchor="middle">{{ tick.text }}</text>
          </g>
          <g
            v-for="bar in chart.bars"
            :key="bar.row.stack"
            :class="['bench-bar', `bench-bar-${bar.row.family}`, { 'bench-bar-best': bar.best }]"
          >
            <text class="bench-label" :x="LABEL - 8" :y="bar.y + BAR / 2 + 4" text-anchor="end">{{ bar.row.stack }}</text>
            <rect :x="LABEL" :y="bar.y" :width="bar.width" :height="BAR" rx="2" />
            <text class="bench-value" :x="LABEL + bar.width + 6" :y="bar.y + BAR / 2 + 4">{{ bar.row.step_seconds.toFixed(2) }} s</text>
          </g>
        </svg>
      </figure>
      <table class="training-table">
        <thead>
          <tr>
            <th>Stack</th>
            <th>Step</th>
            <th v-if="chart.run.rows.some((r) => r.tokens_per_second)">Tokens/s</th>
            <th v-if="chart.run.memory !== false">Peak a GPU</th>
            <th>{{ chart.run.quality }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in chart.run.rows" :key="row.stack">
            <td>{{ row.stack }}<span v-if="row.note" class="training-note"> ({{ row.note }})</span></td>
            <td>{{ row.step_seconds.toFixed(2) }} s</td>
            <td v-if="chart.run.rows.some((r) => r.tokens_per_second)">{{ thousands(row.tokens_per_second) }}</td>
            <td v-if="chart.run.memory !== false">{{ row.peak_gib?.toFixed(1) }} GiB</td>
            <td>{{ row.quality }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.training-bench {
  margin: 1.25rem 0 2rem;
}
.training-meta {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
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
.swatch-best {
  background: var(--bench-best);
}
.swatch-linnet {
  background: var(--bench-linnet);
}
.swatch-reference {
  background: var(--bench-reference);
}
.training-run {
  margin-bottom: 1.5rem;
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
.bench-bar-linnet rect {
  fill: var(--bench-linnet);
}
.bench-bar-reference rect {
  fill: var(--bench-reference);
}
.bench-bar-best rect {
  fill: var(--bench-best);
}
.training-table {
  font-size: 0.82rem;
  margin: 0.5rem 0 0;
}
.training-note {
  color: var(--vp-c-text-3);
}
</style>
