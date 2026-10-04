<script setup lang="ts">
// Training step times from bench/results/training.json: one chart per run,
// one bar per stack (lower is better, the fastest in the accent), its
// fastest Linnet step against its fastest reference step, and a table of
// what each run measured besides.
import results from "../../../bench/results/training.json";
import { times } from "./compare";

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
  gpus: number;
  title: string;
  quality: string;
  memory?: boolean;
  rows: Row[];
}

const data = results as { date: string; device: string; model: string; note: string; runs: Run[] };

const charts = data.runs.map((run) => {
  const max = Math.max(...run.rows.map((r) => r.step_seconds));
  const best = Math.min(...run.rows.map((r) => r.step_seconds));
  const step = (family: Row["family"]) =>
    Math.min(...run.rows.filter((r) => r.family === family).map((r) => r.step_seconds));
  const speedup = step("reference") / step("linnet");
  return {
    run,
    speedup: times(speedup),
    faster: speedup >= 1,
    bars: run.rows.map((row) => ({
      row,
      best: row.step_seconds === best,
      share: (100 * row.step_seconds) / max,
    })),
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
        <figcaption>
          <span>{{ chart.run.title }}</span>
          <span class="training-headline">
            <strong>{{ chart.speedup }}</strong>
            <span>{{ chart.faster ? "faster" : "slower" }} than TRL</span>
          </span>
        </figcaption>
        <div
          class="training-bars"
          role="img"
          :aria-label="chart.bars.map((b) => `${b.row.stack} ${b.row.step_seconds.toFixed(2)} s`).join(', ')"
        >
          <div
            v-for="bar in chart.bars"
            :key="bar.row.stack"
            :class="['bench-bar', `bench-bar-${bar.row.family}`, { 'bench-bar-best': bar.best }]"
            :title="`${bar.row.stack}: ${bar.row.step_seconds.toFixed(2)} s a step`"
          >
            <span class="bench-label">{{ bar.row.stack }}</span>
            <span class="bench-track">
              <i class="bench-mark" :style="{ width: `${bar.share}%` }"></i>
              <span class="bench-value">{{ bar.row.step_seconds.toFixed(2) }} s</span>
            </span>
          </div>
        </div>
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
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 0.75rem;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--vp-c-text-2);
  margin-bottom: 0.4rem;
}
.training-headline {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  line-height: 1.1;
  white-space: nowrap;
}
.training-headline strong {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  font-variant-numeric: tabular-nums;
}
.training-headline span {
  font-size: 0.7rem;
  font-weight: 400;
  color: var(--vp-c-text-3);
}
.training-bars {
  display: grid;
  grid-template-columns: minmax(5.5rem, max-content) 1fr;
  gap: 0.3rem 0.6rem;
  padding: 0.2rem 0 0.3rem;
}
.bench-bar {
  display: contents;
}
.bench-label {
  font-size: 0.78rem;
  line-height: 1.25;
  text-align: right;
  align-self: center;
  color: var(--vp-c-text-1);
}
.bench-track {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  min-width: 0;
  padding-right: 3.6rem;
}
.bench-mark {
  flex: none;
  height: 16px;
  min-width: 2px;
  border-radius: 0 4px 4px 0;
}
.bench-value {
  flex: none;
  margin-right: -3.6rem;
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  white-space: nowrap;
}
.bench-bar-linnet .bench-mark {
  background: var(--bench-linnet);
}
.bench-bar-reference .bench-mark {
  background: var(--bench-reference);
}
.bench-bar-best .bench-mark {
  background: var(--bench-best);
}
.training-table {
  font-size: 0.82rem;
  margin: 0.5rem 0 0;
}
.training-table td:not(:first-child):not(:last-child) {
  white-space: nowrap;
}
.training-note {
  color: var(--vp-c-text-3);
}
</style>
