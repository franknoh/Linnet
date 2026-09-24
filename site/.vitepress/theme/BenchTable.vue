<script setup lang="ts">
// Renders bench/results/latest.json: one table per model configuration,
// one row per runtime variant. The JSON is written by bench/run.py.
import results from "../../../bench/results/latest.json";

interface Variant {
  name: string;
  latency_ms: number | null;
  throughput: number | null;
  max_abs_diff: number | null;
  note?: string;
  kernels?: number | null;
}

interface Run {
  model: string;
  config: string;
  entry: string;
  unit: string;
  variants: Variant[];
}

interface Timing {
  name: string;
  seconds: number;
}

interface Results {
  measured_at: string | null;
  environment: Record<string, string>;
  runs: Run[];
  timings: Timing[];
}

const data = results as Results;

function ms(value: number | null): string {
  return value === null ? "—" : value >= 100 ? value.toFixed(0) : value.toFixed(2);
}

function rate(value: number | null): string {
  return value === null ? "—" : value >= 1000 ? Math.round(value).toLocaleString() : value.toFixed(1);
}

function diff(value: number | null): string {
  return value === null ? "—" : value === 0 ? "0" : value.toExponential(1);
}

function speedup(run: Run, variant: Variant): string {
  const base = run.variants[0];
  if (variant === base || base.latency_ms === null || variant.latency_ms === null) {
    return "";
  }
  return `${(base.latency_ms / variant.latency_ms).toFixed(2)}×`;
}
</script>

<template>
  <div v-if="data.runs.length === 0" class="bench-pending">
    <p>
      No measurements are published yet. The harness is in
      <code>bench/run.py</code>; results appear here once
      <code>bench/results/latest.json</code> is committed.
    </p>
  </div>
  <div v-else>
    <p class="bench-env">
      Measured {{ data.measured_at }} on
      <span v-for="(value, key, i) in data.environment" :key="key">
        <template v-if="i > 0">, </template><strong>{{ key }}</strong> {{ value }}
      </span>.
    </p>
    <div v-for="run in data.runs" :key="run.model + run.config + run.entry" class="bench-run">
      <h3>{{ run.model }} · {{ run.config }} · <code>{{ run.entry }}</code></h3>
      <table>
        <thead>
          <tr>
            <th>Variant</th>
            <th>Latency (ms)</th>
            <th>{{ run.unit }}</th>
            <th>vs reference</th>
            <th>Kernels</th>
            <th>max |Δ|</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="variant in run.variants" :key="variant.name">
            <td>
              {{ variant.name }}
              <span v-if="variant.note" class="bench-note">{{ variant.note }}</span>
            </td>
            <td>{{ ms(variant.latency_ms) }}</td>
            <td>{{ rate(variant.throughput) }}</td>
            <td>{{ speedup(run, variant) }}</td>
            <td>{{ variant.kernels ?? "—" }}</td>
            <td>{{ diff(variant.max_abs_diff) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <h3 v-if="data.timings.length">Compile and load</h3>
    <table v-if="data.timings.length">
      <thead>
        <tr>
          <th>Step</th>
          <th>Seconds</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="timing in data.timings" :key="timing.name">
          <td>{{ timing.name }}</td>
          <td>{{ timing.seconds.toFixed(2) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.bench-pending {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 12px 16px;
  color: var(--vp-c-text-2);
}
.bench-env {
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.bench-note {
  display: block;
  color: var(--vp-c-text-3);
  font-size: 0.85em;
}
</style>
