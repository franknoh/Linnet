<script setup lang="ts">
// The model zoo's measurements from bench/results/zoo.json (written by
// bench/zoo.py): real checkpoints on one GPU, against the stacks people
// already run them with. `part` picks what to draw: the decoders' small
// multiples, everything else's, or the full table for one model.
import { computed, ref } from "vue";
import zoo from "../../../bench/results/zoo.json";

interface Row {
  key: string;
  method: string;
  kind: string;
  metrics: Record<string, number>;
  max_abs_diff: number | null;
  notes: string;
  error: string | null;
}

interface Model {
  name: string;
  title: string;
  task: string;
  parameters: number | null;
  reference: string | null;
  primary: string | null;
  rows: Row[];
}

const props = defineProps<{ part: "decoders" | "serving" | "others" | "table" }>();
const data = zoo as unknown as { date: string; environment: Record<string, string>; models: Model[] };
const NEST = "https://nest.franknoh.dev/models/";

type Family = "reference" | "linnet";

function family(row: Row): Family {
  return row.kind === "linnet" ? "linnet" : "reference";
}

const LABELS: Record<string, string> = {
  "transformers-eager": "transformers",
  "transformers-compile": "transformers, compiled",
  "diffusers-eager": "diffusers",
  "diffusers-compile": "diffusers, compiled",
  "sentence-transformers": "sentence-transformers",
  vllm: "vLLM",
  "linnet-torch": "Linnet, generated",
  "linnet-cudagraphs": "Linnet, CUDA graphs",
  "linnet-jax": "Linnet, XLA",
  "linnet-onnx": "Linnet, ONNX Runtime",
  "linnet-offload": "Linnet, offloaded",
  "keras-hub": "KerasHub (JAX)",
  "onnx-reference": "torch.onnx, ONNX Runtime",
  "triton-onnx": "Triton, torch.onnx",
  "triton-linnet-onnx": "Triton, Linnet ONNX",
  "serve-vllm": "vLLM",
  "serve-transformers": "transformers, batched",
  "serve-keras-hub": "KerasHub, static batches",
  "serve-triton-vllm": "Triton, vLLM backend",
  "serve-linnet-torch": "Linnet serve, CUDA graphs",
  "serve-linnet-jax": "Linnet serve, XLA",
};
const label = (row: Row) => LABELS[row.key] ?? row.method;

function size(parameters: number | null): string {
  if (!parameters) return "";
  return parameters >= 1e9 ? `${(parameters / 1e9).toFixed(1)} B` : `${Math.round(parameters / 1e6)} M`;
}

// What the charts can show. A view reads one number off a row; `speedup`
// divides the reference row's time by each row's.
interface View {
  id: string;
  label: string;
  metric: (m: Model) => string | null;
  speedup: boolean;
  unit: string;
}
const DECODER_VIEWS: View[] = [
  { id: "decode", label: "Decode speed", metric: () => "decode_tok_s", speedup: false, unit: "tok/s" },
  { id: "ttft", label: "Time to first token", metric: () => "ttft_ms", speedup: false, unit: "ms" },
  { id: "memory", label: "Peak GPU memory", metric: () => "peak_vram_mib", speedup: false, unit: "GiB" },
];
const OTHER_VIEWS: View[] = [
  { id: "latency", label: "Latency", metric: (m) => m.primary, speedup: false, unit: "ms" },
  { id: "throughput", label: "Throughput", metric: () => "throughput_per_s", speedup: false, unit: "/s" },
  { id: "memory", label: "Peak GPU memory", metric: () => "peak_vram_mib", speedup: false, unit: "GiB" },
];
const SERVING_VIEWS: View[] = [
  { id: "serve", label: "Throughput", metric: () => "serve_tok_s", speedup: false, unit: "tok/s" },
  { id: "serve_ttft", label: "Time to first token", metric: () => "serve_ttft_ms", speedup: false, unit: "ms" },
  { id: "memory", label: "Peak GPU memory", metric: () => "peak_vram_mib", speedup: false, unit: "GiB" },
];
const views = computed(() =>
  props.part === "decoders" ? DECODER_VIEWS : props.part === "serving" ? SERVING_VIEWS : OTHER_VIEWS,
);
const viewId = ref(props.part === "decoders" ? "decode" : props.part === "serving" ? "serve" : "latency");
const view = computed(() => views.value.find((v) => v.id === viewId.value) ?? views.value[0]);

const models = computed(() =>
  data.models
    .filter((m) => (props.part === "others" ? m.task !== "decoder" : m.task === "decoder"))
    // Serving rows (`serve-*`) are their own section; the others, single requests.
    .map((m) => ({
      ...m,
      rows: m.rows.filter((r) => (props.part === "serving") === r.key.startsWith("serve-")),
    }))
    .filter((m) => m.rows.length > 0)
    .sort((a, b) => (a.parameters ?? 0) - (b.parameters ?? 0)),
);

const PRIMARY_LABEL: Record<string, string> = {
  latency_ms: "forward pass",
  step_ms: "one denoising step",
  encode_ms: "encoder",
};

const WIDTH = 460;
const LABEL = 150;
const BAR = 16;
const GAP = 6;
const TOP = 4;

function format(value: number, v: View): string {
  if (v.speedup) return `${value.toFixed(2)}×`;
  if (v.unit === "GiB") return `${value.toFixed(1)} GiB`;
  if (v.unit === "ms") return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ms`;
  if (v.unit === "/s") return `${value >= 1000 ? value.toFixed(0) : value.toFixed(1)}/s`;
  return `${value.toFixed(0)} ${v.unit}`;
}

function hover(row: Row): string {
  const lines = [row.method];
  for (const [k, v] of Object.entries(row.metrics)) lines.push(`${k}: ${v >= 100 ? v.toFixed(0) : v.toFixed(2)}`);
  if (row.max_abs_diff !== null) lines.push(`max |diff| vs reference: ${row.max_abs_diff}`);
  if (row.notes) lines.push(row.notes);
  if (row.error) lines.push(`failed: ${row.error}`);
  return lines.join("\n");
}

const charts = computed(() => {
  const v = view.value;
  return models.value.map((whole) => {
    const metric = v.metric(whole);
    // Rows that measured this, and rows that failed: never "not measured".
    // A reserved pool (vLLM) and a deliberate cap (offloading) are settings,
    // not memory a model needed: they are left out of the memory view.
    const unlike = (r: Row) =>
      v.unit === "GiB" &&
      (r.key === "linnet-offload" || /reserv\w* .*pool|pool .*reserv|gpu_memory_utilization/i.test(r.notes));
    const model = {
      ...whole,
      rows: whole.rows.filter((r) => !unlike(r) && (r.error || (metric !== null && metric in r.metrics))),
    };
    const reference = model.rows.find((r) => r.key === model.reference);
    const base = metric ? reference?.metrics[metric] : undefined;
    const value = (row: Row): number | null => {
      const raw = metric ? row.metrics[metric] : undefined;
      if (raw === undefined || raw === null) return null;
      if (v.speedup) return base ? base / raw : null;
      return v.unit === "GiB" ? raw / 1024 : raw;
    };
    const values = model.rows.map(value);
    const max = Math.max(1, ...values.filter((x): x is number => x !== null));
    const plot = WIDTH - LABEL - 78;
    const scale = (x: number) => (plot * x) / max;
    // The best bar in the accent: speed-ups and rates are higher-is-better,
    // times and memory lower. A reserved pool is a setting, so it never wins.
    const lower = !v.speedup && (v.unit === "ms" || v.unit === "GiB");
    // The offloaded row runs under a cap by design and wins nothing.
    const eligible = values.map((x, i) => (x !== null && !model.rows[i].error && model.rows[i].key !== "linnet-offload" ? x : null));
    const present = eligible.filter((x): x is number => x !== null);
    const target = present.length ? (lower ? Math.min(...present) : Math.max(...present)) : null;
    const bars = model.rows.map((row, i) => {
      const x = values[i];
            return {
        key: row.key,
        label: label(row),
        family: family(row),
        best: target !== null && eligible[i] === target,
        y: TOP + i * (BAR + GAP),
        width: x === null ? 0 : Math.max(2, scale(x)),
        text: row.error ? "failed" : x === null ? "not measured" : format(x, v),
        hover: hover(row),
      };
    });
    return {
      key: model.name,
      title: model.title,
      size: size(model.parameters),
      what: v.id === "latency" && model.primary ? `${PRIMARY_LABEL[model.primary] ?? model.primary}, batch 1` : v.id === "throughput" ? "items per second at the family's large batch" : "",
      href: `${NEST}${model.name}/benchmarks`,
      height: TOP + model.rows.length * (BAR + GAP) + 2,
      bars,
      baseline: v.speedup ? LABEL + scale(1) : null,
    };
  }).filter((chart) => chart.bars.some((bar) => bar.width > 0)); // nothing measured: no card
});

// The table: every row of one model.
const selected = ref("llama-3.1-8b-instruct");
const table = computed(() => data.models.find((m) => m.name === selected.value) ?? data.models[0]);
const COLUMNS: [string, string][] = [
  ["ttft_ms", "TTFT ms"],
  ["decode_tok_s", "tok/s"],
  ["latency_ms", "latency ms"],
  ["throughput_per_s", "items/s"],
  ["step_ms", "step ms"],
  ["encode_ms", "encode ms"],
  ["transcribe_ms", "transcribe ms"],
  ["peak_vram_mib", "peak GiB"],
  ["load_s", "load s"],
];
const columns = computed(() => COLUMNS.filter(([k]) => table.value.rows.some((r) => k in r.metrics)));
function cell(row: Row, key: string): string {
  const x = row.metrics[key];
  if (x === undefined) return "";
  if (key === "peak_vram_mib") return (x / 1024).toFixed(1);
  return x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2);
}
function diff(row: Row): string {
  if (row.max_abs_diff === null || row.max_abs_diff === undefined) return "";
  return row.max_abs_diff === 0 ? "0" : row.max_abs_diff.toPrecision(2);
}
</script>

<template>
  <div v-if="part !== 'table'" class="zoo">
    <div class="zoo-bar">
      <span class="zoo-legend">
        <i class="swatch swatch-best"></i> best in the chart
        <i class="swatch swatch-linnet"></i> Linnet
        <i class="swatch swatch-reference"></i> existing stacks
      </span>
      <span class="zoo-views" role="group" aria-label="What the bars show">
        <button
          v-for="v in views"
          :key="v.id"
          type="button"
          :class="{ active: v.id === viewId }"
          :aria-pressed="v.id === viewId"
          @click="viewId = v.id"
        >
          {{ v.label }}
        </button>
      </span>
    </div>
    <div class="zoo-grid">
      <figure v-for="chart in charts" :key="chart.key" class="zoo-chart">
        <figcaption>
          <a :href="chart.href">{{ chart.title }}</a>
          <span class="zoo-size">{{ chart.size }}</span>
          <span v-if="chart.what" class="zoo-what">{{ chart.what }}</span>
        </figcaption>
        <svg :viewBox="`0 0 ${WIDTH} ${chart.height}`" role="img" :aria-label="`${chart.title}: ${view.label}`">
          <line
            v-if="chart.baseline !== null"
            class="zoo-baseline"
            :x1="chart.baseline"
            :x2="chart.baseline"
            :y1="0"
            :y2="chart.height"
          />
          <g v-for="bar in chart.bars" :key="bar.key" :class="['zoo-row', `zoo-${bar.family}`, { 'zoo-best': bar.best }]">
            <title>{{ bar.hover }}</title>
            <rect class="zoo-hit" x="0" :y="bar.y - GAP / 2" :width="WIDTH" :height="BAR + GAP" />
            <text class="zoo-label" :x="LABEL - 8" :y="bar.y + BAR / 2 + 4" text-anchor="end">{{ bar.label }}</text>
            <rect v-if="bar.width" class="zoo-mark" :x="LABEL" :y="bar.y" :width="bar.width" :height="BAR" rx="2" />
            <text class="zoo-value" :x="LABEL + bar.width + 6" :y="bar.y + BAR / 2 + 4">{{ bar.text }}</text>
          </g>
        </svg>
      </figure>
    </div>
  </div>
  <div v-else class="zoo-table">
    <label class="zoo-select">
      Model
      <select v-model="selected">
        <option v-for="m in data.models" :key="m.name" :value="m.name">{{ m.title }}</option>
      </select>
      <a :href="`${NEST}${table.name}/benchmarks`">its page in the zoo</a>
    </label>
    <div class="zoo-scroll">
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th v-for="[k, name] in columns" :key="k" class="num">{{ name }}</th>
            <th class="num">max |diff|</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in table.rows" :key="row.key" :class="{ linnet: row.kind === 'linnet' }">
            <td>{{ row.method }}</td>
            <td v-for="[k] in columns" :key="k" class="num">{{ cell(row, k) }}</td>
            <td class="num">{{ diff(row) }}</td>
            <td class="notes">{{ row.error ? `failed: ${row.error}` : row.notes }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.zoo {
  margin: 1.5rem 0 2rem;
}
.zoo-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.75rem;
  font-size: 0.82rem;
  color: var(--vp-c-text-2);
  margin-bottom: 0.75rem;
}
.zoo-legend {
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
.zoo-views {
  display: inline-flex;
  border: 1px solid var(--vp-c-divider);
  border-radius: 0.25rem;
  overflow: hidden;
}
.zoo-views button {
  padding: 0.2rem 0.7rem;
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
  background: transparent;
}
.zoo-views button + button {
  border-left: 1px solid var(--vp-c-divider);
}
.zoo-views button.active {
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg-soft);
  font-weight: 600;
}
.zoo-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr));
  gap: 0.75rem;
}
.zoo-chart {
  margin: 0;
  padding: 0.6rem 0.8rem 0.5rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 0.25rem;
  background: var(--vp-c-bg-alt);
}
.zoo-chart figcaption {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.2rem 0.5rem;
  font-size: 0.82rem;
  margin-bottom: 0.4rem;
}
.zoo-chart figcaption a {
  font-weight: 600;
  color: var(--vp-c-text-1);
  text-decoration: none;
}
.zoo-chart figcaption a:hover {
  color: var(--linnet-red);
}
.zoo-size {
  font-family: var(--vp-font-family-mono);
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
}
.zoo-what {
  flex-basis: 100%;
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
}
.zoo-chart svg {
  width: 100%;
  height: auto;
  display: block;
  font-family: var(--vp-font-family-base);
}
.zoo-hit {
  fill: transparent;
}
.zoo-row:hover .zoo-hit {
  fill: var(--vp-c-bg-soft);
}
.zoo-baseline {
  stroke: var(--vp-c-text-3);
  stroke-width: 1;
  stroke-dasharray: 3 3;
}
.zoo-label {
  font-size: 11px;
  fill: var(--vp-c-text-1);
}
.zoo-value {
  font-size: 10.5px;
  paint-order: stroke;
  stroke: var(--vp-c-bg-alt);
  stroke-width: 3px;
  fill: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
}
.zoo-reference .zoo-mark {
  fill: var(--bench-reference);
}
.zoo-linnet .zoo-mark {
  fill: var(--bench-linnet);
}
.zoo-best .zoo-mark {
  fill: var(--bench-best);
}
.zoo-table {
  margin: 1rem 0 2rem;
}
.zoo-select {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.6rem;
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  margin-bottom: 0.5rem;
}
.zoo-select select {
  border: 1px solid var(--vp-c-divider);
  border-radius: 0.25rem;
  padding: 0.2rem 0.4rem;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
}
.zoo-scroll {
  overflow-x: auto;
}
.zoo-scroll table {
  font-size: 0.8rem;
  margin: 0;
}
.num {
  text-align: right;
  font-family: var(--vp-font-family-mono);
  white-space: nowrap;
}
.notes {
  color: var(--vp-c-text-2);
  min-width: 14rem;
}
tr.linnet td:first-child {
  color: var(--vp-c-text-1);
  font-weight: 600;
}
</style>
