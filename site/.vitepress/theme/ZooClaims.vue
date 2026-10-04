<script setup lang="ts">
// What the zoo's measurements let Linnet claim (bench/results/zoo.json, its
// `compare` from the zoo's bench/compare.json). `part` picks the piece: the
// headline numbers, Linnet against the stack it replaces model by model
// (`ids` names the matchups, a tab each), or where each runtime ran. The
// training tile reads bench/results/training.json.
import { computed, ref } from "vue";
import training from "../../../bench/results/training.json";
import zoo from "../../../bench/results/zoo.json";
import {
  type Compare,
  type CompareRow,
  type MatchupResult,
  type PairResult,
  evaluate,
  higher,
  label,
  percent,
  times,
} from "./compare";

interface Row extends CompareRow {
  first_token: number | null;
  notes: string;
}
interface Model {
  name: string;
  title: string;
  task: string;
  parameters: number | null;
  rows: Row[];
}

const props = defineProps<{ part: "tiles" | "matchups" | "coverage"; ids?: string }>();
const data = zoo as unknown as { compare: Compare; models: Model[] };
const compare = data.compare;
const NEST = "https://nest.franknoh.dev/models/";
const models = [...data.models].sort((a, b) => (a.parameters ?? 0) - (b.parameters ?? 0));

const UNITS: Record<string, string> = {
  decode_tok_s: "tok/s",
  serve_tok_s: "tok/s",
  latency_ms: "ms",
  transcribe_ms: "ms",
  step_ms: "ms",
  encode_ms: "ms",
};
const MEASURES: Record<string, string> = {
  decode_tok_s: "decode speed",
  serve_tok_s: "serving throughput",
  latency_ms: "batch-1 latency",
  transcribe_ms: "transcription",
  step_ms: "one denoising step",
  encode_ms: "encoder pass",
};

function amount(v: number, metric: string): string {
  const n =
    v >= 1000 ? Math.round(v).toLocaleString("en-US") : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
  return `${n} ${UNITS[metric] ?? ""}`.trim();
}
function size(parameters: number | null): string {
  if (!parameters) return "";
  return parameters >= 1e9 ? `${(parameters / 1e9).toFixed(1)} B` : `${Math.round(parameters / 1e6)} M`;
}
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function results(id: string): { model: Model; result: MatchupResult }[] {
  const matchup = compare.matchups.find((m) => m.id === id);
  if (!matchup) return [];
  return models.flatMap((model) => {
    const result = evaluate(matchup, model.rows);
    return result ? [{ model, result }] : [];
  });
}
// Each training run's fastest Linnet step against its fastest reference step.
function trainingSpeedups(): { gpus: number; speedup: number }[] {
  const runs = (training as { runs: { gpus: number; rows: { family: string; step_seconds: number }[] }[] }).runs;
  return runs.map((run) => {
    const step = (family: string) =>
      Math.min(...run.rows.filter((r) => r.family === family).map((r) => r.step_seconds));
    return { gpus: run.gpus, speedup: step("reference") / step("linnet") };
  });
}
function sameFirst(pairs: PairResult[]): { same: number; compared: number } {
  const both = pairs.filter((p) => (p.them as Row).first_token != null && (p.us as Row).first_token != null);
  return { same: both.filter((p) => (p.them as Row).first_token === (p.us as Row).first_token).length, compared: both.length };
}

// ---- the headline numbers, each a claim the sections below back

const tiles = computed(() => {
  const out: { href: string; value: string; label: string; detail: string }[] = [];
  const vs = results("vllm");
  if (vs.length) {
    const s = vs.map((r) => r.result.lead.speedup);
    out.push({
      href: "#vs-vllm",
      value: `${s.filter((x) => x > 1).length} of ${vs.length}`,
      label: "decoders run faster than on vLLM, one request at a time",
      detail: `${times(Math.min(...s))} to ${times(Math.max(...s))} vLLM's decode speed`,
    });
  }
  const own = results("pytorch");
  if (own.length) {
    const s = own.map((r) => r.result.lead.speedup);
    out.push({
      href: "#own-framework",
      value: times(median(s)),
      label: "median speed-up over each model's reference implementation in PyTorch",
      detail: `faster on ${s.filter((x) => x > 1).length} of ${own.length} models`,
    });
  }
  const ex = results("export");
  if (ex.length) {
    const pairs = ex.flatMap((r) => r.result.pairs);
    const worst = Math.max(...pairs.map((p) => Math.abs(p.speedup - 1)));
    const { same, compared } = sameFirst(pairs);
    out.push({
      href: "#exports",
      value: `≤ ${Math.max(1, Math.round(worst * 100))}%`,
      label: "from the original checkpoint, running Linnet's export in vLLM, SGLang, and TGI",
      detail: compared ? `the same first token in ${same} of ${compared}` : "",
    });
  }
  const sv = results("serve");
  if (sv.length) {
    const s = sv.map((r) => r.result.lead.speedup);
    out.push({
      href: "#serving",
      value: times(median(s)),
      label: "median serving throughput against vLLM, 256 requests with 64 in flight",
      detail: `ahead on ${s.filter((x) => x > 1).length} of ${sv.length} models`,
    });
  }
  const trained = trainingSpeedups();
  const one = trained.filter((r) => r.gpus === 1).map((r) => r.speedup);
  const many = trained.filter((r) => r.gpus > 1);
  if (one.length) {
    out.push({
      href: "#training",
      value: times(median(one)),
      label: "median training step speed against TRL on one GPU: LoRA SFT, DPO, and GRPO",
      detail: [
        `${times(Math.min(...one))} to ${times(Math.max(...one))}`,
        ...many.map((r) => `${times(r.speedup)} fully sharded on ${r.gpus} GPUs`),
      ].join("; "),
    });
  }
  const linnet = models.flatMap((m) => m.rows.filter((r) => r.kind === "linnet"));
  const ran = linnet.filter((r) => !r.error).length;
  out.push({
    href: "#coverage",
    value: `${ran} of ${linnet.length}`,
    label: `Linnet runs completed, on ${compare.runtimes.length} runtimes from one source`,
    detail: `${linnet.length - ran} failed, each explained below`,
  });
  return out;
});

// ---- Linnet against the stack it replaces, one card per model

const ids = computed(() =>
  (props.ids ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),
);
const tab = ref(ids.value[0] ?? "");
const current = computed(() => compare.matchups.find((m) => m.id === tab.value) ?? null);
const cards = computed(() => results(tab.value));

const verdict = computed(() => {
  const list = cards.value;
  const m = current.value;
  if (!list.length || !m) return "";
  if (m.parity) {
    const pairs = list.flatMap((c) => c.result.pairs);
    const apart = pairs.map((p) => Math.abs(p.speedup - 1));
    const close = apart.filter((d) => d <= 0.05).length;
    const { same, compared } = sameFirst(pairs);
    const first = compared ? `; the same first token in ${same} of ${compared}` : "";
    const within = close === pairs.length ? `all ${pairs.length}` : `${close} of ${pairs.length}`;
    return `Within 5% of the original in ${within} engine and model pairs, ${(median(apart) * 100).toFixed(1)}% apart at the median${first}.`;
  }
  const s = list.map((c) => c.result.lead.speedup);
  const wins = s.filter((x) => x > 1).length;
  return `Faster on ${wins} of ${list.length} models: ${times(Math.min(...s))} to ${times(Math.max(...s))}, median ${times(median(s))}.`;
});
const how = computed(() => {
  const m = current.value;
  if (!m) return "";
  const measure =
    m.metric === "primary"
      ? "Each model by its own measure: decode speed for a decoder, transcription for Whisper, batch-1 latency for the rest."
      : `${MEASURES[m.metric] ?? m.metric} in ${UNITS[m.metric] ?? ""}, ${higher(m.metric) ? "higher" : "lower"} is better.`;
  const pairs = m.parity
    ? "Each card is one model on its own scale, the original's bar above the export's."
    : "Each card is one model on its own scale, the stack's bar above Linnet's; the faster of the two is red.";
  return `${measure} ${pairs} Each side is its fastest configuration.`;
});

const WIDTH = 300;
const END = 78; // room for a value at a bar's end
const PITCH = 30;
const BAR = 11;
const PAIR_GAP = 8;

function card(result: MatchupResult) {
  const bars: {
    key: string;
    y: number;
    text: string;
    width: number;
    kind: "linnet" | "reference";
    best: boolean;
    value: string;
    tip: string;
  }[] = [];
  const max = Math.max(...result.pairs.flatMap((p) => [p.themValue, p.usValue]));
  const scale = (v: number) => Math.max(6, ((WIDTH - END) * v) / max);
  let y = 0;
  for (const pair of result.pairs) {
    const winner = result.matchup.parity ? null : pair.speedup > 1 ? "us" : pair.speedup < 1 ? "them" : null;
    for (const side of ["them", "us"] as const) {
      const row = pair[side];
      const v = side === "them" ? pair.themValue : pair.usValue;
      bars.push({
        key: `${pair.them.key}-${side}`,
        y,
        text: label(compare, row),
        width: scale(v),
        kind: side === "us" ? "linnet" : "reference",
        best: winner === side,
        value: amount(v, result.metric),
        tip: `${row.method}: ${amount(v, result.metric)}`,
      });
      y += PITCH;
    }
    y += PAIR_GAP;
  }
  return { bars, height: y - PAIR_GAP + 2 };
}
function headline(result: MatchupResult): { number: string; word: string } {
  const s = result.lead.speedup;
  if (result.matchup.parity) return { number: percent(s), word: Math.abs(s - 1) < 0.03 ? "the same" : s > 1 ? "faster" : "slower" };
  return { number: times(s), word: s >= 1 ? "faster" : "slower" };
}
const drawn = computed(() => cards.value.map((c) => ({ ...c, drawing: card(c.result), head: headline(c.result) })));

// ---- where each of Linnet's runtimes ran

const TASKS = ["encoder", "vision", "audio", "diffusion", "decoder"];
const coverage = computed(() =>
  [...models]
    .sort((a, b) => TASKS.indexOf(a.task) - TASKS.indexOf(b.task) || (a.parameters ?? 0) - (b.parameters ?? 0))
    .map((model) => ({
    model,
    cells: compare.runtimes.map((runtime) => {
      const rows = model.rows.filter((r) => runtime.keys.includes(r.key));
      const ran = rows.filter((r) => !r.error);
      const state = !rows.length ? "none" : ran.length === rows.length ? "ran" : ran.length ? "partial" : "failed";
      const title = rows.map((r) => `${label(compare, r)}: ${r.error ? `failed (${r.error})` : "ran"}`).join("\n");
      return { id: runtime.id, state, title, count: `${ran.length} of ${rows.length}` };
    }),
  })),
);
</script>

<template>
  <div v-if="part === 'tiles'" class="claims-tiles">
    <a v-for="t in tiles" :key="t.href" :href="t.href" class="claims-tile">
      <span class="claims-value">{{ t.value }}</span>
      <span class="claims-label">{{ t.label }}</span>
      <span v-if="t.detail" class="claims-detail">{{ t.detail }}</span>
    </a>
  </div>

  <div v-else-if="part === 'matchups'" class="claims-matchups">
    <div v-if="ids.length > 1" class="claims-tabs" role="tablist">
      <button
        v-for="id in ids"
        :key="id"
        type="button"
        role="tab"
        :aria-selected="id === tab"
        :class="{ active: id === tab }"
        @click="tab = id"
      >
        {{ compare.matchups.find((m) => m.id === id)?.short ?? id }}
      </button>
    </div>
    <p class="claims-verdict">{{ verdict }}</p>
    <p class="claims-how">
      <span class="claims-key"><i class="swatch swatch-reference"></i>the stack Linnet replaces</span>
      <span class="claims-key"><i class="swatch swatch-linnet"></i>Linnet</span>
      <span v-if="!current?.parity" class="claims-key"><i class="swatch swatch-best"></i>the faster of the two</span>
      <span>{{ how }}</span>
    </p>
    <div class="claims-grid">
      <figure v-for="c in drawn" :key="c.model.name" class="claims-card">
        <figcaption>
          <span class="claims-model">
            <a :href="`${NEST}${c.model.name}/benchmarks`">{{ c.model.title }}</a>
            <span class="claims-size">{{ size(c.model.parameters) }}</span>
            <span v-if="current?.metric === 'primary'" class="claims-measure">{{ MEASURES[c.result.metric] }}</span>
          </span>
          <span class="claims-headline" :class="{ behind: c.result.lead.speedup < 0.97 }">
            <strong>{{ c.head.number }}</strong>
            <span>{{ c.head.word }}</span>
          </span>
        </figcaption>
        <svg
          :viewBox="`0 0 ${WIDTH} ${c.drawing.height}`"
          role="img"
          :aria-label="c.drawing.bars.map((b) => `${b.text} ${b.value}`).join(', ')"
        >
          <g
            v-for="b in c.drawing.bars"
            :key="b.key"
            :class="['claims-bar', `is-${b.kind}`, { 'is-best': b.best }]"
          >
            <title>{{ b.tip }}</title>
            <rect class="claims-hit" x="0" :y="b.y" :width="WIDTH" :height="PITCH - 2" />
            <text class="claims-text" x="0" :y="b.y + 10">{{ b.text }}</text>
            <path
              class="claims-mark"
              :d="`M0,${b.y + 14}h${b.width - 3}a3,3 0 0 1 3,3v${BAR - 6}a3,3 0 0 1 -3,3h${-(b.width - 3)}z`"
            />
            <text class="claims-number" :x="b.width + 6" :y="b.y + 14 + BAR - 1">{{ b.value }}</text>
          </g>
        </svg>
      </figure>
    </div>
  </div>

  <div v-else class="claims-coverage">
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th v-for="r in compare.runtimes" :key="r.id" class="claims-rt">{{ r.title }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in coverage" :key="row.model.name">
          <td class="claims-name">
            <a :href="`${NEST}${row.model.name}/benchmarks`">{{ row.model.title }}</a>
          </td>
          <td v-for="cell in row.cells" :key="cell.id" :class="`is-${cell.state}`" :title="cell.title">
            <template v-if="cell.state === 'ran'">✓</template>
            <template v-else-if="cell.state === 'partial'">✓ <small>{{ cell.count }}</small></template>
            <template v-else-if="cell.state === 'failed'">✕ <small>failed</small></template>
            <template v-else><span class="claims-none" aria-label="not run">·</span></template>
          </td>
        </tr>
      </tbody>
    </table>
    <p class="claims-how">
      ✓ ran · ✓ <small>n of m</small> some of its configurations ran · ✕ failed · · not a runtime this model takes part in.
      Hover a cell for each configuration.
    </p>
  </div>
</template>

<style scoped>
.claims-tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 168px), 1fr));
  gap: 0.75rem;
  margin: 1.5rem 0 2rem;
}
.claims-tile {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  padding: 0.9rem 1rem 0.85rem;
  border: 1px solid var(--vp-c-divider);
  border-top: 3px solid var(--bench-linnet);
  border-radius: 0.25rem;
  background: var(--vp-c-bg-alt);
  text-decoration: none !important;
  color: inherit !important;
  transition: border-color 0.15s;
}
.claims-tile:hover {
  border-top-color: var(--bench-best);
}
.claims-value {
  font-size: 1.85rem;
  line-height: 1.1;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--vp-c-text-1);
  font-variant-numeric: tabular-nums;
}
.claims-label {
  font-size: 0.82rem;
  line-height: 1.35;
  color: var(--vp-c-text-2);
}
.claims-detail {
  margin-top: auto;
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
}

.claims-matchups {
  margin: 1rem 0 2rem;
}
.claims-tabs {
  display: inline-flex;
  flex-wrap: wrap;
  border: 1px solid var(--vp-c-divider);
  border-radius: 0.25rem;
  overflow: hidden;
  margin-bottom: 0.8rem;
}
.claims-tabs button {
  padding: 0.25rem 0.8rem;
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
  background: transparent;
}
.claims-tabs button + button {
  border-left: 1px solid var(--vp-c-divider);
}
.claims-tabs button.active {
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg-soft);
  font-weight: 600;
}
.claims-verdict {
  font-size: 1.15rem;
  font-weight: 650;
  line-height: 1.4;
  color: var(--vp-c-text-1);
  margin: 0 0 0.35rem;
}
.claims-how {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.25rem 0.9rem;
  font-size: 0.78rem;
  line-height: 1.45;
  color: var(--vp-c-text-2);
  margin: 0 0 0.9rem;
}
.claims-key {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  white-space: nowrap;
}
.swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
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
.claims-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 270px), 1fr));
  gap: 0.75rem;
}
.claims-card {
  margin: 0;
  padding: 0.7rem 0.85rem 0.6rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 0.25rem;
  background: var(--vp-c-bg-alt);
}
.claims-card figcaption {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 0.5rem;
  margin-bottom: 0.55rem;
}
.claims-model {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.1rem 0.45rem;
  font-size: 0.84rem;
  min-width: 0;
}
.claims-model a {
  font-weight: 600;
  color: var(--vp-c-text-1);
  text-decoration: none;
}
.claims-model a:hover {
  color: var(--linnet-red);
}
.claims-size,
.claims-measure {
  font-size: 0.72rem;
  color: var(--vp-c-text-3);
}
.claims-size {
  font-family: var(--vp-font-family-mono);
}
.claims-measure {
  flex-basis: 100%;
}
.claims-headline {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  line-height: 1.1;
  white-space: nowrap;
}
.claims-headline strong {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  font-variant-numeric: tabular-nums;
}
.claims-headline span {
  font-size: 0.7rem;
  color: var(--vp-c-text-3);
}
.claims-headline.behind strong {
  color: var(--vp-c-text-2);
}
.claims-card svg {
  width: 100%;
  height: auto;
  display: block;
  overflow: visible;
  font-family: var(--vp-font-family-base);
}
.claims-hit {
  fill: transparent;
}
.claims-bar:hover .claims-hit {
  fill: var(--vp-c-bg-soft);
}
.claims-text {
  font-size: 10.5px;
  fill: var(--vp-c-text-2);
}
.is-linnet .claims-text {
  fill: var(--vp-c-text-1);
  font-weight: 600;
}
.claims-number {
  font-size: 10.5px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
.is-reference .claims-mark {
  fill: var(--bench-reference);
}
.is-linnet .claims-mark {
  fill: var(--bench-linnet);
}
.is-best .claims-mark {
  fill: var(--bench-best);
}

.claims-coverage {
  margin: 1rem 0 2rem;
  overflow-x: auto;
}
.claims-coverage table {
  font-size: 0.8rem;
  margin: 0 0 0.5rem;
  display: table;
  width: 100%;
}
.claims-coverage th,
.claims-coverage td {
  padding: 0.22rem 0.22rem;
}
.claims-coverage td {
  text-align: center;
  white-space: nowrap;
  color: var(--vp-c-text-1);
}
.claims-coverage td.claims-name {
  text-align: left;
  font-size: 0.76rem;
  font-weight: 500;
  white-space: nowrap;
  padding-left: 0.5rem;
}
.claims-name a {
  color: var(--vp-c-text-1);
  text-decoration: none;
}
.claims-name a:hover {
  color: var(--linnet-red);
}
.claims-rt {
  font-size: 0.68rem;
  font-weight: 600;
  line-height: 1.2;
  text-transform: none;
  letter-spacing: 0;
  vertical-align: bottom;
  white-space: normal;
}
.claims-coverage small {
  font-size: 0.7rem;
  color: var(--vp-c-text-3);
}
.claims-coverage td.is-failed {
  color: var(--bench-best);
}
.claims-none {
  color: var(--vp-c-text-3);
}
</style>
