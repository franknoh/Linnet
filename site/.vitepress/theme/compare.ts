// How benchmark rows relate, read from the zoo's `bench/compare.json`: which
// runtime each row belongs to, and which stack each Linnet row replaces. The
// same file serves linnet.franknoh.dev and nest.franknoh.dev, so both read a
// comparison the same way.

export interface CompareRow {
  key: string;
  method: string;
  kind: string; // "linnet" or "reference"
  metrics: Record<string, number | null | undefined>;
  error: string | null;
}

export interface Group {
  id: string;
  title: string;
  keys: string[];
}

export interface Pair {
  them: string[];
  us: string[];
}

export interface Matchup {
  id: string;
  title: string;
  short: string;
  group: string | null;
  metric: string; // a metric key, or "primary": the model's own
  parity?: boolean; // an export against the original: the question is "the same?"
  pairs: Pair[];
}

export interface Runtime {
  id: string;
  title: string;
  keys: string[];
}

export interface Compare {
  groups: Group[];
  matchups: Matchup[];
  runtimes: Runtime[];
  labels: Record<string, string>;
}

// A row's short name; the harness's own `method` where the file has none.
export function label(compare: Compare, row: CompareRow): string {
  return compare.labels[row.key] ?? row.method;
}

// The number a model is judged by, in the order tried: a decoder's decode
// speed, a speech model's transcription, everything else's batch-1 latency.
export const PRIMARY = ["decode_tok_s", "transcribe_ms", "latency_ms", "step_ms", "encode_ms"];

// Rates are higher-is-better; times, memory, and load are lower-is-better.
export function higher(metric: string): boolean {
  return metric.endsWith("_tok_s") || metric.endsWith("_per_s");
}

export function value(row: CompareRow | undefined, metric: string): number | null {
  if (!row || row.error) return null;
  const v = row.metrics[metric];
  return typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null;
}

export function primary(rows: CompareRow[]): string | null {
  return PRIMARY.find((m) => rows.some((r) => value(r, m) !== null)) ?? null;
}

// The best row among `keys` on `metric`; a tie goes to the key listed first.
function best(rows: CompareRow[], keys: string[], metric: string): CompareRow | null {
  let found: CompareRow | null = null;
  let score = 0;
  for (const key of keys) {
    const v = value(
      rows.find((r) => r.key === key),
      metric,
    );
    if (v === null) continue;
    const s = higher(metric) ? v : -v;
    if (found === null || s > score) {
      found = rows.find((r) => r.key === key) ?? null;
      score = s;
    }
  }
  return found;
}

export interface PairResult {
  them: CompareRow;
  us: CompareRow;
  themValue: number;
  usValue: number;
  // How many times faster Linnet's row is: above 1 it wins, below it loses.
  speedup: number;
}

export interface MatchupResult {
  matchup: Matchup;
  metric: string;
  pairs: PairResult[];
  // The pair that speaks for the matchup: the only one, or for parity the one
  // furthest from even.
  lead: PairResult;
}

export function evaluate(matchup: Matchup, rows: CompareRow[]): MatchupResult | null {
  const metric = matchup.metric === "primary" ? primary(rows) : matchup.metric;
  if (!metric) return null;
  const pairs: PairResult[] = [];
  for (const pair of matchup.pairs) {
    const them = best(rows, pair.them, metric);
    const us = best(rows, pair.us, metric);
    if (!them || !us) continue;
    const themValue = value(them, metric) as number;
    const usValue = value(us, metric) as number;
    pairs.push({ them, us, themValue, usValue, speedup: higher(metric) ? usValue / themValue : themValue / usValue });
  }
  if (!pairs.length) return null;
  const lead = matchup.parity
    ? pairs.reduce((a, b) => (Math.abs(Math.log(b.speedup)) > Math.abs(Math.log(a.speedup)) ? b : a))
    : pairs[0];
  return { matchup, metric, pairs, lead };
}

// Rows in runtime order, each runtime's stacks before Linnet; a key the file
// does not know goes last, in a group of its own.
export function grouped<R extends CompareRow>(compare: Compare, rows: R[]): { group: Group; rows: R[] }[] {
  const out: { group: Group; rows: R[] }[] = [];
  const placed = new Set<string>();
  for (const group of compare.groups) {
    const members = group.keys.map((k) => rows.find((r) => r.key === k)).filter((r): r is R => !!r);
    members.forEach((r) => placed.add(r.key));
    if (members.length) out.push({ group, rows: members });
  }
  const rest = rows.filter((r) => !placed.has(r.key));
  if (rest.length) out.push({ group: { id: "other", title: "Other", keys: [] }, rows: rest });
  return out;
}

// What a Linnet row is measured against on its own ground: the matchup of its
// runtime it takes part in, and that matchup's best stack on `metric`.
export function against(
  compare: Compare,
  rows: CompareRow[],
  row: CompareRow,
  metric: string,
): { them: CompareRow; speedup: number; parity: boolean } | null {
  const group = compare.groups.find((g) => g.keys.includes(row.key));
  if (!group || row.kind !== "linnet") return null;
  for (const matchup of compare.matchups) {
    if (matchup.group !== group.id) continue;
    const pair = matchup.pairs.find((p) => p.us.includes(row.key));
    if (!pair) continue;
    const them = best(rows, pair.them, metric);
    const mine = value(row, metric);
    const theirs = value(them ?? undefined, metric);
    if (!them || mine === null || theirs === null) return null;
    return { them, speedup: higher(metric) ? mine / theirs : theirs / mine, parity: !!matchup.parity };
  }
  return null;
}

// "2.3×" for a speed-up, "0.86×" below one; parity reads as a percentage.
export function times(speedup: number): string {
  if (speedup >= 10) return `${speedup.toFixed(0)}×`;
  return `${speedup.toFixed(2)}×`;
}

export function percent(speedup: number): string {
  const d = (speedup - 1) * 100;
  const rounded = Math.abs(d) < 0.05 ? 0 : d;
  return `${rounded > 0 ? "+" : rounded < 0 ? "−" : "±"}${Math.abs(rounded).toFixed(1)}%`;
}
