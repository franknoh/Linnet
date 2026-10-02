<script setup lang="ts">
// The landing page's picture of where a Linnet model goes. Linnet is the
// root on the left; two groups hang off it. Frameworks connect both ways (a
// PyTorch, JAX, or ONNX model imports into Linnet, and a Linnet model runs in
// each); targets are what the compiler and exporters write, and from them a
// rail runs to the devices. Two layouts share one description: a wide one,
// and a tree for narrow screens with the groups stacked down a trunk.
import { useId } from "vue";
import { withBase } from "vitepress";
import { logos } from "./archLogos";

interface Chip {
  label: string;
  logo?: string;
  open?: boolean; // a dashed outline: a kind of thing, open-ended
}

const frameworks: Chip[] = [
  { label: "PyTorch", logo: "pytorch" },
  { label: "JAX" },
  { label: "ONNX", logo: "onnx" },
];

const targets: Chip[] = [
  { label: "StableHLO · MLIR", logo: "llvm" },
  { label: "ONNX Runtime", logo: "onnx" },
  { label: "TensorRT", logo: "nvidia" },
  { label: "vLLM · SGLang · TGI", logo: "vllm" },
  { label: "GGUF · llama.cpp" },
  { label: "Ollama", logo: "ollama" },
  { label: "Triton Server" },
  { label: "Custom · plan IR", open: true },
];

const devices = ["GPU", "TPU", "NPU", "CPU", "…"];

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface PlacedChip extends Box {
  chip: Chip;
}

interface Group extends Box {
  title: string;
  note: string;
  chips: PlacedChip[];
}

interface Layout {
  width: number;
  height: number;
  font: number; // chip label size
  linnet: Box;
  groups: Group[];
  accent: { d: string; both: boolean }[]; // Linnet's own connectors
  rail: string[]; // targets to devices, no arrowheads
  arrows: string[]; // rail to each device
  devices: PlacedChip[];
  devicesLabel: { x: number; y: number };
}

// Chips laid out in `columns` columns from (x, y), `w` wide and `h` tall.
function grid(
  chips: Chip[],
  x: number,
  y: number,
  w: number,
  h: number,
  columns: number,
  gap: number,
): PlacedChip[] {
  return chips.map((chip, i) => ({
    chip,
    x: x + (i % columns) * (w + gap),
    y: y + Math.floor(i / columns) * (h + gap),
    w,
    h,
  }));
}

function wide(): Layout {
  const linnet = { x: 2, y: 8, w: 178, h: 84 };
  const fx = 258;
  const groupW = 430;
  const frameworksBox = { x: fx, y: 8, w: groupW, h: 84 };
  const targetsBox = { x: fx, y: 136, w: groupW, h: 210 };
  const fChipW = (groupW - 32 - 2 * 10) / 3;
  const tChipW = (groupW - 32 - 10) / 2;
  const fy = frameworksBox.y + frameworksBox.h / 2;
  const ty = targetsBox.y + targetsBox.h / 2;
  const cx = linnet.x + linnet.w / 2;
  const right = targetsBox.x + targetsBox.w;
  const railX = right + 60;
  const chipX = railX + 40;
  const spacing = 42;
  const centers = devices.map((_, i) => ty + (i - (devices.length - 1) / 2) * spacing);
  return {
    width: chipX + 110 + 2,
    height: targetsBox.y + targetsBox.h + 4,
    font: 13,
    linnet,
    groups: [
      {
        ...frameworksBox,
        title: "Frameworks",
        note: "imported into Linnet · run from Linnet",
        chips: grid(frameworks, fx + 16, frameworksBox.y + 38, fChipW, 32, 3, 10),
      },
      {
        ...targetsBox,
        title: "Targets",
        note: "compiled and exported from Linnet",
        chips: grid(targets, fx + 16, targetsBox.y + 38, tChipW, 32, 2, 10),
      },
    ],
    accent: [
      { d: `M${linnet.x + linnet.w} ${fy} H${fx}`, both: true },
      {
        d: `M${cx} ${linnet.y + linnet.h} V${ty - 8} Q${cx} ${ty} ${cx + 8} ${ty} H${fx}`,
        both: false,
      },
    ],
    rail: [`M${right} ${ty} H${railX}`, `M${railX} ${centers[0]} V${centers[centers.length - 1]}`],
    arrows: centers.map((c) => `M${railX} ${c} H${chipX}`),
    devices: devices.map((label, i) => ({
      chip: { label, open: label === "…" },
      x: chipX,
      y: centers[i] - 15,
      w: 110,
      h: 30,
    })),
    devicesLabel: { x: chipX, y: centers[0] - 26 },
  };
}

function narrow(): Layout {
  const width = 360;
  const linnet = { x: 2, y: 2, w: 178, h: 64 };
  const trunk = 24;
  const gx = 56;
  const groupW = width - gx - 2;
  const frameworksBox = { x: gx, y: 92, w: groupW, h: 76 };
  const chipH = 30;
  const targetsH = 34 + targets.length * chipH + (targets.length - 1) * 8 + 12;
  const targetsBox = { x: gx, y: 196, w: groupW, h: targetsH };
  const fy = frameworksBox.y + frameworksBox.h / 2;
  const ty = targetsBox.y + targetsBox.h / 2;
  const bottom = targetsBox.y + targetsBox.h;
  const railY = bottom + 30;
  const chipY = railY + 16;
  const dGap = 6;
  const dW = (groupW - (devices.length - 1) * dGap) / devices.length;
  const centers = devices.map((_, i) => gx + i * (dW + dGap) + dW / 2);
  const middle = gx + groupW / 2;
  return {
    width,
    height: chipY + 28 + 4,
    font: 12,
    linnet,
    groups: [
      {
        ...frameworksBox,
        title: "Frameworks",
        note: "into Linnet · out of it",
        chips: grid(frameworks, gx + 12, frameworksBox.y + 34, (groupW - 24 - 16) / 3, chipH, 3, 8),
      },
      {
        ...targetsBox,
        title: "Targets",
        note: "compiled and exported",
        chips: grid(targets, gx + 12, targetsBox.y + 34, groupW - 24, chipH, 1, 8),
      },
    ],
    accent: [
      { d: `M${trunk + 2} ${fy} H${gx}`, both: true },
      { d: `M${trunk} ${linnet.y + linnet.h} V${ty - 8} Q${trunk} ${ty} ${trunk + 8} ${ty} H${gx}`, both: false },
    ],
    rail: [`M${middle} ${bottom} V${railY}`, `M${centers[0]} ${railY} H${centers[centers.length - 1]}`],
    arrows: centers.map((c) => `M${c} ${railY} V${chipY}`),
    devices: devices.map((label, i) => ({
      chip: { label, open: label === "…" },
      x: centers[i] - dW / 2,
      y: chipY,
      w: dW,
      h: 28,
    })),
    devicesLabel: { x: gx, y: bottom + 18 },
  };
}

const id = useId();
const layouts = [
  { name: "wide", layout: wide() },
  { name: "narrow", layout: narrow() },
];
const logo = withBase("/logo.svg");
</script>

<template>
  <section class="arch" aria-labelledby="arch-heading">
    <div class="arch-inner">
      <div class="arch-text">
        <h2 id="arch-heading">Where a Linnet model goes</h2>
        <p>
          Frameworks go both ways: a PyTorch, JAX, or ONNX model imports into Linnet, and a Linnet
          model runs in each. Targets are what the compiler and the exporters write, and the devices
          are where those run.
        </p>
      </div>
      <svg
        v-for="{ name, layout } in layouts"
        :key="name"
        :class="['arch-svg', `arch-${name}`]"
        :viewBox="`0 0 ${layout.width} ${layout.height}`"
        role="img"
        :aria-labelledby="`${id}-${name}-title ${id}-${name}-desc`"
      >
        <title :id="`${id}-${name}-title`">Linnet's interoperability and portability</title>
        <desc :id="`${id}-${name}-desc`">
          Linnet, on the left, connects both ways with frameworks (PyTorch, JAX, ONNX) and outward
          to targets ({{ targets.map((t) => t.label).join(", ") }}), which run on devices ({{
            devices.slice(0, -1).join(", ")
          }}, and others).
        </desc>
        <defs>
          <marker
            v-for="tone in ['accent', 'neutral']"
            :id="`${id}-${name}-${tone}`"
            :key="tone"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="8"
            markerHeight="8"
            markerUnits="userSpaceOnUse"
            orient="auto-start-reverse"
          >
            <path d="M0 0 L8 4 L0 8 z" :class="`arch-head-${tone}`" />
          </marker>
        </defs>

        <!-- Connectors first, so boxes sit on top of their ends. -->
        <path
          v-for="(line, i) in layout.accent"
          :key="`accent-${i}`"
          :d="line.d"
          class="arch-line arch-line-accent"
          :marker-start="line.both ? `url(#${id}-${name}-accent)` : undefined"
          :marker-end="`url(#${id}-${name}-accent)`"
        />
        <path v-for="(d, i) in layout.rail" :key="`rail-${i}`" :d="d" class="arch-line" />
        <path
          v-for="(d, i) in layout.arrows"
          :key="`arrow-${i}`"
          :d="d"
          class="arch-line"
          :marker-end="`url(#${id}-${name}-neutral)`"
        />

        <!-- Linnet -->
        <g class="arch-root">
          <rect
            :x="layout.linnet.x"
            :y="layout.linnet.y"
            :width="layout.linnet.w"
            :height="layout.linnet.h"
            rx="4"
          />
          <image
            :href="logo"
            :x="layout.linnet.x + 14"
            :y="layout.linnet.y + (layout.linnet.h - 36) / 2"
            width="36"
            height="36"
          />
          <text
            class="arch-root-name"
            :x="layout.linnet.x + 60"
            :y="layout.linnet.y + layout.linnet.h / 2 - 7"
          >
            Linnet
          </text>
          <text
            class="arch-root-note"
            :x="layout.linnet.x + 60"
            :y="layout.linnet.y + layout.linnet.h / 2 + 12"
          >
            .linnet source
          </text>
        </g>

        <!-- Frameworks and targets -->
        <g v-for="group in layout.groups" :key="group.title" class="arch-group">
          <rect :x="group.x" :y="group.y" :width="group.w" :height="group.h" rx="4" />
          <text class="arch-group-title" :x="group.x + (name === 'wide' ? 16 : 12)" :y="group.y + 21">
            {{ group.title }}
          </text>
          <text
            class="arch-group-note"
            :x="group.x + group.w - (name === 'wide' ? 16 : 12)"
            :y="group.y + 21"
            text-anchor="end"
          >
            {{ group.note }}
          </text>
          <g
            v-for="placed in group.chips"
            :key="placed.chip.label"
            :class="['arch-chip', { 'arch-chip-open': placed.chip.open }]"
          >
            <rect :x="placed.x" :y="placed.y" :width="placed.w" :height="placed.h" rx="3" />
            <path
              v-if="placed.chip.logo"
              :d="logos[placed.chip.logo]"
              class="arch-logo"
              :transform="`translate(${placed.x + 10} ${placed.y + (placed.h - 16) / 2}) scale(${16 / 24})`"
            />
            <text
              :x="placed.x + (placed.chip.logo ? 32 : 12)"
              :y="placed.y + placed.h / 2"
              :font-size="layout.font"
              dominant-baseline="central"
            >
              {{ placed.chip.label }}
            </text>
          </g>
        </g>

        <!-- Devices -->
        <text class="arch-group-title" :x="layout.devicesLabel.x" :y="layout.devicesLabel.y">
          Devices
        </text>
        <g
          v-for="placed in layout.devices"
          :key="placed.chip.label"
          :class="['arch-chip', 'arch-device', { 'arch-chip-open': placed.chip.open }]"
        >
          <rect :x="placed.x" :y="placed.y" :width="placed.w" :height="placed.h" rx="3" />
          <text
            :x="placed.x + placed.w / 2"
            :y="placed.y + placed.h / 2"
            :font-size="layout.font"
            text-anchor="middle"
            dominant-baseline="central"
          >
            {{ placed.chip.label }}
          </text>
        </g>
      </svg>
    </div>
  </section>
</template>

<style scoped>
.arch {
  padding: 0 24px 48px;
}

/* The hero's column: the same width and edges as the text above. */
.arch-inner {
  max-width: 1152px;
  margin: 0 auto;
}

.arch-text {
  max-width: 720px;
  margin-bottom: 20px;
}

.arch-text h2 {
  margin: 0 0 8px;
  border: none;
  padding: 0;
  font-size: 1.5rem;
  font-weight: 600;
  line-height: 1.3;
  letter-spacing: -0.01em;
  color: var(--vp-c-text-1);
}

.arch-text p {
  margin: 0;
  line-height: 1.6;
  color: var(--vp-c-text-2);
}

.arch-svg {
  display: block;
  width: 100%;
  height: auto;
  font-family: var(--vp-font-family-base);
}

.arch-wide {
  max-width: 1024px;
}

.arch-narrow {
  display: none;
  max-width: 480px;
}

@media (min-width: 640px) {
  .arch {
    padding: 0 48px 56px;
  }
}

@media (min-width: 960px) {
  .arch {
    padding: 0 64px 64px;
  }
}

@media (max-width: 767px) {
  .arch-wide {
    display: none;
  }

  .arch-narrow {
    display: block;
    margin: 0 auto;
  }
}

.arch-line {
  fill: none;
  stroke: var(--vp-c-text-3);
  stroke-width: 1.25;
}

.arch-line-accent {
  stroke: var(--linnet-red);
  stroke-width: 1.75;
}

.arch-head-accent {
  fill: var(--linnet-red);
}

.arch-head-neutral {
  fill: var(--vp-c-text-3);
}

.arch-root rect {
  fill: var(--vp-c-bg-soft);
  stroke: var(--linnet-red);
  stroke-width: 1.5;
}

.arch-root-name {
  font-size: 19px;
  font-weight: 700;
  fill: var(--vp-c-text-1);
}

.arch-root-note {
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  fill: var(--vp-c-text-2);
}

.arch-group > rect {
  fill: var(--vp-c-bg-soft);
  stroke: var(--vp-c-divider);
  stroke-width: 1;
}

.arch-group-title {
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  fill: var(--vp-c-text-2);
}

.arch-group-note {
  font-size: 11px;
  fill: var(--vp-c-text-3);
}

.arch-chip rect {
  fill: var(--vp-c-bg);
  stroke: var(--vp-c-border);
  stroke-width: 1;
}

.arch-chip text {
  fill: var(--vp-c-text-1);
  font-weight: 500;
}

.arch-chip-open rect {
  stroke-dasharray: 4 3;
}

.arch-chip-open text {
  fill: var(--vp-c-text-2);
}

.arch-logo {
  fill: var(--vp-c-text-2);
}

.arch-device text {
  font-family: var(--vp-font-family-mono);
  font-weight: 500;
}
</style>
