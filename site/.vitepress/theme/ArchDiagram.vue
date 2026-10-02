<script setup lang="ts">
// The landing page's hero image: where a Linnet model goes. Linnet is the
// root on the left; two groups hang off it. Frameworks connect both ways (a
// PyTorch, JAX, or ONNX model imports into Linnet, and a Linnet model runs in
// each); targets are what the compiler and exporters write, and from them a
// rail runs to the devices. One layout fills the column beside the hero's
// text; a narrower one, with the frameworks under Linnet, fits a phone.
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
  name: string;
  width: number;
  height: number;
  font: number; // chip labels
  icon: number; // chip logos, square
  pad: number; // group and chip insets
  linnet: Box & { logo: number; font: number };
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

// The targets' chips in one column from (x, y), `w` wide, and the devices in
// a column ending at `right`, fed by one rail; both layouts share it.
function targetsAndDevices(
  x: number,
  y: number,
  w: number,
  chipH: number,
  gap: number,
  pad: number,
  deviceW: number,
  spacing: number,
  right: number,
) {
  const h = 34 + targets.length * chipH + (targets.length - 1) * gap + 12;
  const ty = y + h / 2;
  const chipX = right - deviceW;
  const railX = (x + w + chipX) / 2;
  const centers = devices.map((_, i) => ty + (i - (devices.length - 1) / 2) * spacing);
  return {
    box: { x, y, w, h },
    ty,
    chips: grid(targets, x + pad, y + 34, w - 2 * pad, chipH, 1, gap),
    rail: [`M${x + w} ${ty} H${railX}`, `M${railX} ${centers[0]} V${centers[centers.length - 1]}`],
    arrows: centers.map((c) => `M${railX} ${c} H${chipX}`),
    devices: devices.map((label, i) => ({
      chip: { label, open: label === "…" },
      x: chipX,
      y: centers[i] - chipH / 2,
      w: deviceW,
      h: chipH,
    })),
    devicesLabel: { x: chipX, y: centers[0] - chipH / 2 - 9 },
  };
}

// Beside the hero's text: Linnet and the frameworks side by side, the
// targets below the frameworks, the devices beside the targets.
function hero(): Layout {
  const width = 500;
  const pad = 12;
  const linnet = { x: 1, y: 1, w: 150, h: 72, logo: 30, font: 18 };
  const gx = 182;
  const frameworksBox = { x: gx, y: 1, w: width - 1 - gx, h: 72 };
  const fChipW = (frameworksBox.w - 2 * pad - 2 * 8) / 3;
  const fy = frameworksBox.y + frameworksBox.h / 2;
  const cx = linnet.x + linnet.w / 2;
  const rest = targetsAndDevices(gx, 105, 208, 30, 7, pad, 75, 46, width - 1);
  return {
    name: "hero",
    width,
    height: rest.box.y + rest.box.h + 2,
    font: 13,
    icon: 16,
    pad,
    linnet,
    groups: [
      {
        ...frameworksBox,
        title: "Frameworks",
        note: "imported into Linnet · run from it",
        chips: grid(frameworks, gx + pad, frameworksBox.y + 30, fChipW, 30, 3, 8),
      },
      { ...rest.box, title: "Targets", note: "compiled and exported", chips: rest.chips },
    ],
    accent: [
      { d: `M${linnet.x + linnet.w} ${fy} H${gx}`, both: true },
      {
        d: `M${cx} ${linnet.y + linnet.h} V${rest.ty - 8} Q${cx} ${rest.ty} ${cx + 8} ${rest.ty} H${gx}`,
        both: false,
      },
    ],
    rail: rest.rail,
    arrows: rest.arrows,
    devices: rest.devices,
    devicesLabel: rest.devicesLabel,
  };
}

// A phone: Linnet on top, a trunk down its left, the frameworks and the
// targets branching off it.
function phone(): Layout {
  const width = 340;
  const pad = 10;
  const linnet = { x: 1, y: 1, w: 150, h: 56, logo: 28, font: 17 };
  const trunk = 24;
  const gx = 48;
  const frameworksBox = { x: gx, y: 80, w: width - 1 - gx, h: 68 };
  const fChipW = (frameworksBox.w - 2 * pad - 2 * 6) / 3;
  const fy = frameworksBox.y + frameworksBox.h / 2;
  const rest = targetsAndDevices(gx, 172, 196, 26, 6, pad, 61, 40, width - 1);
  return {
    name: "phone",
    width,
    height: rest.box.y + rest.box.h + 2,
    font: 12,
    icon: 14,
    pad,
    linnet,
    groups: [
      {
        ...frameworksBox,
        title: "Frameworks",
        note: "into Linnet · out of it",
        chips: grid(frameworks, gx + pad, frameworksBox.y + 30, fChipW, 26, 3, 6),
      },
      { ...rest.box, title: "Targets", note: "compiled · exported", chips: rest.chips },
    ],
    accent: [
      { d: `M${trunk + 2} ${fy} H${gx}`, both: true },
      {
        d: `M${trunk} ${linnet.y + linnet.h} V${rest.ty - 8} Q${trunk} ${rest.ty} ${trunk + 8} ${rest.ty} H${gx}`,
        both: false,
      },
    ],
    rail: rest.rail,
    arrows: rest.arrows,
    devices: rest.devices,
    devicesLabel: rest.devicesLabel,
  };
}

const id = useId();
const layouts = [hero(), phone()];
const logo = withBase("/logo.svg");
</script>

<template>
  <div class="arch">
    <svg
      v-for="layout in layouts"
      :key="layout.name"
      :class="['arch-svg', `arch-${layout.name}`]"
      :viewBox="`0 0 ${layout.width} ${layout.height}`"
      role="img"
      :aria-labelledby="`${id}-${layout.name}-title ${id}-${layout.name}-desc`"
    >
      <title :id="`${id}-${layout.name}-title`">Where a Linnet model goes</title>
      <desc :id="`${id}-${layout.name}-desc`">
        Linnet, on the left, connects both ways with frameworks (PyTorch, JAX, ONNX) and outward
        to targets ({{ targets.map((t) => t.label).join(", ") }}), which run on devices ({{
          devices.slice(0, -1).join(", ")
        }}, and others).
      </desc>
      <defs>
        <marker
          v-for="tone in ['accent', 'neutral']"
          :id="`${id}-${layout.name}-${tone}`"
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
        :marker-start="line.both ? `url(#${id}-${layout.name}-accent)` : undefined"
        :marker-end="`url(#${id}-${layout.name}-accent)`"
      />
      <path v-for="(d, i) in layout.rail" :key="`rail-${i}`" :d="d" class="arch-line" />
      <path
        v-for="(d, i) in layout.arrows"
        :key="`arrow-${i}`"
        :d="d"
        class="arch-line"
        :marker-end="`url(#${id}-${layout.name}-neutral)`"
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
          :x="layout.linnet.x + layout.pad"
          :y="layout.linnet.y + (layout.linnet.h - layout.linnet.logo) / 2"
          :width="layout.linnet.logo"
          :height="layout.linnet.logo"
        />
        <text
          class="arch-root-name"
          :x="layout.linnet.x + 2 * layout.pad + layout.linnet.logo"
          :y="layout.linnet.y + layout.linnet.h / 2 - 6"
          :font-size="layout.linnet.font"
        >
          Linnet
        </text>
        <text
          class="arch-root-note"
          :x="layout.linnet.x + 2 * layout.pad + layout.linnet.logo"
          :y="layout.linnet.y + layout.linnet.h / 2 + 12"
        >
          .linnet source
        </text>
      </g>

      <!-- Frameworks and targets -->
      <g v-for="group in layout.groups" :key="group.title" class="arch-group">
        <rect :x="group.x" :y="group.y" :width="group.w" :height="group.h" rx="4" />
        <text class="arch-group-title" :x="group.x + layout.pad" :y="group.y + 20">
          {{ group.title }}
        </text>
        <text
          class="arch-group-note"
          :x="group.x + group.w - layout.pad"
          :y="group.y + 20"
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
            :transform="`translate(${placed.x + layout.pad - 2} ${placed.y + (placed.h - layout.icon) / 2}) scale(${layout.icon / 24})`"
          />
          <text
            :x="placed.x + (placed.chip.logo ? layout.pad + layout.icon + 4 : layout.pad)"
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
</template>

<style scoped>
.arch {
  width: 100%;
}

.arch-svg {
  display: block;
  width: 100%;
  height: auto;
  margin: 0 auto;
  font-family: var(--vp-font-family-base);
}

.arch-hero {
  max-width: 520px;
}

.arch-phone {
  display: none;
  max-width: 380px;
}

@media (max-width: 639px) {
  .arch-hero {
    display: none;
  }

  .arch-phone {
    display: block;
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
  font-weight: 700;
  fill: var(--vp-c-text-1);
}

.arch-root-note {
  font-family: var(--vp-font-family-mono);
  font-size: 10px;
  fill: var(--vp-c-text-2);
}

.arch-group > rect {
  fill: var(--vp-c-bg-soft);
  stroke: var(--vp-c-divider);
  stroke-width: 1;
}

.arch-group-title {
  font-family: var(--vp-font-family-mono);
  font-size: 10.5px;
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
}
</style>
