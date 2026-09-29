<script setup lang="ts">
// The landing hero's two-line summary of a Linnet model: the architecture
// is one file, the weights another.
const files = [
  { name: "model.linnet", role: "architecture" },
  { name: "model.safetensors", role: "weights" },
];
</script>

<template>
  <dl class="hero-files">
    <div v-for="file in files" :key="file.name" class="hero-files-row">
      <dt><code>{{ file.name }}</code></dt>
      <dd>{{ file.role }}</dd>
    </div>
  </dl>
</template>

<style scoped>
.hero-files {
  display: inline-grid;
  gap: 0.35rem;
  margin: 1.5rem 0 0;
  padding: 0.85rem 1.1rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 0.25rem;
  background: var(--vp-c-bg-alt);
  font-family: var(--vp-font-family-mono);
  font-size: 0.95rem;
  line-height: 1.5;
}

.hero-files-row {
  display: grid;
  grid-template-columns: 11.5rem auto;
  gap: 1.25rem;
}

.hero-files dt,
.hero-files dd {
  margin: 0;
}

.hero-files code {
  padding: 0;
  background: none;
  color: var(--vp-c-text-1);
  font-weight: 500;
}

.hero-files dd {
  color: var(--vp-c-text-2);
}

@media (max-width: 480px) {
  .hero-files {
    font-size: 0.85rem;
  }

  .hero-files-row {
    grid-template-columns: 9.5rem auto;
    gap: 0.75rem;
  }
}
</style>
