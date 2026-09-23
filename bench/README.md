# Benchmarks

`run.py` times the Linnet backends against a hand-written PyTorch
implementation of the same architecture and writes `results/latest.json`,
which the documentation site renders on its Benchmarks page. Nothing on that
page is typed in by hand.

## What runs

The model is the repository's Llama-style decoder (`examples/05-llama`) with
random weights, at these widths:

| Config | H | Heads / KV | Inner | Layers | Vocab | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` | 64 | 4 / 2 | 128 | 2 | 256 | smoke test |
| `small` | 512 | 8 / 8 | 1376 | 8 | 32000 | ≈ 60 M |
| `medium` | 2048 | 32 / 8 | 5632 | 22 | 32000 | ≈ 1.1 B (TinyLlama) |
| `large` | 4096 | 32 / 8 | 14336 | 32 | 32000 | ≈ 8 B (Llama-3-8B shape) |

For each: the `forward` entry over `B=1, S=512` and the `decode` entry (one
token through the KV caches at position `S/2`), in four variants — PyTorch
eager, `torch.compile`, Linnet materialized in PyTorch with native kernels
(`numerics="equivalent"`), and Linnet compiled through StableHLO and run by
XLA (`linnet_jax`). Every variant's output is compared against the eager
reference; the table shows the maximum absolute difference.

Latency is the median of `--iters` timed calls after `--warmup` calls, with
the device synchronized around each call. `bf16` on CUDA, `f32` on CPU unless
`--dtype` says otherwise.

## Running

An environment with `linnet_torch` and (for the XLA rows) `linnet_jax`
installed, a `jax` that sees the same accelerator as `torch`, and the
compiler on `PATH` or in `LINNET_BIN`:

```bash
uv venv .bench && source .bench/bin/activate
uv pip install -e python/linnet_torch -e python/linnet_jax torch "jax[cuda12]"
LINNET_BIN=build/release/linnet python bench/run.py --device cuda --configs small,medium
```

`--no-xla` skips the JAX rows; `--configs tiny --device cpu` is a quick
check of the harness itself.

## Published numbers

`results/latest.json` records the date, device, and library versions it was
measured with. The published run was made on a RunPod cloud GPU from a clean
checkout; see the `environment` field.
