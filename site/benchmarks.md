# Benchmarks

What the compiler costs and what it saves, measured. Every row comes from
`bench/run.py`; this page renders the JSON it writes
(`bench/results/latest.json`).

On an H100 the small Llama forward runs in 0.69 ms through XLA and 1.83 ms
as generated PyTorch under `torch.compile` with `numerics="fast"`, against
3.5 ms for eager PyTorch and 1.4 ms for the compiled reference; the medium
model (TinyLlama shape) takes 5.0 ms, 5.0 ms, 9.1 ms, and 4.0 ms
respectively. One decode step through the KV caches takes 0.9 ms and 2.5 ms
in XLA. The `fast` tier (softmax, normalization, and attention in bf16, as
the reference computes them) closes most of the remaining gap to the compiled
reference on the medium model: 6.5 ms to 5.0 ms in PyTorch, 5.6 ms to 5.0 ms
in XLA. Every variant agrees with the reference within bf16 rounding.

<BenchTable />

## Setup

The model is the Llama example (`examples/05-llama`) with random weights:

| Config | H | Heads / KV | Inner | Layers | Vocab | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| small | 512 | 8 / 8 | 1376 | 8 | 32000 | about 60 M |
| medium | 2048 | 32 / 8 | 5632 | 22 | 32000 | about 1.1 B |

Each configuration times `forward` over `B=1, S=512` and one `decode` step
at position 256, in bf16, as:

| Variant | |
| --- | --- |
| PyTorch reference | the hand-written implementation from the test suite, eager |
| PyTorch reference, compiled | the same under `torch.compile` |
| Linnet, interpreted | `load(..., numerics="equivalent")` |
| Linnet, generated source | `load(..., compile=True)` |
| Linnet, generated and compiled | `load(..., compile="inductor")` |
| Linnet, XLA | `linnet.jax.load` under `jax.jit` |
| `numerics=fast` rows | the same, with softmax, normalization, and attention in bf16 rather than f32, as the reference computes them |

Latency is the median of timed calls after warm-up with the device
synchronized; throughput is tokens per second for `forward` and steps per
second for `decode`. Outputs are compared against the eager reference and the
largest difference is shown. Compile and load times are listed separately.

## Reading the table

The interpreted path pays a Python dispatch per Core IR operation on every
call; the arithmetic is the same kernels, so the gap shrinks as the model
grows. Generated source removes that overhead and gives `torch.compile` a
whole function to trace. The XLA path is one compiled program with static
shapes and fused elementwise chains.

## Reproducing

```bash
cd python/linnet && uv sync --all-extras
cd ../..
LINNET_BIN=build/release/linnet python bench/run.py --device cuda --out bench/results/latest.json
```

`bench/README.md` lists the configurations and the environment the published
numbers came from; `bench/setup-pod.sh` prepares a fresh GPU machine.
