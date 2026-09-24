# Benchmarks

What the compiler costs and what it saves, measured. Every row comes from
`bench/run.py`; this page renders the JSON it writes
(`bench/results/latest.json`).

On an H100 the medium model (TinyLlama shape, `bf16`) runs a forward pass in
7.2 ms as plain generated PyTorch — ahead of the 8.6 ms the hand-written
eager reference takes — 4.2 ms under `torch.compile`, and 3.8 ms replayed as
CUDA graphs, against 3.7 ms for the compiled reference. The small model takes
3.2 ms, 1.5 ms, and 0.8 ms against 2.7 ms eager and 1.3 ms compiled. Every
variant agrees with the reference within bf16 rounding.

Four changes account for most of it: the generated code calls the kernels the
reference calls (`F.embedding`, `scaled_dot_product_attention` with
`is_causal` and `enable_gqa`, `index_copy` for a KV cache position), it
computes input-independent values such as rotary tables once per shape rather
than once per layer per call, `torch.softmax` and `torch.rms_norm` are called
without the f32 casts their kernels make redundant, and a whole step can be
replayed as one CUDA graph.

A single decode step is the exception: it is launch-bound, not
arithmetic-bound. One layer issues 62 kernels for 0.15 ms of GPU work, so the
22-layer step spends most of its 12.7 ms in Python dispatch. `torch.compile`
halves that and CUDA graphs bring it to 3.5 ms, which is what a decoding loop
should use.

<BenchChart />

Bars are speed-ups over eager PyTorch (the dashed line is 1×); the toggle
shows latency instead. The table below has every row: latency, throughput,
speed-up, and the largest difference from the reference.

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
| Linnet, interpreted | `load(..., compile=False)` |
| Linnet, generated source | `load(..., compile=True)`, the default on CUDA |
| Linnet, generated and compiled | `load(..., compile="inductor")` |
| Linnet, CUDA graphs | `load(..., compile="reduce-overhead")`, `numerics="fast"` |
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
