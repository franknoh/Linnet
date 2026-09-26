# Benchmarks

What the compiler costs and what it saves, measured. Two sets of numbers:
24 real checkpoints from the [model zoo](https://nest.franknoh.dev), run
against the stacks people already serve them with, and a synthetic Llama that
isolates what the generated code itself costs. Nothing on this page is typed
in by hand: the charts render `bench/results/zoo.json` and
`bench/results/latest.json`.

## Real models

Every model in the zoo was measured on one H100 80GB (SXM), in `bf16`, each
method in a process of its own, against the stacks people already run these
checkpoints with: transformers, diffusers, sentence-transformers, vLLM,
KerasHub (the maintained JAX implementation of these architectures), ONNX
Runtime on the reference model's own `torch.onnx` export, and Triton
Inference Server over that export, over Linnet's, and in front of vLLM.
Linnet runs as generated PyTorch, as the same source replayed as CUDA
graphs, as XLA through `linnet.jax`, and as ONNX. Every output is checked
against the reference stack's, and peak GPU memory is read from the driver
(for JAX rows, from JAX's allocator: the driver sees its pool, which grows
in whole regions). Each model's page in the zoo has the samples it produced
and every number.

### Decoders, one request at a time

A 512-token prompt, then 128 tokens greedily, batch 1. From 1.7 B
parameters up, Linnet under XLA decodes as fast as vLLM or a little faster:
170 against 158 tokens per second for Llama 3.1 8B, 178 against 166 for
Mistral 7B, 161 against 153 for Qwen3 8B, 296 against 249 for Phi-3 mini.
KerasHub on JAX lands between the two (159 for Llama). On the smallest
models KerasHub is ahead of everyone (1399 tokens per second for GPT-2).
vLLM has the shortest first token from 4 B up; below that Linnet's CUDA
graphs do (1.8 ms for GPT-2, 4.0 ms for Qwen2.5 0.5B).

<ZooBench part="decoders" />

The offloaded rows run Llama 3.1 8B and Qwen3 8B on a GPU capped at 8 GiB:
half the layers stay on the device and the rest stream in from host memory
as they are needed (`device_map="auto", max_memory=...`). Llama peaks at
8.9 GiB and decodes 4.4 tokens per second, bound by the host link, where
otherwise it would not load at all. It answers a different question from the
other rows -- what a small GPU can do -- so it is never marked best, and it
and vLLM (which reserves 85% of the GPU for its cache pool before it runs)
are left out of the memory view; both are in the table.

gpt-oss 20B is where Linnet is furthest behind: 54 tokens per second under
XLA and 7 as CUDA graphs, against vLLM's 303. Its experts are stored in
MXFP4 and Linnet dequantizes them in the graph on every step (the chosen
four when decoding one request, all of them for a batch); vLLM runs fused
MXFP4 kernels.

### Serving: many requests at once

256 requests of 128 to 512 prompt tokens, each wanting 128 new tokens, all
waiting from the start, at most 64 in flight. This is where a server spends
its GPU, and where vLLM's paged attention and scheduler are well ahead:
5697 tokens per second for Llama 3.1 8B against 3227 for Linnet's own
continuous batching (`linnet.serve`) under XLA and 2967 as CUDA graphs,
and 20241 against 6183 for TinyLlama. Linnet is ahead of Triton Inference
Server's vLLM backend on the smallest models (it adds HTTP and a Python
backend in front of vLLM; 3769 for Llama), and three to four times ahead of
transformers' own continuous batching (`generate_batch`, 840) and about eight of
KerasHub, which has only static batches (391).

<ZooBench part="serving" />

`linnet.serve` keeps each request in a fixed row of a cache compiled for the
longest prompt plus completion, with no paging, and every decoding step runs
all 64 rows; the time to first token is mostly time spent waiting for a
free row, for every stack. Memory here is again a setting for vLLM and for
transformers' paged pool, so the memory view shows the others.

### Encoders, vision, audio, and diffusion

One forward pass at batch 1 (latency) and at a large batch (throughput).
XLA gives the largest gains on the text encoders, where one compiled program
replaces a long chain of small kernels: 0.85 ms for BERT against 3.65 eager
and 2.18 for ONNX Runtime, 0.75 for RoBERTa. The SD VAE decoder takes 7.2 ms
against 22.4, SAM's image encoder 14.0 against 39.5. The convolutional
models and SDXL's UNet are where `torch.compile` or ONNX Runtime win, and
Whisper large's encoder is slower under XLA than eager. KerasHub is slower
than eager on every encoder it loads. The ONNX rows run in f32, the
checkpoints' own precision, against `bf16` for the rest; Triton adds HTTP to
them.

<ZooBench part="others" />

Two ONNX rows failed and are shown as such: the SD VAE's export trips
ONNX Runtime's CUDA `Concat`, and Whisper large is past the 2 GB a single
ONNX file can hold without external data. Nothing in JAX loads Phi-3,
ModernBERT, DINOv2, SigLIP, ResNet, Whisper, SAM, or the diffusion models,
so those have no KerasHub row.

### Every row

<ZooBench part="table" />

The zoo's `bench/run-all.sh` reproduces all of it on a fresh GPU machine,
after `bench/setup-pod.sh` (on NVIDIA's Triton Inference Server image);
`python bench/zoo.py <zoo checkout>` refreshes this page's copy.

## The generated code

A Llama-shaped model with random weights (`examples/05-llama`), timed by
`bench/run.py` against a hand-written PyTorch implementation of the same
architecture, shows what the generated code costs on its own.

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

### Setup

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

### Reading the table

The interpreted path pays a Python dispatch per Core IR operation on every
call; the arithmetic is the same kernels, so the gap shrinks as the model
grows. Generated source removes that overhead and gives `torch.compile` a
whole function to trace. The XLA path is one compiled program with static
shapes and fused elementwise chains.

### Reproducing

```bash
cd python/linnet && uv sync --all-extras
cd ../..
LINNET_BIN=build/release/linnet python bench/run.py --device cuda --out bench/results/latest.json
```

`bench/README.md` lists the configurations and the environment the published
numbers came from; `bench/setup-pod.sh` prepares a fresh GPU machine.
