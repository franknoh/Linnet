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
Runtime on the reference model's own `torch.onnx` export, llama.cpp, and
Triton Inference Server over those exports, over Linnet's, and in front of
vLLM. Linnet runs every way it can:

- generated PyTorch, as it is, under `torch.compile` (inductor), and
  replayed as CUDA graphs;
- XLA through `linnet.jax`, from StableHLO and from generated JAX source;
- ONNX Runtime in `f32`, `f16`, and `bf16`, on its CUDA kernels and on
  TensorRT;
- Triton Inference Server, over Linnet's ONNX export and as a Python backend;
- vLLM and llama.cpp, loading what `linnet.hf` and `linnet.gguf` export.

Every output is checked against the reference stack's, and peak GPU memory is
read from the driver (for JAX rows, from JAX's allocator: the driver sees its
pool, which grows in whole regions). Each model's page in the zoo has the
samples it produced and every number.

### Decoders, one request at a time

A 512-token prompt, then 128 tokens greedily, batch 1. Linnet decodes faster
than vLLM on every decoder but gpt-oss (below). As CUDA graphs, each step
replayed as one graph with sibling projections joined into one product: 168
against 157 tokens per second for Llama 3.1 8B, 176 against 165 for Mistral
7B, 159 against 153 for Qwen3 8B, 916 against 511 for Qwen2.5 0.5B, 1965
against 840 for GPT-2. Under XLA the larger models are as fast or faster
(170 for Llama 3.1 8B, 298 against 250 for Phi-3 mini), and the CUDA graphs
have the shortest first token on the small models (0.8 ms for GPT-2, 2.6 ms
for TinyLlama, against vLLM's 7.7 and 7.6).

A model Linnet exports runs on its target as fast as that target's own
conversion: Llama 3.1 8B decodes at 157 tokens per second in vLLM from
`linnet.hf` (157 from the original checkpoint), 158 in SGLang (158), 115 in
Text Generation Inference (112), and 171 in llama.cpp from `linnet.gguf`,
and the same holds for the Qwen2.5, Qwen3, and Phi-3 exports (153 against
153 for Qwen3 8B in vLLM). Every export gives the same first token as the
original. The ONNX rows copy the logits to the host for every step's argmax,
which holds them to 47 to 75 tokens per second on the 7 and 8 B models.

<ZooBench part="decoders" />

gpt-oss 20B stores its experts in MXFP4. Linnet unpacks them once, when the
model loads (`--prepare`), and each step then reads only the four experts a
token routes to -- the down projection as one grouped matrix product
(`std.nn.moe::linear_experts`) on CUDA: 299 tokens per second from generated
JAX source and 203 as CUDA graphs, against vLLM's 303 (fused MXFP4 kernels,
a quarter of the bytes) and transformers' 45. On ONNX Runtime it runs at 20.

On two GPUs, four ways of splitting a model (decode tokens per second):

| | Llama 3.1 8B | Qwen3 8B |
| --- | --- | --- |
| vLLM, tensor parallel | 247 | 229 |
| Linnet PyTorch (DTensor), tensor parallel | 180 | 151 |
| Linnet XLA, tensor parallel | 89 | 69 |
| Linnet, layers split across the two | 88 | 68 |

Linnet's PyTorch step on two GPUs is captured as one CUDA graph per process,
its all-reduces inside. vLLM stays ahead with its own one-shot all-reduce
and projections it joins on each GPU, which a split held as DTensors
cannot.

The offloaded rows run Llama 3.1 8B and Qwen3 8B on a GPU capped at 8 GiB:
half the layers stay on the device and the rest stream in from host memory
as they are needed. Llama peaks at 8.9 GiB and decodes 5.7 tokens per
second, bound by the host link, where otherwise it would not load at all. It
answers a different question from the other rows -- what a small GPU can do
-- so it is never marked best. It and every row that starts vLLM (which
reserves 85% of the GPU for its cache pool before it runs) are left out of
the memory view; all of them are in the table.

### Serving: many requests at once

256 requests of 128 to 512 prompt tokens, each wanting 128 new tokens, all
waiting from the start, at most 64 in flight. vLLM's paged attention and
scheduler lead from 1 B up: 5658 tokens per second for Llama 3.1 8B against
4858 for Linnet's own continuous batching (`linnet.serve`) as CUDA graphs and
4734 under XLA. Below that Linnet is ahead (38320 against 25858 on GPT-2,
29020 against 18830 on Qwen2.5 0.5B). A Linnet model served by vLLM through
`linnet.hf` matches vLLM (5607), Triton over `linnet.serve` adds its HTTP
front (4484), and Linnet is ahead of Triton's own vLLM backend (2998) on
every model but gpt-oss, and of transformers' `generate_batch` and KerasHub's
static batches on all of them.

<ZooBench part="serving" />

`linnet.serve` keeps each request in a fixed row of a cache compiled for the
longest prompt plus completion, with no paging, and every decoding step runs
all 64 rows; the time to first token is mostly time spent waiting for a free
row, for every stack. On ONNX Runtime the argmax is taken in the graph and
each step replays as a CUDA graph: 3558 tokens per second for Llama 3.1 8B.

### Encoders, vision, audio, and diffusion

One forward pass at batch 1 (latency) and at a large batch (throughput).
Linnet's CUDA graphs are the fastest way to run the text encoders at batch 1
(0.74 ms for BERT against 3.60 eager and 1.67 under `torch.compile`), and
TensorRT over Linnet's `f16` ONNX export is faster still (0.53 ms), as it is
for every convolutional model (0.21 ms for ResNet-18). XLA leads where one
program replaces many large kernels: SAM's image encoder takes 14 ms against
39 eager, and the SD VAE decoder 7.4 against 22 (5.4 on TensorRT over the
`f16` export). Whisper transcribes over its KV caches: large-v3 in 67 ms as
CUDA graphs, against 307 eager and 192 under `torch.compile`. SDXL's UNet
step is where `torch.compile` and Linnet's CUDA graphs tie (35 ms). KerasHub
is slower than eager on every encoder it loads, and Triton adds its HTTP
front to the rows it serves.

<ZooBench part="others" />

Nothing in JAX loads Phi-3, ModernBERT, DINOv2, SigLIP, ResNet, Whisper, SAM,
or the diffusion models, so those have no KerasHub row.

### What did not run

Nine rows of 528 failed, and are shown as such:

- TensorRT cannot build an engine for gpt-oss's graph (its Myelin compiler
  fails inside NVRTC), or for the 8 B decoders in `f32`.
- gpt-oss in `f32` is 84 GB of weights, more than the GPU holds, and its
  ONNX serving row runs out of memory in the batched expert contraction.
- transformers' `generate_batch` returns no tokens for gpt-oss, and KerasHub's
  batched gpt-oss fails in XLA's autotuner.

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
