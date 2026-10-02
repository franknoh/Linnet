# Benchmarks

Linnet compiles one checked source to each framework's own fast path. What
that buys, measured: 24 real checkpoints from the
[model zoo](https://nest.franknoh.dev), on one H100 in `bf16`, against the
stacks people already run them with, and a synthetic Llama that isolates what
the generated code itself costs. Nothing on this page is typed in by hand:
the numbers, and the claims above each chart, are computed from
`bench/results/zoo.json` and `bench/results/latest.json`.

<ZooClaims part="tiles" />

## Faster than vLLM, one request at a time {#vs-vllm}

A single request -- a chat turn, an agent's next step -- waits on how fast
one token follows another. For each decoder: Linnet's fastest path on one GPU
(its generated PyTorch replayed as CUDA graphs, or XLA) against vLLM on the
same checkpoint, a 512-token prompt and then 128 tokens chosen greedily,
batch 1.

<ZooClaims part="matchups" ids="vllm" />

Portability costs nothing here. The generated code calls the kernels a
hand-written implementation would -- fused attention, cuBLAS, cuDNN -- and a
whole decoding step replays as one CUDA graph or runs as one XLA program, so
the host is out of the loop between tokens. gpt-oss comes within 2%: its
experts are read as MXFP4 in both, by a Triton kernel of Linnet's own for a
decoding step's few rows.

## Faster than PyTorch's and JAX's own implementations {#own-framework}

Linnet does not replace PyTorch, JAX, or ONNX Runtime; it writes code for
them. The fair baseline on each is the implementation people run there
today: transformers or diffusers in PyTorch (eager or under `torch.compile`,
whichever is faster), KerasHub in JAX, the model's own `torch.onnx` export on
ONNX Runtime, and that export behind Triton Inference Server. In PyTorch and
JAX, Linnet's code is the faster one; on ONNX Runtime the two exports come
out even.

<ZooClaims part="matchups" ids="pytorch,jax,onnx,triton" />

In PyTorch the gap is widest where the host sets the pace -- small models,
single decoding steps, batch-1 encoders -- and one CUDA graph removes it. The
SD VAE decoder is where `torch.compile` stays ahead of Linnet's CUDA graphs;
Linnet's XLA program is faster than both there (7.4 ms). KerasHub keeps the
edge on the two smallest decoders. On ONNX Runtime the two exports land close
together, since ONNX Runtime optimizes both graphs itself; behind Triton the
HTTP front is most of the time for the small models.

## Exports run as the originals do {#exports}

`linnet.hf` writes a transformers checkpoint and `linnet.gguf` a GGUF file,
so a model defined in Linnet runs in the engines people deploy. For an
export the question is sameness, not speed: the same engine on Linnet's
export and on the original checkpoint, one request at a time and serving.

<ZooClaims part="matchups" ids="export,serve-export" />

A model written in Linnet is not held to Linnet's runtimes: its export is a
checkpoint like any other, and the engine cannot tell. The serving pairs of
the smallest models vary the most, since their runs last a second or two.

## Serving many requests {#serving}

256 requests of 128 to 512 prompt tokens, each wanting 128 new tokens, all
waiting from the start, at most 64 in flight. `linnet.serve` batches them
continuously over fixed cache rows -- no paging -- on the same generated
code; it is also what Linnet puts behind Triton Inference Server as a Python
backend.

<ZooClaims part="matchups" ids="serve,serve-triton" />

Up to 4 B parameters Linnet serves faster: a step is bound by launches
there, and one CUDA graph per step removes them, while the prompts go in
packed end to end. From 7 B up the two come within 3% of each other, vLLM's
paged attention and scheduler a little ahead, and gpt-oss -- its experts read
as MXFP4 with OpenAI's `triton_kernels` -- is G_SERVE_RATIO ahead. Behind Triton,
`linnet.serve` outpaces Triton's own vLLM backend on every decoder. The time
to first token is mostly time spent waiting for a free row, for every stack.

## Where Linnet is behind {#behind}

Split across two GPUs, one request at a time, vLLM keeps a small lead:

<ZooClaims part="matchups" ids="tp" />

Each process runs its own shard of the generated code -- its projections
joined, its small all-reduces summed by a one-shot Triton kernel over
symmetric memory, the output head split by vocabulary -- and replays it as
one CUDA graph. The other place Linnet trails is gpt-oss's first token,
G_TTFT ms against vLLM's 9.9: the products of a prompt's experts, a few
thousand rows, are slower than vLLM's.

The offloaded rows run Llama 3.1 8B and Qwen3 8B on a GPU capped at 8 GiB:
half the layers stay on the device and the rest stream in from host memory
as they are needed. Llama peaks at 8.9 GiB and decodes 5.7 tokens per
second, bound by the host link, where otherwise it would not load at all. It
answers a different question -- what a small GPU can do -- so it is never
marked best, and it and every row that starts vLLM (which reserves 85% of
the GPU for its cache pool before it runs) are left out of the memory views.

## One source, every runtime {#coverage}

Every model in the zoo is one `.linnet` source. Where each of Linnet's
runtimes ran it:

<ZooClaims part="coverage" />

The runs that failed:

- TensorRT cannot build an engine for gpt-oss's graph (its Myelin compiler
  fails inside NVRTC), or for the 8 B decoders in `f32`.
- gpt-oss in `f32` is 84 GB of weights, more than the GPU holds, and its
  ONNX serving row runs out of memory in the batched expert contraction.
- transformers' `generate_batch` returns no tokens for gpt-oss, and KerasHub's
  batched gpt-oss fails in XLA's autotuner.

## Every measurement

Each model's rows, grouped by where they run, each runtime's existing stacks
above Linnet's. The toggle switches the measure; hover a bar for its notes
and how it compares with the stack it replaces. Each model's page in the zoo
has the samples it produced and every number.

### Decoders, one request at a time

<ZooBench part="decoders" />

### Serving

<ZooBench part="serving" />

### Encoders, vision, audio, and diffusion

One forward pass at batch 1 (latency) and at a large batch (throughput).

<ZooBench part="others" />

Nothing in JAX loads Phi-3, ModernBERT, DINOv2, SigLIP, ResNet, Whisper, SAM,
or the diffusion models, so those have no KerasHub row.

### Every row

<ZooBench part="table" />

### How the zoo was measured

Every model was measured on one H100 80GB (SXM), in `bf16`, each method in a
process of its own, against the stacks people already run these checkpoints
with: transformers, diffusers, sentence-transformers, vLLM, SGLang, Text
Generation Inference, KerasHub (the maintained JAX implementation of these
architectures), ONNX Runtime on the reference model's own `torch.onnx`
export, llama.cpp, and Triton Inference Server over those exports, over
Linnet's, and in front of vLLM. Linnet runs every way it can: generated
PyTorch as it is, under `torch.compile`, and replayed as CUDA graphs; XLA
from StableHLO and from generated JAX source; ONNX Runtime in `f32`, `f16`,
and `bf16` on its CUDA kernels and on TensorRT; Triton over Linnet's ONNX
export and as a Python backend; and vLLM, SGLang, TGI, and llama.cpp loading
what `linnet.hf` and `linnet.gguf` export.

Every output is checked against the reference stack's, and peak GPU memory is
read from the driver (for JAX rows, from JAX's allocator: the driver sees its
pool, which grows in whole regions). The zoo's `bench/run-all.sh` reproduces
all of it on a fresh GPU machine, after `bench/setup-pod.sh` (on NVIDIA's
Triton Inference Server image); `python bench/zoo.py <zoo checkout>` refreshes
this page's copy. Which rows compare with which is the zoo's
`bench/compare.json`.

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
