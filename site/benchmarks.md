# Benchmarks

Linnet compiles one checked source to each framework's own fast path.
Measured on one H100 in `bf16`: 24 real checkpoints from the
[model zoo](https://nest.franknoh.dev) against the stacks people already run
them with, and a synthetic Llama that isolates the generated code.

<ZooClaims part="tiles" />

## Faster than vLLM at batch 1 {#vs-vllm}

<ZooClaims part="matchups" ids="vllm" />

One request at a time, the speed a chat turn or agent step sees, Linnet's
fastest path (generated PyTorch as CUDA graphs, or XLA) decodes faster than
vLLM on the same checkpoint. Portability costs nothing here: the generated
code calls the kernels hand-written code would, and each decoding step runs
as one CUDA graph or XLA program, with the host out of the loop.

## Faster in PyTorch and JAX {#own-framework}

<ZooClaims part="matchups" ids="pytorch,jax,onnx,triton" />

Against the implementation people run on each framework today, Linnet's code
is faster on every model in PyTorch and JAX. The PyTorch gap is widest where
the host sets the pace (small models, single decoding steps, batch-1
encoders), because one CUDA graph removes that overhead. It closes on the
large convolutional models: the SD VAE decoder and SDXL UNet come out even
with `torch.compile`, though Linnet's XLA program decodes the VAE fastest
(7.4 ms). On ONNX Runtime and behind Triton, Linnet is ahead on most models,
but the reference export is faster on MiniLM by 16% and on ResNet-50 by 13%,
and 21% ahead on ResNet-50 behind Triton.

## Exports run as the originals do {#exports}

<ZooClaims part="matchups" ids="export,serve-export" />

A model exported with [`linnet.hf`](/docs/integrations#vllm-sglang-tgi) or
[`linnet.gguf`](/docs/integrations#llama-cpp-and-ollama) runs in the engines
people deploy like the original checkpoint, one request at a time and
serving: the engine cannot tell the difference. The smallest models' serving
pairs vary the most, since their runs last a second or two.

## Serving many requests {#serving}

<ZooClaims part="matchups" ids="serve,serve-triton" />

[`linnet.serve`](/docs/integrations#continuous-batching) serves faster than
vLLM on every decoder. The lead is widest on small models, where a step is
bound by kernel launches that one CUDA graph per step removes. From 4 B up it
is 2-4%, and on gpt-oss 40%. Behind Triton Inference Server, `linnet.serve`
outpaces Triton's own vLLM backend on every decoder.

## Two GPUs and offloading {#behind}

<ZooClaims part="matchups" ids="tp" />

Split across two GPUs at batch 1, the two are within 4%: Linnet is ahead on
Llama 3.1 8B, vLLM on Qwen3 8B. The offloaded rows
run Llama 3.1 8B and Qwen3 8B on a GPU capped at 8 GiB, streaming half the
layers from host memory: Llama peaks at 8.9 GiB and decodes 5.7 tokens per
second, where otherwise it would not load at all. These rows show what a
small GPU can do, so they are never marked best.

## One source, every runtime {#coverage}

<ZooClaims part="coverage" />

Every model in the zoo is one `.linnet` source. The runs that failed:

- TensorRT cannot build an engine for gpt-oss (its Myelin compiler fails
  inside NVRTC) or for the 8 B decoders in `f32`.
- gpt-oss in `f32` is 84 GB of weights, more than the GPU holds, and its ONNX
  serving row runs out of memory.
- transformers' `generate_batch` returns no tokens for gpt-oss, and KerasHub's
  batched gpt-oss fails in XLA's autotuner.

## Training {#training}

The same runs in Linnet and in TRL, step for step. On one GPU, Linnet's
PyTorch step is 1.76 times TRL's for LoRA SFT, 2.4 times for DPO and 2.7
times for GRPO. Fully fine-tuned on four GPUs, the three stacks are within 5%
of each other, with JAX the fastest. Each pair of runs reaches the same
held-out loss, accuracy or reward. [Training](/docs/training#compared-with-trl)
lists the conditions that differ.

<TrainingBench />

## Every measurement

Hover a bar for its notes and how it compares with the stack it replaces.
Each model's page in the zoo has its samples and every number.

### Decoders, one request at a time

<ZooBench part="decoders" />

### Serving

<ZooBench part="serving" />

### Encoders, vision, audio, and diffusion

<ZooBench part="others" />

Models with no KerasHub implementation have no JAX baseline.

### Every row

<ZooBench part="table" />

## The generated code

<BenchChart />

<BenchTable />

A Llama-shaped model with random weights, against a hand-written PyTorch
implementation, shows what the generated code costs on its own. On the medium
model, plain generated PyTorch runs a forward pass in 7.2 ms against 8.6 ms
for the eager reference, and CUDA graphs bring it to 3.8 ms against 3.7 ms
for the compiled reference. The small model's plain generated code is slower
than eager (3.2 ms against 2.7 ms) until compiled (1.5 ms) or replayed as
CUDA graphs (0.8 ms). A decode step is launch-bound: the medium model's
spends most of its 12.7 ms in Python dispatch, so use CUDA graphs in a
decoding loop, which bring it to 3.5 ms.

## How it was measured

- **Hardware:** one H100 80GB (SXM), each method in its own process.
- **Dtype:** `bf16`.
- **Workloads:** one request at a time is a 512-token prompt, then 128 greedy
  tokens. Serving is 256 requests of 128 to 512 prompt tokens, each wanting
  128 new tokens, all waiting from the start, at most 64 in flight. Encoders,
  vision, audio and diffusion run one forward pass at batch 1 (latency) and
  at a large batch (throughput).
- **Baselines:** transformers or diffusers (eager or `torch.compile`,
  whichever is faster), sentence-transformers, KerasHub in JAX, the model's
  own `torch.onnx` export on ONNX Runtime and behind Triton Inference Server,
  vLLM (prefix cache off, since every timed call sends the same prompt), SGLang, Text Generation Inference, and llama.cpp.
- **Linnet's row:** its fastest configuration across generated PyTorch (as
  is, under `torch.compile`, or as CUDA graphs), XLA, ONNX Runtime (CUDA or
  TensorRT), and Triton. Every output is checked against the reference
  stack's.
- **Memory:** peak GPU memory from the driver, or from JAX's allocator for
  JAX rows. The memory views leave out the offloaded rows and every row that
  starts vLLM, which reserves 85% of the GPU for its cache pool.
- **Generated code:** `examples/01-llama` with random weights, small (about
  60 M parameters) and medium (TinyLlama shape, about 1.1 B), timing
  `forward` over `B=1, S=512` and one `decode` step at position 256. The
  variants are the [`linnet.torch.load`](/docs/torch) modes and
  `linnet.jax.load` under `jax.jit`.
- **Training:** Llama 3.1 8B Instruct on H100s, packed into 4096-token
  rows. SFT uses Alpaca, DPO UltraFeedback pairs, and GRPO two-number
  multiplications with an exact-answer reward. The reference is TRL with
  PEFT, FSDP2 for the four-GPU run and vLLM colocated for GRPO. Every stack
  uses the same hyperparameters and step count.
- **Data:** the charts and tables come from `bench/results/zoo.json`,
  `bench/results/latest.json` and `bench/results/training.json`; the prose
  quotes them. The zoo's
  `bench/compare.json` pairs the rows, and its `bench/setup-pod.sh` and
  `bench/run-all.sh` reproduce them on NVIDIA's Triton Inference Server
  image. `python bench/zoo.py <zoo checkout>` refreshes this page's copy.

To rerun the generated-code benchmark (`bench/setup-pod.sh` prepares a fresh
GPU machine; `bench/README.md` lists the published environment):

```bash
cd python/linnet && uv sync --all-extras
cd ../..
LINNET_BIN=build/release/linnet python bench/run.py --device cuda --out bench/results/latest.json
```
