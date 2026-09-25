# Benchmarks

What the compiler costs and what it saves, measured. Two sets of numbers:
24 real checkpoints from the [model zoo](https://nest.franknoh.dev), run
against the stacks people already serve them with, and a synthetic Llama that
isolates what the generated code itself costs. Nothing on this page is typed
in by hand: the charts render `bench/results/zoo.json` and
`bench/results/latest.json`.

## Real models

Every model in the zoo was measured on one H100 80GB (SXM) in `bf16` at
batch 1, each method in a process of its own: transformers, diffusers,
sentence-transformers, and vLLM beside Linnet's generated PyTorch, the same
source replayed as CUDA graphs, XLA through `linnet.jax`, and ONNX Runtime
through `linnet.onnx`. Decoders read a 512-token prompt and generate 128
tokens greedily; the rest time one forward pass. Every Linnet output is
checked against the reference stack's, and peak GPU memory is sampled from the
driver. Each model's page in the zoo has the samples it produced and the full
numbers.

### Decoders

From 1.7 B parameters up, Linnet under XLA decodes as fast as vLLM or faster:
165 against 156 tokens per second for Llama 3.1 8B, 172 against 165 for
Mistral 7B, 158 against 152 for Qwen3 8B, 268 against 250 for Phi-3 mini,
where transformers decodes Llama at 114 compiled and 70 eager. Below that vLLM wins
decisively (607 against 316 for Qwen2.5 0.5B): a step that small is bound by
launching work, not by the arithmetic, and vLLM's runtime is built for
exactly that. The CUDA graphs path has the shortest first token on small
models (4.6 ms for Qwen2.5 0.5B, 11.7 ms under vLLM) and matches it at 8 B.

<ZooBench part="decoders" />

The XLA path is not free. Its first token is three to four times slower than
the PyTorch paths', and it currently holds the weights twice, once for the
prompt and once for decoding, which is what its memory bars show. The memory
view leaves out two rows whose number is a setting rather than a need: vLLM
reserves 85% of the GPU for its KV-cache pool before it runs, and the
offloaded row is held under an 8 GiB cap on purpose; both are in the table. gpt-oss 20B's card has no KV-cache entries yet, so only
its first token is timed.

The offloaded rows run Llama 3.1 8B and Qwen3 8B on a GPU capped at 8 GiB:
half the layers stay on the device and the rest stream in from host memory
as they are needed (`device_map="auto", max_memory=...`). Llama peaks at
9.1 GiB and decodes 5.7 tokens per second, bound by the host link, where
otherwise it would not load at all. It answers a different question from the
other rows -- what a small GPU can do -- so it is never marked best.

### Encoders, vision, audio, and diffusion

Bars are raw numbers; in every chart the best one is red (the shortest
where lower is better, the longest where higher is). XLA gives the largest gains on encoders, where one compiled program replaces a
long chain of small kernels: 5.2× for BERT, 6.7× for ModernBERT, 3.0× for
the Stable Diffusion VAE decoder, 2.8× for SAM's image encoder. On the
larger convolutional and transformer blocks the paths come close to
`torch.compile` rather than ahead of it (SDXL's UNet: 1.22× against 1.36×),
and Whisper large's encoder is slightly slower than eager.

<ZooBench part="others" />

Two ONNX rows failed and are shown as such: the SD VAE's export trips
ONNX Runtime's CUDA `Concat`, and Whisper large is past the 2 GB a single
ONNX file can hold without external data.

### Every row

<ZooBench part="table" />

The zoo's `bench/run-all.sh` reproduces all of it on a fresh GPU machine,
after `bench/setup-pod.sh`; `python bench/zoo.py <zoo checkout>` refreshes
this page's copy.

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
