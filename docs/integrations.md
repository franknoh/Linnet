# Serving and tools

A Linnet model is a checked source plus a checkpoint, so serving systems
and node editors get it through the same two things every framework does.
This page covers Linnet's own batching engine, Triton Inference Server,
the LLM servers, and ComfyUI; the framework adapters are on their own pages.

## Continuous batching

```python
from linnet import nest
from linnet.serve import Engine, Request

model = nest.load("llama-3.1-8b-instruct", device="cuda",
                  generics={"Batch": 64, "MaxSeq": 2048})
engine = Engine(model)
done, stats = engine.run([Request(prompt=ids, max_new_tokens=256) for ids in prompts])
stats.tokens_per_second, done[0].tokens, done[0].ttft
```

`linnet.serve.Engine` decodes many requests together, each at its own
length: a request joins the batch as soon as a row is free and leaves it
at its token budget, an end-of-sequence token, or the end of the cache. It
drives two entries, which every decoder in the zoo has:

| Entry | Does |
| --- | --- |
| `prefill_slots<M, S>(tokens: [M, S], slots: [M], lengths: [M]) -> [M, Vocab]` | writes `M` prompts into rows `slots` of the KV caches in one pass; each row of `tokens` is padded to one of a few compiled lengths and `lengths` says where each prompt ends |
| `decode_rows(tokens: [Batch, 1], positions: [Batch]) -> [Batch, Vocab]` | one token for every row, each at its own position |
| `prefill_packed<P>(tokens: [P], rows: [P], positions: [P], segments: [P], last: [Batch]) -> [Batch, Vocab]` | optional: several prompts packed end to end into one pass of `P` tokens, each token given its cache row, its position, and which prompt it belongs to; returns the logits after each prompt's last token, `last[m]` |

They are built from `std.nn.cache::write_slots`, `write_rows` (a cache
row written at each sequence's own position) and `write_tokens` (each
token of a packed pass into its own row), and
`std.nn.attention::grouped_attention_rows` (a mask per sequence). The
caches are fixed slots sized by the model's `Batch` and `MaxSeq` generics,
so `Batch` is the most requests in flight and `MaxSeq` the longest prompt
plus completion; there is no paging. The generated PyTorch writes the
caches in place and the step is replayed as a CUDA graph. Waiting prompts
are packed end to end, longest first, into passes of up to `pack` tokens
(4096 by default) when the model has `prefill_packed` -- a pass is padded
only after its last prompt, to one of six compiled sizes, and each size is
compiled and replayed like the step -- and otherwise go through in passes
of up to 8, each padded to its longest; with
`linnet.jax.load_model` (or `nest.load(..., backend="jax_model")`) both
entries are XLA programs over one copy of the weights, with the caches
donated so XLA updates them in place. The tokens a step produces feed the
next step on the device, and the engine queues each step before reading
the previous one's tokens back, so the GPU does not wait on the host
between steps.

Decoding is greedy unless a request sets `temperature`:
`Request(prompt=ids, max_new_tokens=256, temperature=0.8, top_p=0.95,
top_k=50, seed=7)` draws from the softmax of the logits over the
temperature, from the `top_k` most likely tokens and then the fewest whose
probabilities reach `top_p` (the order Transformers filters in), on the
device. A token is drawn as the argmax of the logits plus Gumbel noise
hashed from the request's seed, the token's position, and the token id, so
a seeded request completes the same way alone or in any batch, and on
PyTorch, JAX, or ONNX Runtime alike; without a seed the engine picks one
(`completion.sampling.seed`). Rows that decode greedily cost what they did
before; for Llama 3.1 8B at 64 rows, a temperature costs under 1% of
throughput and top-p, which sorts every row's logits, about 6% (PyTorch)
and 8% (JAX). On ONNX Runtime a sampled step's logits cross to the host,
where NumPy draws.

A server whose requests arrive while it runs drives the same engine step by
step: `engine.submit(request)` queues one and returns its `Completion`, which
fills in as it runs; `engine.step()` admits waiting requests into free rows,
queues one decoding step, reads the one before it, and returns the requests
that finished (a request is seen to finish one step after it does);
`engine.busy` says whether anything is left. `run` is `submit` for every
request, then `step` until nothing is. A request's `on_token` is called
with its completion as each token is read (its `reason` set on the last),
which is how a server streams, and `engine.cancel(completion)` ends a
request early -- at a stop string, or when its client has gone -- freeing
its row for the next step.

### Over HTTP

```bash
python -m linnet.serve llama-3.1-8b-instruct --batch 32 --max-seq 4096 --port 8000
```

serves a Nest decoder with OpenAI's API: `GET /v1/models`,
`POST /v1/completions`, and `POST /v1/chat/completions`, streamed as
server-sent events with `"stream": true` (and `stream_options.include_usage`),
plus `GET /health`. OpenAI's clients work against it as they are:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Name three primary colors."}],
    max_tokens=64,
)
```

The tokenizer and its chat template come from Transformers, from the card's
checkpoint repository unless `--tokenizer` names another, and the
end-of-sequence tokens from it and the checkpoint's generation config.
`--backend jax` or `--backend onnx` serves through those runtimes, and
`--warmup 128 512` compiles those prompt lengths before the first request
(others compile when first seen). Requests take `max_tokens`
(`max_completion_tokens`), `temperature` (1 unless given, as OpenAI has it),
`top_p`, `top_k`, `seed`, and `stop` (up to four strings, cut from the text
and never streamed).

| Field | Does |
| --- | --- |
| `n` | that many choices, each its own request in the batch; with a `seed`, choice `i` draws with `seed + i` |
| `logprobs` (completions), `logprobs` and `top_logprobs` (chat) | each token's log-probability and up to 20 alternatives, in OpenAI's two formats, from the model's logits before temperature and filtering |
| `echo` (completions) | the prompt before the completion's text |

A step computes log probabilities only when a request in it asks, outside
the step's CUDA graph, so serving without them costs nothing extra.
`echo` with `logprobs` is refused: it would need the prompt's own log
probabilities, which a prompt pass does not return.
`linnet.serve.server.Server(engine, tokenizer, name=...)` is the same
server around an engine of one's own, and `Request(logprobs=k)` asks the
engine itself for them.

Work that reads nothing but weights -- dequantizing a quantized
checkpoint, say -- runs once when a model is loaded for inference, not on
every call: `linnet torch --prepare` and `linnet jax --prepare` move it
into a `prepare` function, and the runtime shares each result across the
entries that compute it. A weight seen through views and casts is not
prepared, since that would only copy it; a model loaded with
`trainable=True` keeps the work in the graph, where gradients reach it.

## Triton Inference Server

```bash
python -m linnet.triton export gpt2 -o model_repository --bind B=1 --bind S=64
tritonserver --model-repository=model_repository
```

`linnet.triton.export` writes one model directory the server loads as it
is: `config.pbtxt` with the entry's inputs and outputs (names from the
signature, dtypes, static dims from the bindings) and a version directory
holding the model. `model` is a Nest name or directory (the card supplies
source, generics, checkpoint, and bindings) or a `.linnet` file with
`--weights` and `--bindings`.

| Backend | What is written | Use it for |
| --- | --- | --- |
| `--backend onnx` (default) | `1/model.onnx`, self-contained: `linnet onnx` output with the checkpoint as initializers (external data beyond 1 GB), `platform: "onnxruntime_onnx"` | stateless entries such as `forward` |
| `--backend python` | `1/model.py` plus the source and weights; `backend: "python"`, runs the entry through `linnet.torch` | entries with `state` (KV caches), any dtype, `numerics="fast"` |

Shapes are static (`max_batch_size: 0`); export one model per shape you
serve, or one per batch size. Entry generics (`B`, `S`) come from `--bind`
with a file, or from the card's `[check]` table with a Nest model, and
`--bind` overrides both. The Python backend needs `linnet-lang[torch]` and
the compiler in the server's environment (`LINNET_BIN`, `LINNET_STD`); it
returns `bf16` results as `f32`, which the config declares.

The same packaging is available directly: `linnet.onnx.export_model(source,
generics=..., weights=..., entry=...)` returns the ONNX model with its
inputs and outputs, and `.save(path)`.

```python
from linnet import triton

repository = triton.export("models/tinyllama-1.1b-chat", "model_repository",
                           generics={"S": 128}, backend="python", numerics="fast")
repository.inputs      # (Tensor(name='tokens', dtype='i32', dims=(1, 128)),)
```

## vLLM, SGLang, TGI

```bash
python -m linnet.hf export tinyllama-1.1b-chat -o serve/tinyllama
vllm serve serve/tinyllama
```

The LLM serving stacks implement a fixed set of architectures and load
them from Transformers checkpoint directories. `linnet.hf.export` writes
that directory for a Linnet model whose structure is one of them. It
recognizes the family from the typed program's parameter paths and generics
and from which optional tensors (biases) the checkpoint has:

| Family | Structure |
| --- | --- |
| `llama` | grouped-query attention, RMS norms, SwiGLU; biases on all four attention projections or none, on all three MLP projections or none |
| `qwen2` | Llama's layout with biases on the query, key, and value projections only |
| `qwen3` | an RMS norm over each query and key head, and a head width of its own (`HeadDim`) |
| `phi3` | one projection for query, key, and value, one for gate and up; as many key/value heads as query heads |
| `gpt2` | learned positions, biased LayerNorm, GELU MLP |

`config.json` comes from the generics, the module constants (`THETA`
becomes `rope_theta`, and `FACTOR`, `LOW_FREQ_FACTOR`, `HIGH_FREQ_FACTOR`, and
`ORIGINAL_MAX_POSITION_EMBEDDINGS`, when all four are defined, Llama 3.1's
`llama3` `rope_scaling`), and the epsilon the program's norms actually pass
(a literal at each call, or passed down to it; they must all agree). An
output head bound to the embedding's own tensor is written once, with
`tie_word_embeddings`. The export checks each tensor's shape and dtype
against the program, streams the checkpoint into `model.safetensors` under
the Transformers tensor names, and copies the tokenizer files from the
card's Hub repository, whose `generation_config.json` supplies the
`bos_token_id` and `eos_token_id`. The result is an ordinary checkpoint of
that family, in the layout vLLM, SGLang, TGI, and `transformers` read.
Every layer attends over the whole sequence, as the programs do: Phi-3's
own config slides a 2047-position window, the Nest card does not, and its
export says `sliding_window: null`.

The same command takes a `.linnet` file with `--weights`, `--bind`, and
`--tokenizer <repo>`, so a model trained or modified in Linnet ships the
same way as long as it keeps a known structure. A structure outside the
recognized families is refused with the paths that did not fit; a general
out-of-tree vLLM model class for arbitrary Linnet programs is not part of
this release.

`transformers` agrees with the Linnet interpreter on the exported model to
1e-4 (the tests export tiny GPT-2, Llama, Llama 3.1, Qwen2, Qwen3, and
Phi-3 configurations and compare logits). The Nest cards of Qwen2.5 0.5B,
Qwen3 4B and 8B, and Phi-3 mini export to checkpoints whose tensors are the
Hub checkpoints' own: `transformers` gives bitwise-identical logits for
Qwen3 8B, vLLM 0.30 decodes the same greedy tokens as from the Hub
checkpoints (Qwen3 8B in eager mode; its compiled kernels, built per
checkpoint, round differently at two near-ties), and llama.cpp's converter
writes the same GGUF tensors from either. The export of
`tinyllama-1.1b-chat` from Nest was served by vLLM 0.30 on an H100 as it is:
`/v1/completions` and `/v1/chat/completions` answer with the model's chat
template applied. The benchmarks run the Llama-family exports and GPT-2's
through vLLM, SGLang, and TGI, one request at a time and 256 at once: each
export gives the same first token as the original checkpoint, at the same
speed within run-to-run noise.

## llama.cpp and Ollama

```bash
python -m linnet.gguf export tinyllama-1.1b-chat -o serve/tinyllama \
    --converter ~/llama.cpp/convert_hf_to_gguf.py --outtype q8_0
ollama create tinyllama -f serve/tinyllama/Modelfile
```

llama.cpp runs the same fixed set of architectures from GGUF files, and
its `convert_hf_to_gguf.py` is the one place that knows their tensor layout,
tokenizer metadata, and quantization. `linnet.gguf.export` writes the
Transformers checkpoint as above, runs that converter on it (`--converter`,
or `LLAMA_CPP` pointing at a llama.cpp checkout), and writes an Ollama
`Modelfile` next to the GGUF file, with the chat template translated when
the tokenizer's template is one Ollama has an equivalent for (ChatML,
Zephyr, Llama 2). The family restriction is the same as for vLLM: `llama`,
`qwen2`, `qwen3`, `phi3`, and `gpt2`. A model outside them cannot run in
llama.cpp anyway, and Linnet does not pretend otherwise.

## ComfyUI

[linnet-comfyui](https://github.com/franknoh/linnet-comfyui) is a custom
node pack. Install it into `custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/franknoh/linnet-comfyui
pip install -r linnet-comfyui/requirements.txt
```

| Node | |
| --- | --- |
| Load Linnet model | a `.linnet` file or package, generics as JSON, a SafeTensors path, `numerics`, `compile`, device; outputs a `LINNET_MODEL` |
| Load from Nest | a registry name; downloads the source and checkpoint |
| Run entry | a model, an entry name, up to four `TENSOR` inputs, entry generics as JSON; outputs the results |
| Tensor from list / to string | JSON lists and readable summaries for token ids and small tensors |
| Image to tensor / Tensor to image | ComfyUI `IMAGE` (`[B, H, W, C]` float) to `[B, C, H, W]` in a dtype, and back |
| Argmax | the token ids of logits |
| Model info | `print(model)`: the block tree with names and shapes |
| Reset state | zeroes the model's `state` members between generations |

The nodes are thin: each maps to one call on `linnet.torch` or
`linnet.nest`, and the entry's signature is checked before it runs, so a
wrong shape is an error at the node, not a crash inside a kernel. The
compiler comes from `LINNET_BIN` or `PATH`.
