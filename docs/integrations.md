# Serving and tools

Serve a Linnet model with Linnet's own batching engine, or export it to Triton
Inference Server, vLLM, SGLang, TGI, llama.cpp, Ollama, or ComfyUI. The
framework adapters have their own pages: [PyTorch](torch.md), [JAX](jax.md),
[ONNX](onnx.md).

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

`linnet.serve.Engine` decodes many requests together, each at its own length.
A request takes a free cache row and leaves at its token budget, an
end-of-sequence token, or the end of the cache. The model's `Batch` generic
sets the most requests in flight, and `MaxSeq` the longest prompt plus
completion. With `linnet.torch` the step replays as a CUDA graph; with
`linnet.jax.load_model` (`nest.load(..., backend="jax_model")`) it is an XLA
program.

### Entries

Every decoder in the zoo has the two required entries.

| Entry | Does |
| --- | --- |
| `prefill_slots<M, S>(tokens: [M, S], slots: [M], lengths: [M]) -> [M, Vocab]` | writes `M` padded prompts into cache rows `slots` in one pass; `lengths` says where each ends |
| `decode_rows(tokens: [Batch, 1], positions: [Batch]) -> [Batch, Vocab]` | one token for every row, each at its own position |
| `prefill_packed<P>(tokens: [P], rows: [P], positions: [P], segments: [P], last: [Batch]) -> [Batch, Vocab]` | optional: prompts packed end to end into one pass of `P` tokens, each token with its cache row, position, and prompt; returns the logits after each prompt's last token, `last[m]` |
| `step_packed<P>(tokens, rows, positions, segments, last, step_tokens: [Batch, 1], step_positions: [Batch]) -> [2 * Batch, Vocab]` | optional, with `prefill_packed`: that pass and a `decode_rows` step in one; returns the logits after each prompt, then each row's step |
| `prefill_paged<P, Rows>(tokens: [P], positions: [P], segments: [P], slots: [P], last: [Rows]) -> [Rows, Vocab]` | for pages: `prefill_packed` with each token's place in the pool instead of its row |
| `decode_paged<Rows, Pages>(tokens: [Rows, 1], positions: [Rows], table: [Rows, Pages]) -> [Rows, Vocab]` | for pages: one token for every row, its positions in the pages its row of `table` lists |
| `step_paged<P, Rows, Pages>(tokens, positions, segments, slots, last, step_tokens, step_positions, table) -> [2 * Rows, Vocab]` | for pages: `step_packed` the same way |

Build them from `std.nn.cache::write_slots`, `write_rows`, `write_tokens`,
and `page_slots`, and `std.nn.attention::grouped_attention_rows` and
`paged_attention`. Every zoo decoder but Phi-3 and gpt-oss has the paged
entries.

### Engine

| Call or option | Does |
| --- | --- |
| `Engine(model, pack=4096)` | with `prefill_packed`, packs waiting prompts into passes of up to `pack` tokens; otherwise prompts go in passes of up to 8, each padded to its longest |
| `Engine(model, mix=True)` | with `step_packed`, runs the decoding rows' step inside a prompt pass that fits the second compiled size (1024 tokens by default); `mix=False` keeps them apart |
| `Engine(model, share=True)` | with packed passes, identical prompts admitted together pass once: the rest copy its cache rows and draw their own tokens; `share=False` passes each |
| `engine.run(requests)` | `submit` for every request, then `step` until nothing is left |
| `engine.submit(request)` | queues a request; returns its `Completion`, which fills in as it runs |
| `engine.step()` | admits waiting requests into free rows, runs one decoding step, and returns the requests that finished, one step after they do |
| `engine.busy` | whether anything is left |
| `engine.cancel(completion)` | ends a request early (at a stop string, or when its client has gone) and frees its row |
| `engine.load_weights(model)` | copies a PyTorch model's weights (adapters merged in) into the served model, in place, between runs; compiled passes and CUDA graphs stay |

### Pages

```python
model = nest.load("llama-3.1-8b-instruct", device="cuda",
                  generics={"Batch": 1, "MaxSeq": 131072})
engine = Engine(model, rows=128, max_len=8192)
```

Loaded with `Batch` 1, a card with the paged entries keeps its caches as one
pool of `MaxSeq` positions, in pages of `PageSize` (64 by default). A request
takes pages as it grows and gives them back when it ends, so it holds the
cache it uses, not a whole row. Requests shorter than the longest allowed fit
more of them in the same memory.

| Option | Does |
| --- | --- |
| `rows=64` | the most requests in flight |
| `max_len` | the longest prompt plus completion; by default every page but one, up to 8192 |
| `reserve` | pages kept free for the rows already decoding; 1% by default |
| `paged` | `True` or `False` to choose; by default pages when the card has the entries and `Batch` is 1 |

When the pool runs short, the request admitted last gives its pages back and
waits. Once pages are free it passes its prompt and its tokens so far as one
longer prompt and continues, with the same tokens it would have drawn
(`completion.preempted`, `stats.preempted`). On CUDA, FlexAttention reads each
row's pages where they lie in the pool.

`linnet serve MODEL --pool 131072 --batch 128 --max-seq 8192` serves the same
way.

### Requests

```python
Request(prompt=ids, max_new_tokens=256, temperature=0.8, top_p=0.95, top_k=50, seed=7)
```

| Field | Does |
| --- | --- |
| `temperature` | samples on the device; without it, decoding is greedy |
| `top_k`, `top_p` | keep the `top_k` most likely tokens, then the fewest whose probabilities reach `top_p` (the order Transformers filters in) |
| `seed` | the same tokens alone or in any batch, on PyTorch, JAX, or ONNX Runtime; without it the engine picks one (`completion.sampling.seed`) |
| `logprobs=k` | each token's log probability and the `k` most likely alternatives |
| `on_token` | called with the completion as each token is read, `reason` set on the last; use it to stream |

For Llama 3.1 8B at 64 rows, a temperature costs under 1% of throughput and
top-p about 6% (PyTorch) and 8% (JAX). On ONNX Runtime, sampling runs on the
host in NumPy.

### Prepared weights

`linnet torch --prepare` and `linnet jax --prepare` move work that reads only
weights, such as dequantizing a quantized checkpoint, into a `prepare`
function that runs once at load instead of on every call. A model loaded with
`trainable=True` keeps that work in the graph, where gradients reach it.

### Over HTTP

```bash
uv add "linnet-lang[serve]"
linnet serve llama-3.1-8b-instruct --batch 32 --max-seq 4096 --port 8000
linnet serve Qwen/Qwen2.5-7B-Instruct          # a transformers checkpoint, converted
```

`linnet serve` runs `python -m linnet.serve` with the Python it was
installed beside (`LINNET_PYTHON` picks another). It takes anything
`linnet.nest.load` does and serves it with OpenAI's API: `GET /v1/models`,
`POST /v1/completions` and `POST /v1/chat/completions` (server-sent events
with `"stream": true`, and `stream_options.include_usage`), and
`GET /health`. OpenAI's clients work against it as they are:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Name three primary colors."}],
    max_tokens=64,
)
```

| Option | Does |
| --- | --- |
| `--batch`, `--max-seq` | the `Batch` and `MaxSeq` generics |
| `--backend jax`, `--backend onnx` | serves through that runtime instead of PyTorch |
| `--tokenizer` | the tokenizer and chat template's repository (default: the card's checkpoint repository) |
| `--warmup 128 512` | compiles those prompt lengths before the first request; others compile when first seen |
| `--port` | the HTTP port |

| Request field | Does |
| --- | --- |
| `max_tokens` (`max_completion_tokens` in a chat) | the token budget |
| `temperature` | 1 unless given, as OpenAI has it |
| `top_p`, `top_k`, `seed` | as in `Request` |
| `stop` | up to four strings, cut from the text and never streamed |
| `n` | that many choices, each its own request in the batch; with a `seed`, choice `i` draws with `seed + i` |
| `logprobs` (completions), `logprobs` and `top_logprobs` (chat) | each token's log-probability and up to 20 alternatives, in OpenAI's two formats, from the logits before temperature and filtering |
| `echo` (completions) | the prompt before the completion's text; refused together with `logprobs` |

Log probabilities cost nothing when no request asks for them. To serve an
engine of your own, use `linnet.serve.server.Server(engine, tokenizer,
name=...)`.

## Triton Inference Server

```bash
python -m linnet.triton export gpt2 -o model_repository --bind B=1 --bind S=64
tritonserver --model-repository=model_repository
```

`linnet.triton.export` writes a model directory the server loads as is:
`config.pbtxt` from the entry's signature and bindings, plus a version
directory holding the model. The model is a Nest name or directory, or a
`.linnet` file with `--weights` and `--bindings`.

| Backend | What is written | Use it for |
| --- | --- | --- |
| `--backend onnx` (default) | `1/model.onnx` with the checkpoint as initializers (external data beyond 1 GB), `platform: "onnxruntime_onnx"` | stateless entries such as `forward` |
| `--backend python` | `1/model.py` plus the source and weights, run through `linnet.torch` | entries with `state` (KV caches), any dtype, `numerics="fast"` |

Shapes are static (`max_batch_size: 0`), so export one model per shape or
batch size. Entry generics such as `B` and `S` come from `--bind`, or from the
card's `[check]` table for a Nest model, which `--bind` overrides. The Python backend needs
`linnet-lang[torch]` and the compiler in the server's environment
(`LINNET_BIN`, `LINNET_STD`), and returns `bf16` results as `f32`.

```python
from linnet import triton

repository = triton.export("models/tinyllama-1.1b-chat", "model_repository",
                           generics={"S": 128}, backend="python", numerics="fast")
repository.inputs      # (Tensor(name='tokens', dtype='i32', dims=(1, 128)),)
```

For the ONNX model alone, use `linnet.onnx.export_model` ([ONNX](onnx.md)).

## vLLM, SGLang, TGI

```bash
python -m linnet.hf export tinyllama-1.1b-chat -o serve/tinyllama
vllm serve serve/tinyllama
```

`linnet.hf.export` writes a Transformers checkpoint directory
(`config.json`, `model.safetensors`, tokenizer files) for a model whose
structure matches one of these families:

| Family | Structure |
| --- | --- |
| `llama` | grouped-query attention, RMS norms, SwiGLU; biases on all four attention projections or none, on all three MLP projections or none |
| `qwen2` | Llama's layout with biases on the query, key, and value projections only |
| `qwen3` | an RMS norm over each query and key head, and a head width of its own (`HeadDim`) |
| `phi3` | one projection for query, key, and value, one for gate and up; as many key/value heads as query heads |
| `gpt2` | learned positions, biased LayerNorm, GELU MLP |

Any other structure is refused, naming the paths that did not fit; there is
no general vLLM model class for arbitrary Linnet programs yet. For a
`.linnet` file, pass `--weights`, `--bind`, and `--tokenizer <repo>`. Define
`THETA` for `rope_theta`, plus `FACTOR`, `LOW_FREQ_FACTOR`,
`HIGH_FREQ_FACTOR`, and `ORIGINAL_MAX_POSITION_EMBEDDINGS` for Llama 3.1's
`llama3` `rope_scaling`, and give every norm the same epsilon. Exports attend
over the whole sequence, as the programs do, so Phi-3's export sets
`sliding_window: null`.

Tested:

- `transformers` matches the Linnet interpreter to 1e-4 on exported tiny
  GPT-2, Llama, Llama 3.1, Qwen2, Qwen3, and Phi-3 configurations.
- Qwen2.5 0.5B, Qwen3 4B and 8B, and Phi-3 mini export to the Hub
  checkpoints' own tensors: bitwise-identical `transformers` logits for Qwen3
  8B, the same greedy tokens in vLLM 0.30 (Qwen3 8B in eager mode), and the
  same GGUF tensors from llama.cpp's converter.
- vLLM 0.30 on an H100 served the `tinyllama-1.1b-chat` export as is, chat
  template included.
- In the [benchmarks](https://linnet.franknoh.dev/benchmarks#exports), each
  Llama-family and GPT-2 export gives the same first token as the original in
  vLLM, SGLang, and TGI, at the same speed within run-to-run noise.

## llama.cpp and Ollama

```bash
python -m linnet.gguf export tinyllama-1.1b-chat -o serve/tinyllama \
    --converter ~/llama.cpp/convert_hf_to_gguf.py --outtype q8_0
ollama create tinyllama -f serve/tinyllama/Modelfile
```

`linnet.gguf.export` writes the Transformers checkpoint above, converts it
with llama.cpp's `convert_hf_to_gguf.py` (`--converter`, or `LLAMA_CPP`
pointing at a llama.cpp checkout), and writes an Ollama `Modelfile` next to
the GGUF file. The chat template carries over when Ollama has an equivalent
(ChatML, Zephyr, Llama 2). The families are the same as for vLLM.

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

Each node checks the entry's signature before it runs, so a wrong shape fails
at the node, not inside a kernel. The compiler comes from `LINNET_BIN` or
`PATH`.
