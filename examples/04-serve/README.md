# Serving

Llama 3.1 8B from [Nest](https://nest.franknoh.dev), served by
`linnet.serve`: continuous batching over a fixed set of cache rows, prompts
packed end to end into one pass, and each step replayed as a CUDA graph. The
model is the card's `.linnet` source; nothing of the original repository
runs.

## Run

```bash
python examples/04-serve/throughput.py                  # generated PyTorch, CUDA graphs
python examples/04-serve/throughput.py --backend jax    # XLA
```

`throughput.py` gives the engine the benchmarks' load: 256 requests of 128
to 512 prompt tokens, at most 64 in flight, each generating 128 tokens. It
prints the generated tokens per second: on one H100, about 5600 with
PyTorch and 4700 with JAX.

On one H100 in `bf16`:

| Model | Linnet | vLLM |
| --- | --- | --- |
| Llama 3.1 8B, 256 requests | 5600 tokens/s | 5449 |
| gpt-oss 20B, 256 requests | 6031 tokens/s | 4313 |
| Llama 3.1 8B, one request | 167 tokens/s (XLA) | 152 |

The [benchmarks](https://linnet.franknoh.dev/benchmarks) have every row and
how it was measured.

## An OpenAI-compatible server

```bash
linnet serve llama-3.1-8b-instruct --batch 32 --max-seq 4096 --port 8000
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model": "llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "Hi"}]}'
```

The server takes OpenAI's completions and chat completions, streamed or
not. [Serving and tools](https://linnet.franknoh.dev/docs/integrations) lists
the options, and the exports to vLLM, Triton and llama.cpp.

## How the engine is fast

- **Fixed rows.** Each request in flight owns one row of the KV caches
  (`Batch` rows of `MaxSeq` positions), so a decoding step has one shape and
  replays as one captured CUDA graph.
- **Packed prompts.** Prompts that arrive together are packed end to end
  into one pass with no padding, each attending only to itself
  (FlexAttention over the blocks it reaches). Prompts that arrive while other
  rows decode join that step's pass.
- **Shared prompts.** Requests with the same prompt prefill it once and copy
  its cache rows.
