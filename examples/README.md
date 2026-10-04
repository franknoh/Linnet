# Examples

What Linnet is for, one example each. [Nest](https://nest.franknoh.dev) has
24 checked models from MiniLM to gpt-oss 20B; these show the language and
what runs on it.

| Example | What it shows |
| --- | --- |
| `01-llama` | The language on a real decoder: grouped-query attention whose `where` clause states what its reshapes rely on, a KV cache in `state`, generation loops in the graph, and every layer as library source. |
| `02-vit` | A vision model: patches cut by reshapes the checker proves, shapes sized by expressions, and pooling chosen by a compile-time `enum`. |
| `03-clip` | A package of four modules: a dual encoder whose two towers share one encoder, with three entries over one parameter set. |
| `04-serve` | Llama 3.1 8B served with continuous batching and CUDA graphs, faster than vLLM, and an OpenAI-compatible server. |
| `05-train-vit` | A model trained from scratch in a plain PyTorch loop: a Linnet model is a `torch.nn.Module`. |
| `06-lora` | Llama 3.1 8B fine-tuned with LoRA on one GPU, 1.8 times TRL's speed in less memory. |
| `07-fsdp` | Llama 3.1 8B fine-tuned in full across four GPUs, in PyTorch and in JAX from the same card. |
| `08-rl` | GRPO with the serving engine sampling, and DPO, on Llama 3.1 8B: 2.4 to 2.7 times TRL's speed. |

The `.linnet` examples check with `linnet lint --std stdlib examples/<name>`,
and the tests under `python/linnet/tests/torch` compare them with
hand-written PyTorch. Weights bind by parameter path
(`linnet inspect --parameters`); the tests fill them with random tensors
through SafeTensors. The examples from `04-serve` on run Nest's Llama 3.1
8B card on H100s, and their READMEs give the numbers measured there.
