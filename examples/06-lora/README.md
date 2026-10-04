# LoRA fine-tuning

Llama 3.1 8B from [Nest](https://nest.franknoh.dev) fine-tuned with LoRA
adapters on one GPU: Alpaca, packed into 4096-token rows, trained by
`linnet.train`.

## Run

```bash
python examples/06-lora/sft.py                    # 30 steps of 4 rows
python examples/06-lora/sft.py --steps 300 --out adapters.safetensors
```

It prints each step, the held-out loss before and after, the speed and peak
memory, and writes the adapters.

On one H100, against TRL with PEFT on the same data and settings:

| | Step (4 rows) | Tokens/s | Peak | Held-out loss after 30 steps |
| --- | --- | --- | --- | --- |
| Linnet, PyTorch | 1.04 s | 15.6K | 36 GiB | 1.369 |
| Linnet, JAX | 1.33 s | 12.2K | 38 GiB | 1.374 |
| TRL + PEFT | 1.83 s | 8.9K | 43 GiB | 1.373 |

## What it shows

- **Adapters without a model class.** `model.add_lora(patterns, rank=16)`
  adds `A` and `B` beside every weight whose path matches a glob pattern. The
  generated code computes `x @ W.T + (x @ A.T) @ B.T * alpha / rank`, and
  only the adapters train. `merge_lora()` folds them back into the weights
  for serving or export.
- **No padding.** `linnet.train.pack` packs examples end to end into rows of
  4096 positions. The card's `loss_packed` entry attends within each one and
  learns only the answers.
- **No `[tokens, vocab]` logits.** The loss runs through
  `std.nn.loss::linear_cross_entropy`, which PyTorch computes a block of
  rows at a time, with the gradient computed alongside.

The same model trains under transformers' `Trainer` and TRL's `SFTTrainer`
through `linnet.torch.CausalLM`
([other trainers](https://linnet.franknoh.dev/docs/training#other-trainers)).
On four GPUs, `fully_shard` splits the frozen weights and keeps the adapters
whole ([07-fsdp](https://linnet.franknoh.dev/examples/07-fsdp)).
