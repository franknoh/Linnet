# Reinforcement learning and preferences

Llama 3.1 8B trained on one GPU with LoRA adapters, two ways: GRPO against a
reward, with answers sampled by `linnet.serve`, and DPO on preference pairs.

## Run

```bash
python examples/08-rl/grpo.py     # learns to get two-number multiplications right
python examples/08-rl/dpo.py      # UltraFeedback's chosen and rejected answers
```

On one H100, against TRL on the same settings:

| | Linnet, PyTorch | Linnet, JAX | TRL |
| --- | --- | --- | --- |
| GRPO step: 16 prompts × 8 answers of up to 128 tokens | 1.18 s | 1.44 s | 3.14 s (vLLM sampling) |
| DPO step: 32 pairs | 1.91 s | 2.41 s | 4.63 s |

## GRPO

Each step, `linnet.train.grpo`:

1. copies the policy's weights, adapters merged, into the engine's model
   (`Engine.load_weights`), with no new compilation;
2. samples `group` answers to each prompt with the engine;
3. scores them with `reward`, and gives each answer its advantage over its
   group's mean;
4. trains the policy on the packed answers with the clipped policy-gradient
   loss.

`correction_cap=2.0` weighs each token by how much likelier the policy finds
it than the engine did, capped at 2 (truncated importance sampling). That
corrects for the two computing the same model with different kernels. The
engine is the serving engine from [04-serve](https://linnet.franknoh.dev/examples/04-serve): continuous
batching, packed prompts, one CUDA graph a step.

## DPO

`linnet.train.dpo` packs each pair's two answers into one batch, sums their
log-probabilities from the card's `log_probs_packed` entry, and computes
`-log sigmoid(beta * (margin - reference margin))`. Without a separate
reference model, the reference log-probabilities are the model's own,
computed for every pair before training.

Both run fully sharded across GPUs in PyTorch (`fully_shard`) and over a
mesh in JAX (`linnet.jax.grpo` and `linnet.jax.dpo` with `mesh=`).
[Training](https://linnet.franknoh.dev/docs/training) has every option.
