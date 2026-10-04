# Training

`linnet.train` trains a Linnet decoder in PyTorch: supervised fine-tuning,
reinforcement learning (GRPO) and preferences (DPO), on one GPU or split
across many. `linnet.jax.train` trains it in [JAX](#jax). `linnet.torch.CausalLM` hands the same model to
`transformers.Trainer` and TRL instead. Each trains through the model's
packed entries (`loss_packed`, `log_probs_packed`, `hidden_packed`), which
every Nest decoder card has. Loading a model to train is in
[PyTorch](torch.md#training).

## Supervised fine-tuning

```python
from linnet.train import Example, cosine_schedule, pack, train

examples = [Example.prompted(prompt, completion) for prompt, completion in data]
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
history = train(model, pack(examples, tokens=4096), optimizer=optimizer, steps=1000,
                accumulate=8, schedule=cosine_schedule(optimizer, 100, 1000), save_to="run/")
```

`pack` packs examples into batches of exactly `tokens` positions, padding
the rest so the model compiles once. `Example.prompted` learns only the
completion. `train` runs the model's `loss_packed` entry; every Nest decoder
card and `examples/01-llama` have one. It sums `accumulate` batches per
optimizer step, weighing each by the step's total count of learned
positions, then clips the gradient norm to `clip` (1.0) and steps the
schedule. It saves the weights to `save_to`, or the adapters alone after
`add_lora`.

Under `torchrun`, each process trains a copy on its own batches: `train`
sums the learned-position counts and the gradients across processes, so
every copy takes the same step. Pass
`torch.distributed.optim.ZeroRedundancyOptimizer` to split the optimizer
state. Only the first process saves.

## Checkpoints

With `checkpoint="run/"`, `train` resumes from the latest checkpoint
there, skipping the batches its steps took, and writes one every
`checkpoint_every` steps and at the end, keeping the two latest. `grpo` and
`dpo` take the same two arguments.

- A checkpoint holds the parameters being trained, the optimizer's state,
  the step and the schedule. A LoRA run's holds only the adapters.
- Reload the model as the run loaded it, then pass the same `checkpoint`.
- Under `torchrun`, every process writes its own part, including the parts
  of a model split by `fully_shard`.

`save_checkpoint` and `load_checkpoint` do the same for a loop of your own.

## Sharded training

`fully_shard` splits every parameter across the processes (FSDP). Each
process then holds a part of the weights, the gradients and the optimizer
state:

```python
from linnet.torch import fully_shard

torch.distributed.init_process_group("nccl")
model = nest.load(card, backend="torch", device=f"cuda:{rank}", compile="inductor",
                  trainable=True)
fully_shard(model)  # parts in f32; gathered in the model's dtype
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
train(model, pack(examples_of_this_process, tokens=4096), optimizer=optimizer, steps=1000)
```

- Each process keeps its part of each parameter in `dtype` (f32), and the
  optimizer updates those parts.
- The generated code gathers each parameter, in the dtype the model
  computes in, just before its block runs, and drops it after.
- Gradients are summed back into the parts in f32.
- Backward gathers again rather than keep the whole weights: under
  `compile="inductor"` through recomputation, otherwise through saved-tensor
  hooks.
- `save_weights` gathers the parts. Every process calls it, and the first
  writes the file.

On four H100s, Llama 3.1 8B fine-tunes in full this way. The run used f32
parts, AdamW, and 4096 packed tokens per process per step; held-out Alpaca
loss went from 1.91 to 1.35 in 30 steps.

| `compile` | Step | Tokens/s, four GPUs | Peak per GPU |
| --- | --- | --- | --- |
| `"inductor"` | 549 ms | 29.8K | 47.5 GiB |
| `True` | 729 ms | 22.5K | 54.0 GiB |

Add adapters (`add_lora`) before `fully_shard`. They then stay whole on
every process, with their gradients summed. The weights they adapt do not
train, so they are split in their own dtype, with no f32 copy. A large
model's LoRA run fits this way. Merge the saved adapters on one process.
On four H100s, Llama 3.1 8B's LoRA run took 0.33 s a step in 26 GiB a GPU.
Its held-out loss matched the one-GPU run's (1.369) over the same batches.

PyTorch's own `fully_shard` cannot shard a Linnet model. It gathers a
module's parameters in hooks around that module's `forward`, and generated
code computes every block in one function without calling any of them.

## Other trainers

`CausalLM` gives a Linnet decoder the call a transformers-style trainer
(`transformers.Trainer`, TRL) makes: `model(input_ids=..., attention_mask=...,
labels=..., position_ids=...)` returns `{"loss": ...}`.

```python
from linnet.torch import CausalLM

trainer = transformers.Trainer(model=CausalLM(model), args=args, train_dataset=data)
```

It removes the padding and packs the rows into the model's `loss_packed`
entry, so neither the padding nor the whole logits cost anything. `labels`
follow transformers: `-100` leaves a position out, and the shift happens
inside. With `position_ids`, a 0 starts a new sequence within a row, as in
TRL's padding-free batches. Packed lengths round up to a multiple of
`bucket` (256), so few shapes compile. Given `num_items_in_batch`, which
`transformers.Trainer` passes, the loss is the sum over that count, so
gradient accumulation takes the mean over the whole step.

TRL's `SFTTrainer` takes the same model:

```python
trainer = trl.SFTTrainer(model=CausalLM(model), args=trl.SFTConfig(...),
                         train_dataset=data, processing_class=tokenizer)
```

Its default loss (`chunked_nll`) computes the cross-entropy itself, a block
of tokens at a time. It reads the hidden states from `CausalLM.base_model`
and the head from `get_output_embeddings()`. Both come from the model's
`hidden_packed` entry, which every Nest decoder card has. With
`loss_type="nll"`, TRL reads the logits instead. Pass `logits=True` to get
them in the input's layout, computed without gradients; they take the
memory a transformers model's would. `gradient_checkpointing=True`, TRL's
default, only warns: with `compile="inductor"`, set
`activation_memory_budget` instead.

On one H100, Llama 3.1 8B with LoRA adapters (rank 16, every attention
and MLP projection) trains on Alpaca as follows:

| Trainer | Batches | Result |
| --- | --- | --- |
| `linnet.train`, `compile="inductor"` | 4096 packed tokens | 260 ms per step (15.7K tokens/s), 36 GiB; held-out loss 1.91 to 1.37 in 40 steps |
| `transformers.Trainer`, `compile=True` | 8 padded rows, 2 per step | 20 steps in 7 s, 22 GiB; loss 1.93 to 1.19 |
| TRL `SFTTrainer` on four GPUs (DDP), `compile=True` | 8 padded rows per GPU, 2 per step | 30 steps in 17 s, 24 GiB per GPU; loss 2.05 to 1.31, token accuracy 0.57 to 0.66 |

## Reinforcement learning

```python
from linnet.serve import Engine
from linnet.train.grpo import Prompt, grpo

policy = nest.load(card, backend="torch", device="cuda", compile="inductor", trainable=True)
engine = Engine(nest.load(card, backend="torch", device="cuda", compile=True))
optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-6)
history = grpo(policy, engine, [Prompt(ids, answer) for ids, answer in data], reward,
               optimizer=optimizer, steps=200, group=8, prompts_per_step=8, max_new_tokens=512)
```

`grpo` trains with group relative policy optimization (GRPO). Each step:

1. `engine.load_weights(policy)` copies the policy's weights into the
   engine's model, adapters merged in.
2. The engine samples `group` completions of each prompt. It passes each
   prompt once and copies its cache rows to the rest of the group.
3. `reward(prompt, completion)` scores each one. Its advantage is its reward
   less its group's mean, over the group's standard deviation.
4. The policy recomputes the completions' log-probabilities with its
   `log_probs_packed` entry and takes the clipped policy-gradient step,
   averaged over every completion token.

`iterations` reuses each step's samples for that many optimizer steps; the
ratio is then clipped to `1 - clip[0]`, `1 + clip[1]`. A group whose
rewards are all equal has no advantage: each step reports their share as
`uniform`, and `drop_uniform=True` leaves them out of the loss. `beta` adds
a KL penalty against `reference`, a frozen copy of the model. `grpo_loss`
is the loss alone, for another loop.

The engine samples from the same weights as the policy but computes them
differently: other kernels, CUDA graphs, bf16 throughout. `correction_cap=2`
weighs each token's surrogate by the policy's probability of it over the
engine's, at most 2 (truncated importance sampling), and each step reports
their mean absolute log-probability difference as `mismatch`. It reads the
engine's log-probabilities, so it takes temperature 1 and no top-k or top-p.

Under `torchrun`, each process samples its own prompts with its own engine
(pass each its own `seed`), and the processes train one policy. The policy
can be split by `fully_shard`: `load_weights` then gathers it a tensor at a
time on every process. The loss is the mean over every process's tokens. A
process with fewer packed batches runs empty ones, so every process makes
the same collective calls. `linnet.train.reduce_gradients` and
`clip_gradients` do the same for a loop of your own.

The engine compiles a pass size the first time a step needs one, and the
other processes wait for it meanwhile (about 100 s, a few times a run).
`engine.warmup(prompt_lengths)` compiles them up front instead, but holds
a captured CUDA graph's memory for every size it reaches.

Llama 3.1 8B with GRPO on H100s: 16 prompts × 8 completions of up to 128
tokens per GPU per step, after compiling.

| Setup | Sample | Train | Weight copy | Peak per GPU |
| --- | --- | --- | --- | --- |
| LoRA, one GPU, 45-token prompts | 0.79 s | 0.26 s | 104 ms | 59 GiB |
| LoRA, one GPU, 1267-token prompts | 5.7 s (7.2 s passing every prompt) | 12.5 s | | 59 GiB |
| Full fine-tune, four GPUs, `fully_shard` | 1.6 s | 1.5 s | 109 ms | 74 GiB |

## Preferences

```python
from linnet.train.dpo import Pair, dpo

pairs = [Pair(prompt, chosen, rejected) for prompt, chosen, rejected in data]
history = dpo(model, pairs, optimizer=optimizer, steps=1000, pairs_per_step=32, beta=0.1)
```

`dpo` trains with direct preference optimization (DPO). A pair's two
answers are packed into one batch, and `log_probs_packed` gives each
answer's log-probability. The loss is
`-log sigmoid(beta * ((chosen - ref_chosen) - (rejected - ref_rejected)))`,
averaged over the step's pairs. `label_smoothing` takes that share of pairs
to be labelled the wrong way round.

- With `reference`, a frozen model with the same entry, the reference
  log-probabilities are computed batch by batch.
- Without one, `pairs` must be a sequence. The model's own
  log-probabilities before training are then computed for every pair
  first, so a LoRA run needs one model.

Each step reports the loss, the share of pairs the model now ranks
correctly (`accuracy`), and the mean reward `margin`. Under `torchrun` and
with `fully_shard` it works as `grpo` does.

On one H100, DPO with LoRA on Llama 3.1 8B over UltraFeedback pairs (32 a
step, answers up to 512 tokens) takes 1.9 s a step in 36 GiB. Accuracy
went from 0.61 to 0.68 and the margin from 0.01 to 0.09 between the first
and last five of 30 steps.

## Low-rank adapters

```python
model = load("src/model.linnet", generics={...}, weights="init/", compile="inductor")
model.add_lora("layers.*.attention.*_proj.weight", rank=16, alpha=32)
# ... train: only the adapters require gradients
model.save_weights("adapters.safetensors", names="linnet", include=["*.lora_a", "*.lora_b"])
model.merge_lora()  # plain weights again, for serving or export
```

`add_lora` gives every linear weight whose path matches a pattern a
low-rank adapter (LoRA). The layer then computes
`x @ W.T + (x @ A.T) @ B.T * alpha / rank`, with `A` and `B` as the weight's
block's `lora_a` and `lora_b` parameters. `B` starts at zero, so the model
starts unchanged. Adapters need generated code. To load saved adapters, call
`add_lora` with the same patterns and rank, then
`bind_weights(model, path, strict=False)`.

## JAX

```python
import optax
from linnet.jax.train import train
from linnet.packing import Example, pack

model = nest.load(card, backend="jax_source", entry="loss_packed",
                  generics={"Batch": 1, "MaxSeq": 4096, "T": "bf16"}, cast_dtype=True)
params, history = train(model, pack(examples, tokens=4096), optimizer=optax.adamw(1e-5),
                        steps=1000, accumulate=8)
```

`linnet.jax.train` trains the `loss_packed` entry as generated JAX
(`load_source`, or `nest.load(..., backend="jax_source")`) with an optax
optimizer, and returns the parameters (path -> array) and the steps.

- `linnet.packing` packs examples as numpy arrays, with no PyTorch:
  `linnet.train.pack` is the same packing as tensors.
- One compiled step runs `accumulate` batches as a `lax.scan`, sums their
  gradients weighed by the step's count of learned positions, clips them to
  `clip` (1.0), and applies the optimizer.
- `trainable` picks the parameters that train (all, or glob patterns). They
  are kept in f32 and cast to the dtype the model computes in on each call;
  the rest stay as loaded.
- Paths bound to one checkpoint tensor (a tied embedding and output head)
  load as one array and train as one parameter.
- `model.add_lora(patterns, rank=16, alpha=32)` gives the matching linear
  weights low-rank adapters, as in PyTorch: the generated JAX computes
  `x @ A.T @ B.T` beside each product, and only the adapters train by
  default. `merge_lora(model, params)` adds them into the weights.
- `save_weights(params, path)` writes them under their Linnet paths, which
  `load_source` reads back. `checkpoint` and `checkpoint_every` resume and
  save as in PyTorch, from one host.

With `mesh=Mesh(jax.devices(), ("data",))`, training is data-parallel and
fully sharded (FSDP):

- Each step takes one batch per device per accumulated step. Every device
  computes its own (`shard_map`).
- Every parameter, its gradient and its optimizer state are split along the
  largest axis the device count divides.
- Each block gathers its weights where it runs, in the dtype the model
  computes in (`linnet jax --fully-shard`). Gradients are reduced into the
  parts in f32.
- The step compiles with GSPMD: Shardy crashes XLA on cuDNN attention
  inside `shard_map`.
- Pass the mesh before the model's first call, and the weights load straight
  into their parts.

With `remat=True`, the backward pass computes each layer again instead of
keeping its values (`linnet jax --remat`, a `jax.checkpoint` per layer).
Over a mesh it also gathers the layer's weights again. A step then keeps
each layer's inputs alone, for about a third more compute.

The output head's loss runs a block of rows at a time here too
(`linnet.jax.loss`, a `custom_vjp`), so the `[P, Vocab]` logits are never
held whole.

`linnet.jax.dpo.dpo` and `linnet.jax.grpo.grpo` train the
`log_probs_packed` entry as their PyTorch counterparts do, with the same
options, in one process. `mesh` and `remat` work as for `train`: each
device takes its own batches, and a step's last call is filled out with
batches that learn nothing.

- `dpo(model, pairs, optimizer=...)` takes `reference` as parameters (path
  -> array). Without them, it computes the model's own log-probabilities
  first.
- `grpo(policy, engine, prompts, reward, optimizer=...)` samples with an
  `Engine` over the JAX model (`linnet.jax.load_model`). The engine takes the
  policy's weights before each step, adapters merged in
  (`LinnetModel.copy_weights`). Over a mesh it takes them as they are split
  and places them as its own model's are.
- Over a mesh, split the engine's model over the same devices
  (`model.shard(mesh)`). An engine on one of them holds a whole model
  beside its part of the policy, and every weight copy goes through the
  host.
- Leave XLA's default memory fraction when the engine and the policy share
  devices. Their collectives need memory outside XLA's pool: with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, the 4-GPU run stopped in its first
  collective.

## Compared with TRL

The same runs in Linnet (PyTorch and JAX) and in TRL with PEFT, on Llama
3.1 8B Instruct. They share the data, 4096-token packed rows, the
hyperparameters and the step count. Each ran on one H100 unless noted. Times
are the median step after the first two. Tokens/s counts real tokens, not
padding. The [benchmarks](https://linnet.franknoh.dev/benchmarks#training)
chart them.

| Run | Linnet PyTorch | Linnet JAX | TRL |
| --- | --- | --- | --- |
| SFT, LoRA: Alpaca, 4 rows a step | 1.04 s, 15.6K tokens/s, 36 GiB | 1.33 s, 12.2K tokens/s, 38 GiB | 1.83 s, 8.9K tokens/s, 43 GiB |
| SFT in full, four GPUs | 0.55 s, 29.4K tokens/s, 48 GiB a GPU | 0.54 s, 30.0K tokens/s, 65 GiB a GPU (43 GiB with `remat`) | 0.57 s, 28.6K tokens/s, 52 GiB a GPU (FSDP2) |
| DPO, LoRA: UltraFeedback, 32 pairs a step | 1.91 s, 36 GiB | 2.41 s, 38 GiB | 4.63 s, 48 GiB |
| GRPO, LoRA: 16 prompts × 8, up to 128 tokens | 1.18 s, 59 GiB | 1.44 s, 60 GiB | 3.14 s, 52 GiB (vLLM) |

The three trained alike:

- **SFT, LoRA:** held-out loss after 30 steps was 1.369 (PyTorch), 1.374
  (JAX) and 1.373 (TRL). One evaluator scored all three adapters.
- **SFT in full:** training loss over the last five steps was 1.24 for
  Linnet PyTorch and JAX alike, and 1.26 for TRL. Held-out loss after 30
  steps was 1.342 (PyTorch) and 1.345 (JAX).
- **DPO:** accuracy over the last five steps was 0.68, 0.65 and 0.70.
- **GRPO:** mean reward went from 0.44 to 0.39, 0.44 to 0.40, and 0.43 to
  0.41, first three steps to last three. At this learning rate none learns
  in 20 steps, and all three trace the same curve.
- No run had a loss spike, and the gradient norms stayed in the same range.

Conditions that differ:

- **First step.** It compiles. Linnet PyTorch takes 35 s for SFT, 75 s for
  DPO and 360 s for GRPO, which includes compiling the engine's passes.
  JAX takes 23 to 55 s, and TRL 2 to 9 s.
- **TRL batch sizes.** TRL took DPO 2 pairs at a time and GRPO 8
  completions at a time; larger batches ran out of memory. vLLM held 35% of
  the GPU.
- **JAX on four GPUs.** Measured later on another host, with the step
  compiled by GSPMD. `remat` (each layer recomputed) cut the peak from 65 to
  43 GiB a GPU for a 0.67 s step, with the same losses.

## Losses

`std.nn.loss` holds the losses, over flattened tokens (`[N, V]` logits),
computed in f32:

| Op | Result |
| --- | --- |
| `cross_entropy(logits, targets, weights)` | `sum_n weights[n] * -log p(targets[n])`; pass `mask / count` for a mean over `count` tokens |
| `token_log_probs(logits, targets)` | each row's log-probability of its target, `[N]` |
| `log_softmax(logits)`, `entropy(logits)` | per row |
| `linear_cross_entropy(hidden, weight, targets, weights)` | `cross_entropy` of the output head `hidden @ weight.T` |
| `linear_token_log_probs(hidden, weight, targets)` | `token_log_probs` of the output head |

The `linear_` forms never hold the `[N, V]` logits in PyTorch: they run a
block of about 1 GiB of f32 logits at a time (2 GB in full for 4096 tokens
of Llama 3's vocabulary), and accumulate the weight's gradient in f32.
