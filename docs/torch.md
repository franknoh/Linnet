# PyTorch

`linnet.torch` loads a Linnet root block as a `torch.nn.Module` and exports
a `torch.nn.Module` as Linnet source.

## Install

```bash
cd python/linnet && uv sync --extra torch    # or: pip install ".[torch]"
export LINNET_BIN=/path/to/build/release/linnet     # or put `linnet` on PATH
```

## Load a model

```python
from linnet.torch import load

model = load(
    "examples/05-llama/src/lib.linnet",
    generics={"Vocab": 32000, "H": 512, "Heads": 8, "KvHeads": 4, "Inner": 1376,
              "Layers": 2, "Batch": 1, "MaxSeq": 128, "T": "bf16"},
    std_root="stdlib",
    weights="weights/",          # a .safetensors file or a directory of them
)
logits = model(tokens)
```

Printing the module shows the block hierarchy with Linnet names, shapes
and dtypes:

```text
LinnetModule(
  (root): Model(
    (embedding): Embedding(weight=bfloat16[32000, 512])
    (layers): ModuleList(
      (0-1): 2 x DecoderLayer(
        (attention_norm): RmsNorm(weight=bfloat16[512])
        (attention): GroupedQueryAttention(
          cache_k=bfloat16[1, 4, 128, 64] state, cache_v=bfloat16[1, 4, 128, 64] state
          (q_proj): Linear(weight=bfloat16[512, 512], bias=bfloat16[512]?)
          (k_proj): Linear(weight=bfloat16[256, 512], bias=bfloat16[256]?)
          (v_proj): Linear(weight=bfloat16[256, 512], bias=bfloat16[256]?)
          (o_proj): Linear(weight=bfloat16[512, 512], bias=bfloat16[512]?)
        )
        (mlp_norm): RmsNorm(weight=bfloat16[512])
        (mlp): SwiGlu(
          (gate): Linear(weight=bfloat16[1376, 512], bias=bfloat16[1376]?)
          (up): Linear(weight=bfloat16[1376, 512], bias=bfloat16[1376]?)
          (down): Linear(weight=bfloat16[512, 1376], bias=bfloat16[512]?)
        )
      )
    )
    (norm): RmsNorm(weight=bfloat16[512])
    (lm_head): Linear(weight=bfloat16[32000, 512], bias=bfloat16[32000]?)
  )
)
```

`?` marks an optional parameter, `state` a KV-cache member. `state_dict()`
names each tensor `root.` and its Linnet path
(`root.layers.0.attention.q_proj.weight`).

`load` checks the program with `linnet plan` without running it. With
`weights`, it checks every tensor's name, shape and dtype against the
manifest before copying; optional parameters can be absent. `state`
members start at zero. Entries become methods: `forward` is the entry of
that name or the only one, and entry generics such as `B` and `S` bind
from the inputs.

## Load options

| Option | Effect |
| --- | --- |
| `numerics=` | see [Numerics policy](#numerics-policy) |
| `compile=True` | entries run as generated PyTorch source, once per entry and input shape. Default on CUDA |
| `compile=False` | the interpreter. Default elsewhere |
| `compile="inductor"` | generated source through `torch.compile` |
| `compile="reduce-overhead"` | CUDA graphs; see [CUDA graphs](#cuda-graphs) |
| `trainable=True` | parameters require gradients; glob patterns (`["layers.*.mlp.*"]`) choose a subset. See [Training](#training) |
| `bindings="bindings.json"` | maps Linnet paths to checkpoint tensor names |
| `cast_dtype=True` | converts floating-point weights to the model's dtype as they are read |
| `amp="bf16"` or `"f16"` | mixed precision under `torch.autocast`; weights keep the model's dtype (`T=f32` keeps f32 masters) and gradients arrive in f32. With `"f16"`, scale the loss with `torch.amp.GradScaler` |
| `device_map`, `max_memory`, `offload` | see [Placement and offload](#placement-and-offload) |
| `tensor_parallel`, `tp_rules` | see [Tensor parallelism](#tensor-parallelism) |

### CUDA graphs

`compile="reduce-overhead"` is for decoding. The first call runs through
`torch.compile`, the second captures the step as one CUDA graph, and later
calls replay it. KV caches the entry writes are updated in place.
Replacing a state or weight (`reset_state`) triggers a new capture. It
works under `tensor_parallel` and requires `offload=False`. With blocks on
several devices, or with gradients, `torch.compile`'s own CUDA graphs run
instead.

## Numerics policy

| `numerics=` | Library operations run as |
| --- | --- |
| `"fast"` (default) | PyTorch kernels, with layer normalization and attention in the input dtype, as PyTorch's reference implementations do |
| `"equivalent"` | the same kernels, accumulating in f32 as the canonical bodies specify. Slower on `bf16`, which loses fused attention ([benchmarks](https://linnet.franknoh.dev/benchmarks)) |
| `"exact"` | their canonical `.linnet` bodies: the slowest path, for comparison |

Under `"fast"` on CUDA, attention under an explicit mask (serving batches,
packed prompts, sliding windows, gpt-oss's sink attention) computes only
the key blocks the mask reaches. Mixture-of-experts layers run as one
grouped product over the chosen experts on compute capability 9 or later
with bf16.

## Entries and state

| Call | Effect |
| --- | --- |
| `model.run_entry("generate", [prompt, pos], generics={"Steps": 16})` | runs an entry, naming generics the inputs do not determine |
| `model.run_entry(..., compile=...)` | overrides `compile` for one call, such as CUDA graphs for decoding steps only |
| `model.reset_state()` | zeroes every `state` member |
| `model.state_paths()` | lists the `state` members |
| `model.generated_source(entry)` | shows the code `compile` ran |

## Multiple devices

### Placement and offload

```python
model = linnet.torch.load("model.linnet", generics=generics, weights="weights/",
                          device_map="auto")
print(model.placement.describe())
```

```text
cuda:0: embedding, layers.0-15
cuda:1: layers.16-31, norm, head
```

`device_map="auto"` places each sub-block of the root, and each element of
a sub-block array, on the GPUs in declaration order, keeping a reserve for
activations. Placement is decided from the manifest before any weight is
read. What does not fit stays in host memory and is *offloaded*: copied to
the last GPU when it runs, at the speed of the copies.

| Option | Effect |
| --- | --- |
| `max_memory={0: "20GiB"}` | caps one GPU |
| `offload=False` | raises an error instead of offloading |
| `device_map={"layers.30": "cpu", "layers.31": "cpu"}` | places units by hand |

`linnet torch --place layers.16=1 --offload layers.31` compiles a placement
into the generated source, with a `.to()` where a value crosses devices:

```python
    v212 = v209.to(_dev[1], non_blocking=True)   # the residual stream, once
    ...
    v480 = p291.to(_dev[1], non_blocking=True)   # an offloaded layer's weights
    v481 = F.linear(v479, v480, None)
    ...
    del v480                                     # gone when the layer returns
```

### Tensor parallelism

```python
# torchrun --nproc-per-node 2 serve.py
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
model = linnet.torch.load("model.linnet", generics=generics, weights="weights/",
                          device=f"cuda:{dist.get_rank()}", tensor_parallel=mesh)
```

Tensor parallelism splits every large weight across the GPUs, so all work
on each layer at once. `device_map` instead gives each GPU whole blocks,
which take turns.

A model with a `Shards` generic (1 by default), such as the zoo's
Llama-shaped decoders, declares its own split. A shard holds
`Heads / Shards` query heads, `KvHeads / Shards` key and value heads and
`Inner / Shards` hidden units; projections back out end in
`std.nn.parallel::all_reduce`.
An output projection not tied to the embedding splits into
`Vocab / Shards` rows, joined by `std.nn.parallel::all_gather`. On one
device both return their input. `tensor_parallel=mesh` binds `Shards` to
the mesh size, and each process reads only its part of the checkpoint.

Other models are split as DTensors by `linnet.parallel.DEFAULT_RULES`:

| Weights | Split |
| --- | --- |
| projections into the heads, feed-forward width | by output |
| projections back | by input |
| KV caches | by heads |
| everything else | copied to each process |

`tp_rules={"*.experts.*": 0}` adds or overrides rules by path pattern. An
axis the device count does not divide is copied. Any split computes the
same numbers, and results come back whole on every process.

With `compile="reduce-overhead"`, run `del model` before
`dist.destroy_process_group()`, which otherwise waits on the CUDA graphs'
communicators.

## Training

```python
model = load("src/model.linnet", generics={...}, weights="init/", trainable=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = criterion(model(tokens), labels)
loss.backward()
optimizer.step()
model.save_weights("trained.safetensors")
```

Entries are differentiable, interpreted or generated. `state` members are
detached between calls.

| Call | Effect |
| --- | --- |
| `trainable=True`, or `model.set_trainable(True)` | every floating-point parameter trains |
| `trainable=["layers.*.mlp.*"]` | only parameters whose path matches a glob pattern; returns the paths |
| `model.save_weights(path)` | one SafeTensors file under the checkpoint's own names, so it replaces that checkpoint (`linnet.hf.export(card, weights=path)` exports it) |
| `model.save_weights(path, names="linnet")` | the same under Linnet paths |
| `model.save_weights(path, dtype=torch.bfloat16)` | converts floating-point tensors as it writes |

Parameters the checkpoint ties (two paths bound to one tensor, such as an
embedding and its output head) load as one parameter and are written once.
Optional parameters the checkpoint lacks never train and are not written.
A model trains without weight-only work done ahead or hand-captured CUDA
graphs; a model that ran without gradients compiles its entries again once
any parameter requires them.

Train with `compile="inductor"`. For long packed batches, have it recompute
activations instead of keeping them: set
`torch._functorch.config.activation_memory_budget = 0.3` before the first
step. On one H100, Llama 3.1 8B trains its layers (7 B parameters, SGD) over
4096 packed tokens at 10.9K tokens per second in 29 GiB this way, against
9.2K and 42.5 GiB as generated source without `torch.compile`.

### Supervised fine-tuning

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
card and `examples/05-llama` have one. It sums `accumulate` batches per
optimizer step, weighing each by the step's total count of learned
positions, then clips the gradient norm to `clip` (1.0) and steps the
schedule. It saves the weights to `save_to`, or the adapters alone after
`add_lora`.

Under `torchrun`, each process trains a copy on its own batches: `train`
sums the learned-position counts and the gradients across processes, so
every copy takes the same step. Pass
`torch.distributed.optim.ZeroRedundancyOptimizer` to split the optimizer
state. Only the first process saves.

### Sharded training

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

PyTorch's own `fully_shard` cannot shard a Linnet model. It gathers a
module's parameters in hooks around that module's `forward`, and generated
code computes every block in one function without calling any of them.

### Other trainers

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

### Reinforcement learning

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
2. The engine samples `group` completions of each prompt.
3. `reward(prompt, completion)` scores each one. Its advantage is its reward
   less its group's mean, over the group's standard deviation.
4. The policy recomputes the completions' log-probabilities with its
   `log_probs_packed` entry and takes the clipped policy-gradient step,
   averaged over every completion token.

`iterations` reuses each step's samples for that many optimizer steps; the
ratio is then clipped to `1 - clip[0]`, `1 + clip[1]`. `beta` adds a KL
penalty against `reference`, a frozen copy of the model. `grpo_loss` is the
loss alone, for another loop.

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

On four H100s, GRPO fine-tunes Llama 3.1 8B in full this way. The policy
is split by `fully_shard` (AdamW), and each GPU runs its own engine over 16
prompts × 8 completions per step. After compiling, a step takes 1.6 s to
sample 512 completions and 1.5 s to train. Gathering the weights into
every engine takes 109 ms, and the peak is 73.6 GiB per GPU.

On one H100, GRPO with LoRA on Llama 3.1 8B (16 prompts, 8 completions
each, up to 128 tokens) takes 1.1 s to sample and 0.6 s to train a step
after compiling; copying the weights into the engine, adapters merged,
takes 104 ms. Peak memory for both models is 59 GiB.

### Low-rank adapters

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

### Losses

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

## Functions

An `entry` declared at module level, outside any block, is a function of
its inputs alone, such as a loss (`examples/09-functions`).
`load_function` returns one as a PyTorch function.

```python
from linnet.torch import load, load_function

model = load("functions.linnet", generics={"In": 64, "Classes": 10}, trainable=True)
cross_entropy = load_function("functions.linnet", "cross_entropy")
loss = cross_entropy(model(x), labels)
loss.backward()
```

Generics bind from each call's inputs, so one function serves every batch
size and dtype. Name any the inputs leave open: `positions(offset, N=8)`.
A Python number passes for a scalar input. `compile=` works as for `load`;
unset, it generates source for inputs on CUDA and interprets otherwise.
Autograd differentiates the function, and a model's entries can call it
(the example's `Classifier.loss`).

## PyTorch to Linnet

```python
from torch.export import Dim
from linnet.torch import export_linnet

export_linnet(
    model,
    example_args=(tokens,),
    dynamic_shapes={"tokens": {0: Dim("batch"), 1: Dim("seq")}},
    output="src/model.linnet",
    weights="weights/",          # optional: SafeTensors under the PyTorch names
)
```

`export_linnet` captures the model with `torch.export` and writes checked
Linnet source:

| PyTorch | Linnet |
| --- | --- |
| module tree | blocks; a `ModuleList` or `Sequential` of alike children becomes a sub array |
| forward graph | the root `entry` |
| dynamic dimensions | generics named after their `Dim`s |
| ATen operations | the primitive or library operation with the same meaning (`aten.mm` is `std.linalg::matmul`) |

An unmapped operation stops the export with its name. A parameter path
Linnet cannot spell is renamed, and `weights/bindings.json` records the
mapping.

## Tests

```bash
uv run pytest                       # references, round trips, state, training
LINNET_HF_TESTS=1 uv run pytest tests/torch/test_hf_checkpoints.py   # TinyLlama and GPT-2 vs transformers
uv run pyright
uv run ruff check src tests && uv run ruff format --check src tests
```
