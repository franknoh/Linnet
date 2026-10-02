# PyTorch

`linnet.torch` turns a Linnet root block into a `torch.nn.Module` and
a `torch.nn.Module` into Linnet source. The adapter knows nothing about any
model; everything comes from the compiler.

```bash
cd python/linnet && uv sync --extra torch    # or: pip install ".[torch]"
export LINNET_BIN=/path/to/build/release/linnet     # or put `linnet` on PATH
```

## Loading a model

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

The result is an ordinary module. Printing it shows the block hierarchy with
the Linnet names, shapes, and dtypes:

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
uses the Linnet parameter paths (`layers.0.attention.q_proj.weight`), so a
checkpoint written from PyTorch loads here and back.

What `load` does:

1. Runs `linnet plan`, which checks the program and prints its structure,
   parameter manifest, and Core IR. Nothing in the package is executed.
2. Builds the module tree: `sub` becomes a child module (`ModuleList` for
   arrays), `param` an `nn.Parameter`, `buffer` a buffer, `state` a
   non-persistent buffer starting at zero.
3. With `weights`, checks every tensor's name, shape, and dtype against the
   manifest, then copies the data. Optional parameters may be absent.
4. Exposes entries as methods. `forward` is the entry of that name or the
   only one; entry generics such as `B` and `S` are bound from the inputs.

| Option | |
| --- | --- |
| `numerics="fast"` (default) | the kernels PyTorch's own reference implementations use: layer normalization and attention in the input dtype (attention under an explicit mask, compiled for CUDA, as FlexAttention over only the key blocks the mask reaches: a serving step's rows at their own lengths, packed prompts that each see only themselves, a sliding window), and a mixture of experts' products for the experts each token chose (`std.nn.moe::linear_experts`, and `linear_experts_shared` and `combine_experts`, whose bodies are the dense form a prompt's rows can take) as one `torch._grouped_mm` over the tokens sorted by expert (CUDA, compute capability 9 or later, bf16), which reads each chosen expert's weight where it lies rather than gathering it |
| `numerics="equivalent"` | the same kernels, with the f32 accumulation the canonical bodies specify; on `bf16` this costs the fused attention kernel and about 1.6x |
| | the kernels: `F.conv1d` and `F.conv2d` (square or `conv2d_rect`'s rectangular windows, with the stride and padding the call's generics give), `F.batch_norm`, `F.linear`, `scaled_dot_product_attention` with `is_causal` for a square `causal_mask` and `enable_gqa` for `grouped_attention`, `torch.softmax`, `torch.rms_norm`, `F.layer_norm`, activations, `torch.matmul`; `F.embedding` and the causal mask are exact and selected under every policy |
| `numerics="exact"` | every library operation runs as its canonical `.linnet` body: the slowest path, for comparing against |
| | `torch.softmax` and `torch.rms_norm` are in neither tier: their kernels accumulate in f32 whatever the input dtype, so every policy calls them without casts |
| `compile=True` | entries run as PyTorch source from `linnet torch`, generated once per entry and input shape; input-independent values (rotary tables, masks) are computed once and reused. The default on CUDA; `compile=False` is the interpreter, the default elsewhere |
| `compile="inductor"` | the same, passed through `torch.compile` |
| `compile="reduce-overhead"` | CUDA graphs, for decoding: the first call runs through `torch.compile`, the second captures the whole step as one graph, and every later call copies its inputs into the graph's and replays it, with no per-argument checks. A KV cache the entry writes is written in place, at an address the graph keeps; a state or weight replaced since (`reset_state`) is captured again. Split over a mesh (`tensor_parallel`), each process captures its step with the collectives in it. With blocks placed on several devices, or with gradients, `torch.compile`'s own CUDA graphs run instead |
| `trainable=True` | parameters require gradients |
| `bindings="bindings.json"` | maps Linnet paths to checkpoint tensor names |
| `cast_dtype=True` | converts floating-point weights to the model's dtype as they are read: an f32 checkpoint in a bf16 model, or the reverse |
| `amp="bf16"` (or `"f16"`) | mixed precision: weights stay in the model's dtype (`T=f32` keeps f32 masters) and entries run under `torch.autocast`, so products, convolutions, and attention compute in 16 bits; gradients arrive in f32. With `"f16"`, scale the loss with `torch.amp.GradScaler` |
| `device_map="auto"` | spreads the model over the visible GPUs and streams what does not fit from the host; see below |

Entries with generics the inputs do not determine take them by name:
`model.run_entry("generate", [prompt, pos], generics={"Steps": 16})`.
`run_entry(..., compile=...)` overrides the module's setting for one call,
so a decoding step can replay as a CUDA graph while prompts of many lengths
run without one.
`model.reset_state()` zeroes every `state` member; `model.state_paths()`
lists them; `model.generated_source(entry)` shows the code `compile` ran.

Linear layers that read the same input with weights of their own -- a
layer's query, key, and value projections, its gate and up -- run as one
product over their weights side by side, sliced after: at a decoding step's
few rows, one kernel instead of three. The joined weight is prepared once,
and the runtime first lays each group's weights out one after another in
one buffer, so it is a view of them and costs no memory. This happens for a
model loaded for inference on one device; training keeps the layers apart.

## More than one GPU, and more than the GPU holds

```python
model = linnet.torch.load("model.linnet", generics=generics, weights="weights/",
                          device_map="auto")
print(model.placement.describe())
```

```text
cuda:0: embedding, layers.0-15
cuda:1: layers.16-31, norm, head
```

A model is divided into units: every sub-block of the root, and every
element of a sub-block array. The manifest gives each unit's size before any
weight is read, so the placement is decided up front. Units fill the first
GPU in the order they are declared, then the next; each GPU may use what is
free on it, less a reserve for activations, and `max_memory={0: "20GiB"}`
caps one. When the GPUs are full, the remaining units stay in pinned host
memory and are *offloaded*: each is copied to the last GPU when it runs and
dropped when it returns, so a model larger than the GPU still runs, at the
speed of the copies. `offload=False` makes that an error instead, and a
mapping such as `{"layers.30": "cpu", "layers.31": "cpu"}` places units by
hand.

The placement is compiled, not hooked in at run time. `linnet torch --place
layers.16=1 --offload layers.31` generates the same straight-line source as
an unplaced model, with every operation on its unit's device and a `.to()`
exactly where a value crosses devices:

```python
    v212 = v209.to(_dev[1], non_blocking=True)   # the residual stream, once
    ...
    v480 = p291.to(_dev[1], non_blocking=True)   # an offloaded layer's weights
    v481 = F.linear(v479, v480, None)
    ...
    del v480                                     # gone when the layer returns
```

Placement needs the generated path (it is always used with `device_map`), and
CUDA graphs cannot replay streamed weights, so `compile="reduce-overhead"`
requires `offload=False`.

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

`device_map` puts whole blocks on different GPUs, which then take turns;
tensor parallelism splits every large weight across them, so they work on
each layer at once.

A model can say how it splits. The zoo's Llama-shaped decoders have a
`Shards` generic (1 by default): one shard holds `Heads / Shards` query
heads, `KvHeads / Shards` key and value heads, and `Inner / Shards` hidden
units, and the projections back out of them end in
`std.nn.parallel::all_reduce`, the sum over the shards, which on one device
is the value itself. Those with an output projection of their own (not tied
to the embedding) split it too: each shard holds `Vocab / Shards` rows of
it, and `std.nn.parallel::all_gather` sets the shards' slices of the logits
side by side, which on one device is the one slice, the whole. With
`tensor_parallel=mesh`, each process runs the model with `Shards` bound to
the mesh size on its own part of every weight the checkpoint holds `Shards`
times over, read from the checkpoint alone along the one axis that differs;
`all_reduce` sums across the processes and `all_gather` gathers from them.
A sum of at most 64 KiB -- a decoding step's, one token's hidden state -- is
one Triton kernel over symmetric memory, into which each process writes its
part and from which it reads its peers', in about half the time NCCL takes
for a message that small; larger ones go to NCCL. The entries are the
generated code of one shard on ordinary tensors, so a shard's query, key and
value projections still join into one product (and gate and up into
another), and the decoding step still replays as one CUDA graph with its
all-reduces inside.

A model without `Shards` is split from the outside: each
process of a `torch.distributed` job holds its slice of the weights as DTensors: the
projections into the heads and the feed-forward width split by output, the
projections back by input, the KV caches by heads, and everything else
copied to each (`linnet.parallel.DEFAULT_RULES`; `tp_rules={"*.experts.*": 0}`
adds or overrides rules by path pattern). Every process runs the same
entries and DTensor adds the collectives the splits need; results come back
whole on each. Any split computes the same numbers -- the rules only decide
how much crosses between GPUs -- and an axis a device count does not divide
is copied instead. Sibling linear layers stay apart, since joining split
weights would gather them onto every GPU. With `compile="reduce-overhead"`
the decoding step replays as one CUDA graph per process, all-reduces
included; those graphs hold the process group's communicators, so a
process lets go of the model (`del model`) before
`dist.destroy_process_group()`, which otherwise waits on them.

## Training

```python
model = load("src/model.linnet", generics={...}, weights="init/", trainable=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = criterion(model(tokens), labels)
loss.backward()
optimizer.step()
```

Entries are ordinary differentiable PyTorch arithmetic, interpreted or
generated, so autograd needs nothing else. `state` members are detached
between calls: they carry values, not gradients.

## Functions

An `entry` declared at module level, outside any block, is a function of
its inputs alone: a loss, a preprocessing step, a reward
(`examples/09-functions`). `load_function` returns one as a PyTorch
function.

```python
from linnet.torch import load, load_function

model = load("functions.linnet", generics={"In": 64, "Classes": 10}, trainable=True)
cross_entropy = load_function("functions.linnet", "cross_entropy")
loss = cross_entropy(model(x), labels)
loss.backward()
```

Its generics are bound from each call's inputs, dimensions from shapes and
dtype generics from dtypes, so one function serves every batch size and
dtype. One the inputs leave open is given by name (`positions(offset,
N=8)`), and a Python number passes for a scalar input. `compile=` works as
for `load`: source generated per binding of the generics (through
`torch.compile` or as CUDA graphs when a backend is named), or the
interpreter; unset, it is generated for inputs on a CUDA device. Either way
autograd differentiates the function, so a loss written in Linnet trains a
model as `torch.nn.functional`'s would. A model's own entries can call the
same functions -- the example's `Classifier.loss` does -- which keeps a
training objective inside the model's program.

## Exporting a PyTorch model

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

`export_linnet` captures the model with `torch.export`, decomposes it to core
ATen, and translates it into a plan that `linnet emit` prints as source. The
module tree becomes blocks (`ModuleList` and `Sequential` of alike children
become sub arrays), the forward graph becomes the root `entry`, dynamic
dimensions become generics named after their `Dim`s, and each ATen operation
becomes the primitive or standard-library operation with the same meaning
(`aten.mm` is `std.linalg::matmul`, `_softmax` is `softmax`,
`native_layer_norm` is `layer_norm`). The result is checked before it is
written.

The translation never guesses: an unmapped operation stops the export with
its name. Parameter paths follow PyTorch's; when one cannot be spelled in
Linnet, the member is renamed and `weights/bindings.json` records the
mapping. Exporting runs Python with the model loaded; checking the result
never does.

## Tests

```bash
uv run pytest                       # references, round trips, state, training
LINNET_HF_TESTS=1 uv run pytest tests/torch/test_hf_checkpoints.py   # TinyLlama and GPT-2 vs transformers
uv run pyright
uv run ruff check src tests && uv run ruff format --check src tests
```
