# Running Linnet models in PyTorch

`python/linnet_torch` materializes a Linnet root block as a `torch.nn.Module`.

```bash
cd python/linnet_torch
uv sync
export LINNET_BIN=/path/to/build/release/linnet   # or put `linnet` on PATH
```

```python
from linnet_torch import load

model = load(
    "examples/04-tiny-transformer/src/lib.linnet",
    root="Model",
    generics={"Vocab": 32000, "H": 4096, "Heads": 32, "Inner": 11008, "Layers": 32, "T": "bf16"},
    std_root="stdlib",
    weights="weights/",          # a .safetensors file or a directory of them
    bindings="bindings.json",    # optional: Linnet path -> checkpoint tensor name
)
logits = model(tokens, cos_table, sin_table)
```

What `load` does:

1. Runs `linnet plan`, which checks the program and emits the plan: the root
   block's structure, its parameter manifest, and the Core IR of every function.
   Checking never executes anything from the package.
2. Builds the module hierarchy from the block structure: each `sub` is a child
   module (a `ModuleList` for arrays), each `param` an `nn.Parameter`, each
   `buffer` a buffer, each `state` a non-persistent buffer starting at zero:
   the block's assignments update it during a call, it is kept for the next
   call, `model.reset_state()` zeroes it, and `model.state_paths()` lists it.
   State is never read from or written to the weights. `state_dict()` therefore uses the Linnet parameter paths
   (`layers.0.attention.q_proj.weight`).
3. With `weights`, reads the SafeTensors metadata, checks every required
   tensor's presence, shape, and dtype against the plan, and only then copies
   the data. An optional parameter (`Tensor[...]? = none`) may be absent.
4. Entries become methods; `forward` is the entry named `forward` or the only
   one. Entry generics such as `B` and `S` are bound from the input shapes on
   every call, and the inputs are checked against the declared types.

Evaluation interprets Core IR with PyTorch operations. Index notation is
evaluated on index grids exactly as specified. With the default
`numerics="exact"`, semantic operations run through their canonical `.linnet`
decompositions: this is the correctness path that everything else must agree
with, not a fast implementation.

`numerics="equivalent"` lets the compiler select PyTorch library calls for the
standard library's semantic operations — `torch.nn.functional.linear`,
`scaled_dot_product_attention`, `torch.softmax`, `torch.rms_norm`,
`torch.nn.functional.layer_norm`, the
activations, `torch.matmul`. Each one agrees with the canonical definition up
to floating-point rounding, and the differential tests in `tests/test_native.py`
check that against the decompositions. `linnet explain --numerics equivalent`
shows what would be selected and why.

Generic arguments of the root block with defaults (`T: Float = bf16`) may be
omitted. Root blocks with shape-pack generics are not supported.

## Generated PyTorch source

```python
model = load("src/model.linnet", generics={...}, weights="weights/", compile=True)
fast = load("src/model.linnet", generics={...}, weights="weights/",
            numerics="equivalent", compile="inductor")
```

With `compile`, every entry runs as PyTorch source that `linnet torch`
generates for the input shapes of the first call (one compilation per entry
and shape, cached): straight-line code with the parameters and `state`
members as arguments, no interpreter in the loop. `numerics="equivalent"`
turns library operations into their PyTorch kernels in that source
(`torch.rms_norm`, `F.scaled_dot_product_attention`, ...), and a backend
name such as `"inductor"` passes each generated function through
`torch.compile`. The module hierarchy, weights, `state_dict()`, and
`reset_state()` are the same as without `compile`;
`model.generated_source(entry)` shows the code that ran.

## PyTorch to Linnet

The other direction starts from a live `torch.nn.Module`:

```python
from torch.export import Dim
from linnet_torch import export_linnet

export_linnet(
    model,
    example_args=(tokens,),
    dynamic_shapes={"tokens": {0: Dim("batch"), 1: Dim("seq")}},
    output="src/model.linnet",
    weights="weights/",   # optional: SafeTensors under the PyTorch names
)
```

`export_linnet` captures the model with `torch.export`, decomposes the graph
to core ATen operations, and translates it into a Core IR plan: the module
tree becomes blocks (`param`, `buffer`, and `sub` members; a `ModuleList` or
`Sequential` of identical children becomes a sub array), the forward graph
becomes the root block's `entry`, dimensions marked dynamic become the entry's
generic parameters named after their `Dim`s, and each ATen operation becomes
the Linnet primitive or standard-library operation with the same meaning —
`aten.mm` is `std.linalg::matmul`, `aten._softmax` is `std.nn.softmax::softmax`,
`native_layer_norm` is `std.nn.norm::layer_norm`, reductions and `embedding`
become index notation, views become `reshape` and `permute`. `linnet emit` then prints the plan as formatted source, and the
result is checked before it is written.

The translation never guesses: an operation without a mapping stops the export
with a message naming every such operation. Recovering higher-level structure
(recognizing an attention block, folding an unrolled `ModuleList` back into a
`static for`) is not attempted; the output is the graph as captured, readable
and deterministic.

Parameter paths follow PyTorch's (`layers.0.q.weight`), so weights bind by
name. When a path cannot be spelled in Linnet (a `Sequential` whose children
differ, or an attribute named like a keyword), the member is renamed and
`weights/bindings.json` maps the Linnet path back to the PyTorch name;
`load(..., bindings=...)` reads it. Exporting requires a trusted Python process
with the model loaded; `linnet check` on the result never runs Python.

## Tests

```bash
uv run pytest      # round trips against hand-written PyTorch references
uv run pyright     # strict
uv run ruff check src tests && uv run ruff format --check src tests
```
