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
    "examples/09-tiny-transformer/src/lib.linnet",
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
   `buffer` a buffer. `state_dict()` therefore uses the Linnet parameter paths
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
`scaled_dot_product_attention`, `torch.softmax`, `torch.rms_norm`, the
activations, `torch.matmul`. Each one agrees with the canonical definition up
to floating-point rounding, and the differential tests in `tests/test_native.py`
check that against the decompositions. `linnet explain --numerics equivalent`
shows what would be selected and why.

Generic arguments of the root block with defaults (`T: Float = bf16`) may be
omitted. Root blocks with shape-pack generics are not supported.

## Tests

```bash
uv run pytest      # round trips against hand-written PyTorch references
uv run pyright     # strict
uv run ruff check src tests && uv run ruff format --check src tests
```
