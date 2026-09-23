# Coming from PyTorch

Three things change: shapes are checked before anything runs, source never
carries weights, and one file runs in PyTorch, JAX, XLA, and ONNX Runtime.
The operations, the parameter names, and the weights on disk stay the same.

## The same block, twice

::: code-group

```python [PyTorch]
class Block(nn.Module):
    def __init__(self, h, heads):
        super().__init__()
        self.norm = RMSNorm(h)
        self.qkv = nn.Linear(h, 3 * h, bias=False)
        self.out = nn.Linear(h, h, bias=False)
        self.heads = heads

    def forward(self, x):
        b, s, h = x.shape
        q, k, v = self.qkv(self.norm(x)).split(h, dim=-1)
        q = q.view(b, s, self.heads, -1).transpose(1, 2)
        k = k.view(b, s, self.heads, -1).transpose(1, 2)
        v = v.view(b, s, self.heads, -1).transpose(1, 2)
        mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return x + self.out(mixed.transpose(1, 2).reshape(b, s, h))
```

```linnet [Linnet]
pub block Block<H: Dim, Heads: Dim, T: Float = bf16>
where
    Heads > 0,
    H % Heads == 0
{
    sub norm: RmsNorm<H, T>
    sub qkv: Linear<H, 3 * H, T>
    sub out: Linear<H, H, T>

    pub entry forward<B: Dim, S: Dim>(x: Tensor[B, S, H; T]) -> Tensor[B, S, H; T] {
        let projected = qkv.forward(norm.forward(x))
        let q = heads<B, S, Heads, H / Heads, T>(projected[:, :, 0:H])
        let k = heads<B, S, Heads, H / Heads, T>(projected[:, :, H:2 * H])
        let v = heads<B, S, Heads, H / Heads, T>(projected[:, :, 2 * H:3 * H])
        let mixed = attention(q, k, v, rsqrt(cast<f32>(H / Heads)), some(causal_mask<S, S>()))
        return x + out.forward(reshape(permute(mixed, [0, 2, 1, 3]), [B, S, H]))
    }
}
```

:::

| PyTorch | Linnet |
| --- | --- |
| `b, s, h = x.shape` at runtime | `B`, `S`, `H` are generics; every result shape is derived from them |
| `.view(..., -1)` | `reshape` to a shape the checker proves from `H % Heads == 0` |
| `__init__` builds submodules | `sub norm: RmsNorm<H, T>` declares one; nothing allocates |
| `state_dict()` keys | the same names: `norm.weight`, `qkv.weight`, `out.weight` |
| `F.scaled_dot_product_attention` | `attention` from `stdlib/`, itself readable Linnet |

## Running it

```python
from linnet_torch import load

model = load("src/block.linnet",
             generics={"H": 1024, "Heads": 16, "T": "bf16"},
             weights="checkpoints/block/",     # SafeTensors named by parameter path
             device="cuda",
             numerics="equivalent",            # PyTorch kernels for library operations
             compile="inductor")               # generated source under torch.compile
```

`print(model)` shows a module tree with the Linnet block names and shapes;
`model.state_dict()` uses the Linnet paths, so checkpoints move in both
directions. `trainable=True` turns on gradients and any `torch.optim`
optimizer trains the model. The [PyTorch](/docs/torch) page has the details;
[JAX and Flax](/docs/jax) covers `load`, `load_source`, and NNX.

## Bringing a model over

```python
from linnet_torch import export_linnet

export_linnet(module, (example_input,), output="src/model.linnet", weights="weights/")
```

`export_linnet` traces with `torch.export` and writes Linnet: children become
`sub`s, `ModuleList`s become sub arrays, parameters keep their names, and the
decompositions PyTorch produces (softmax, layer norm, RMS norm, GELU, SiLU)
come back as library calls. The result is formatted and checked, or the
export fails naming the operation it could not express. ONNX models import
with `linnet_onnx.import_onnx`, JAX functions with `linnet_jax.export_linnet`.

## What you get

| | `nn.Module` | Linnet |
| --- | --- | --- |
| Shape errors | at runtime, on the first batch that reaches them | at check time, both shapes named |
| Head split `H / Heads` | `view(..., -1)`, a wrong divisor reshapes silently | proved from `H % Heads == 0` or rejected |
| Loading a model | runs the model's Python; `pickle` in `.pt` files | `linnet check` runs nothing; weights are SafeTensors by path |
| Code and weights | entangled or by convention | separate by construction |
| Other frameworks | rewrite or export a frozen graph | same source: `linnet stablehlo`, `linnet onnx`, `linnet_jax.load` |
| Library code | opaque kernels | source in `stdlib/`, checked like yours |
| Performance | native kernels | the same kernels from generated source (2.1 ms vs 2.7 ms for the compiled reference, small Llama on an H100), or XLA (0.66 ms); see [Benchmarks](/benchmarks) |
| Refactoring | search and hope | rename through the language server; every use is typed |

The trade: no Python inside the model and no data-dependent shapes. Loops
are `while` over scalars or compile-time `static for`.

## Where things live

| PyTorch | Linnet |
| --- | --- |
| `nn.Module` subclass | `block` |
| `__init__` creating submodules | `sub name: Block<...>` |
| `nn.Parameter` | `param name: Tensor[...; T]` |
| `register_buffer` | `buffer`, or `state` for caches the block updates |
| `forward` | `pub entry forward<...>(...)`; any number of entries |
| `nn.ModuleList` | `sub layers: [Layer<...>; N]` and `static for layer in layers` |
| `x @ w.T`, `einsum` | `matmul`, or index notation `sum[k] x[i, k] * w[j, k]` |
| `F.softmax`, `F.layer_norm`, ... | `std.nn.softmax::softmax`, `std.nn.norm::layer_norm`, ... |
| `state_dict()` keys | parameter paths (`linnet inspect --parameters`) |
| `torch.export`, ONNX export | `linnet stablehlo`, `linnet onnx`, `linnet torch`, `linnet plan` |

Next: the [Quickstart](/docs/getting-started) writes and runs a first model;
the [Language tour](/docs/language-tour) covers the rest.
