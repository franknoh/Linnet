# Coming from PyTorch

Three things change: shapes are checked before anything runs, source never
carries weights, and one file runs in PyTorch, JAX, XLA and ONNX Runtime.
Operations, parameter names and weights on disk stay the same.

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
| `b, s, h = x.shape` at runtime | `B`, `S`, `H` are generics; every shape derives from them |
| `.view(..., -1)`: a wrong divisor reshapes silently | `reshape` to a shape proved from `H % Heads == 0`, or rejected |
| `__init__` builds submodules | `sub norm: RmsNorm<H, T>` declares one; nothing allocates |
| `state_dict()` keys | the same names: `norm.weight`, `qkv.weight`, `out.weight` |
| `F.scaled_dot_product_attention` | `attention` from `stdlib/`, written in Linnet |

## Running it

```python
from linnet.torch import load

model = load("src/block.linnet",
             generics={"H": 1024, "Heads": 16, "T": "bf16"},
             weights="checkpoints/block/",     # SafeTensors named by parameter path
             device="cuda",
             numerics="equivalent",            # PyTorch kernels for library operations
             compile="inductor")               # generated source under torch.compile
```

`numerics="fast"` runs softmax, normalization and attention in the input
dtype, as PyTorch reference models do on `bf16`. `trainable=True` lets any
`torch.optim` optimizer train the model. See [PyTorch](/docs/torch) and
[JAX and Flax](/docs/jax).

## Importing a model

```python
from linnet.torch import export_linnet

export_linnet(module, (example_input,), output="src/model.linnet", weights="weights/")
```

`export_linnet` traces with `torch.export` and writes checked Linnet that
keeps the parameter names, or fails naming the operation it cannot express.
For ONNX and JAX, use `linnet.onnx.import_onnx` and `linnet.jax.export_linnet`.

## What you get

| | `nn.Module` | Linnet |
| --- | --- | --- |
| Shape errors | at runtime, on the first batch that reaches them | at check time, both shapes named |
| Loading a model | runs the model's Python; `pickle` in `.pt` files | `linnet check` runs nothing; weights are SafeTensors by path |
| Performance | native kernels | the same kernels from generated source, replayable as CUDA graphs, or XLA (Llama 3.1 8B decodes one request at 167 tok/s on an H100, transformers compiled at 110); see [Benchmarks](/benchmarks) |

The trade: no Python inside the model and no data-dependent shapes. Loops
are `while` over scalars or compile-time `static for`.

## Where things live

| PyTorch | Linnet |
| --- | --- |
| `nn.Module` subclass | `block` |
| `nn.Parameter` | `param name: Tensor[...; T]` |
| `register_buffer` | `buffer`, or `state` for caches the block updates |
| `forward` | `pub entry forward<...>(...)`; any number of entries |
| `nn.ModuleList` | `sub layers: [Layer<...>; N]` and `static for layer in layers` |
| `x @ w.T`, `einsum` | `matmul`, or index notation `sum[k] x[i, k] * w[j, k]` |
| `F.softmax`, `F.layer_norm`, ... | `std.nn.softmax::softmax`, `std.nn.norm::layer_norm`, ... |
| `torch.export`, ONNX export | `linnet stablehlo`, `linnet onnx`, `linnet torch`, `linnet plan` |

Next: the [Quickstart](/docs/getting-started), then the
[Language tour](/docs/language-tour).
