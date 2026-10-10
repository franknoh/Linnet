# Kernels

Write a tile kernel in Linnet and let an op run it in place of its body.
Generated PyTorch launches it with Triton, generated JAX with Pallas. Every
other backend runs the op's body, which stays the op's definition.

```linnet
pub kernel row_softmax<R: Dim, C: Dim, B: Dim>(x: Tensor[R, C; f32]) -> y: Tensor[R, C; f32] grid(R)
where B >= C {
    let row = program_id(0)
    let cols = iota<i32>(B)
    let mask = cols < C
    let v = load(x[row, cols], mask, -1e30)
    let e = exp(v - max[c] v[c])
    store(y[row, cols], e / sum[c] e[c], mask)
}

pub op softmax<R: Dim, C: Dim>(
    x: Tensor[R, C; f32],
) -> Tensor[R, C; f32] kernel row_softmax<R, C, 1024> {
    let top[r] = max[c] x[r, c]
    let e[r, c] = exp(x[r, c] - top[r])
    let total[r] = sum[c] e[r, c]
    let y[r, c] = e[r, c] / total[r]
    return y
}
```

The body runs once per program of the grid, on tiles: small tensors whose
shapes are known when the kernel is compiled.

| In a kernel | Meaning |
| --- | --- |
| `-> y: T grid(a, b)` | named results; one to three compile-time grid sizes |
| `warps(4) stages(3)` | optional launch hints after the grid: warps a program, pipelining stages |
| `program_id(axis)` | the program's `i32` index along a grid axis |
| `load(x[i, j], mask, other)` | read a tensor parameter; `mask` and `other` (default `0`) are optional |
| `store(y[i, j], value, mask)` | write a result; `value` is a tile or a scalar |
| `atomic_add`, `atomic_max`, `atomic_min` | write the same way, combining atomically with what is there; the result starts at zero, the lowest, or the highest value |
| everything else | arithmetic, index notation, reductions, `iota`, `fill`, `for` loops, `fn` calls |

Each index is an integer or an integer tile, and each tile adds its axes in
order: `x[rows, cols]` with `rows: [BM]` and `cols: [BN]` is a `[BM, BN]`
tile. Tensor parameters and results appear only in `load`, `store`, and
the atomics; a result takes one kind of write. Atomics write `f32`, `i32`,
`u32`, `i64` and `u64` (and `f16` adds), so programs can share outputs:
column sums one row a program, split-K products, histograms.

The op names the kernel in its header, `kernel name<args>`, binding the
kernel's generics. The compiler checks that the kernel takes the op's
parameters and writes what the op returns.

## Where it runs

| Backend | Runs |
| --- | --- |
| generated PyTorch (`linnet torch`, `load(compile=True)`) | the kernel through `@triton.jit`, on CUDA tensors |
| generated JAX (`linnet jax`, `load_function`, `load_source`) | the kernel through `pl.pallas_call`, on a GPU |
| either on the CPU, or without Triton or Pallas | the op's body |
| ONNX, StableHLO, `--grad`, the PyTorch interpreter | the op's body |

The body also runs where the kernel's `where` clause does not hold for the
shapes (above, rows longer than 1024), and under `--numerics exact`.
`TRITON_INTERPRET=1` and `LINNET_PALLAS_INTERPRET=1` run kernels on the
CPU in Triton's and Pallas's interpreters, for tests.

## Gradients

Under autograd and `jax.grad`, a kernel's backward pass is the op's `grad`
clause. Without one, it differentiates the op's body, run again.

## Limits

- Tile sizes are powers of two.
- Tiles cannot be sliced, joined, or gathered from; `load` the memory
  instead. `prod` over a tile is not supported yet.
- A product of two tiles at least 16 a side becomes `tl.dot` (`jnp.dot` in
  Pallas), in `f32` at full precision unless `--numerics fast`.
- Inputs are made contiguous, so strides are constants.
- `@triton.jit` reads its function's source: import generated PyTorch from
  a file, as the runtimes do, rather than `exec` it from a string.
