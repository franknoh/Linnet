# 7. Functions, ops, blocks, and entries

## 7.1 `fn`

A `fn` is a pure helper function:

```text
fn rotate_half<*S: Shape, D: Dim, T: Float>(
    x: Tensor[*S, D; T],
) -> Tensor[*S, D; T]
where D % 2 == 0 {
    ...
}
```

A compiler may inline, duplicate, eliminate, or otherwise transform a `fn` without keeping a semantic boundary.

Recursive functions are rejected (§8.6).

## 7.2 `op`

An `op` is a pure function with a stable semantic identity and a canonical decomposition:

```text
pub op linear<*S: Shape, In: Dim, Out: Dim, T: Float>(
    x: Tensor[*S, In; T],
    weight: Tensor[Out, In; T],
    bias: Tensor[Out; T]? = none,
) -> Tensor[*S, Out; T] {
    let y[*s, o] = sum[i] x[*s, i] * weight[o, i]

    return match bias {
        some(b) => y + b
        none    => y
    }
}
```

A backend MAY preserve the op, inline its body, or replace it with a proven equivalent native implementation. Unless declared `extern` (a future extension), the body is the normative fallback semantics.

An op MAY follow its body with a `grad` clause, its backward pass:

```text
pub op truncate<*S: Shape, T: Float>(x: Tensor[*S; T]) -> Tensor[*S; T] {
    return cast<T>(cast<i64>(x))
} grad(y, dy) {
    return dy
}
```

`grad` is a keyword only in this position. Its two names bind the op's result and the gradient arriving at it, both of the result's type; the op's generics and parameters are in scope. It returns the gradient of each parameter that is a tensor of a floating dtype or of a dtype generic bounded by `Float`, in declaration order: one value, or a tuple when there are several. Other parameters, floating scalars among them, take no gradient.

An op with `grad` MUST return one floating tensor or scalar, and its parameters MUST be tensors or scalars. Wherever an entry is differentiated, by a framework running exported code or by the compiler (§15.4), the clause replaces the backward pass of the op's body. The body still defines the op's value.

## 7.3 Semantic identity

An op's identity includes at least:

- package identity and semantic version context;
- module path;
- operation name;
- language-version interpretation.

Backends SHOULD NOT rely only on an unqualified name such as `attention`.

## 7.4 `block`

A block is a structural component:

```text
pub block Linear<In: Dim, Out: Dim, T: Float = bf16> {
    param weight: Tensor[Out, In; T]
    param bias: Tensor[Out; T]? = none

    pub fn forward<*S: Shape>(
        x: Tensor[*S, In; T],
    ) -> Tensor[*S, Out; T] {
        return linear(x, weight, bias)
    }
}
```

A block is not a runtime tensor value; it defines a namespace of parameters, state, and methods.

A block's `where` clause holds inside every member and method and MUST be provable wherever the block is instantiated, as in a `sub`:

```text
pub block Attention<H: Dim, Heads: Dim, T: Float>
where H % Heads == 0, Heads > 0 {
    sub q_proj: Linear<H, H, T>
    ...
}
```

## 7.5 Sub-blocks

```text
sub q_proj: Linear<Hidden, QDim, T>
sub layers: [DecoderLayer<Hidden, T>; Layers]
```

Nested block declarations define deterministic parameter paths (§9.5).

## 7.6 Entries

An entry is a public, backend-visible entry point:

```text
pub entry forward<B: Dim, S: Dim>(
    tokens: Tensor[B, S; i32],
) -> Tensor[B, S, Vocab; bf16] {
    ...
}
```

A block or module MAY expose multiple entries, such as `prefill` and `decode`.

A module-level entry, such as a loss, has no `self` and reads no parameters or state. A backend exports it alone, taking only its inputs. Other functions and entries MAY call it like any `fn`.

`forward` has no language meaning.

## 7.7 Generic defaults

A generic parameter may have a default where unambiguous:

```text
T: Float = bf16
```

The default MUST satisfy the declared constraint.

## 7.8 Overloading

A module cannot declare two functions or ops with the same name; there is no overloading by argument type.

## 7.9 Kernels

A `kernel` is a tile program: its body runs once per program of a launch grid, on tiles, small tensors of compile-time shapes. An op names a kernel that may compute it in place of its body:

```text
pub kernel row_softmax<R: Dim, C: Dim, B: Dim>(
    x: Tensor[R, C; f32],
) -> y: Tensor[R, C; f32] grid(R)
where B >= C {
    let row = program_id(0)
    let cols = iota<i32>(B)
    let mask = cols < C
    let v = load(x[row, cols], mask, -1e30)
    let e = exp(v - max[c] v[c])
    store(y[row, cols], e / sum[c] e[c], mask)
}

pub op softmax<R: Dim, C: Dim>(x: Tensor[R, C; f32]) -> Tensor[R, C; f32] kernel row_softmax<R, C, 1024> {
    ...
}
```

A kernel is declared at module level. Its parameters are tensors in memory or scalars; its named results are tensors in memory. Every tensor has a known rank: no shape pack. `grid(...)` gives one to three compile-time integers, the number of programs along each axis. `warps(n)` (a power of two) and `stages(n)`, integer literals after the grid, are launch hints: the warps a program runs on and the software-pipelining stages of its loops. They change no result.

In the body:

- `program_id(axis)` is the program's `i32` index along grid axis 0, 1, or 2.
- `load(x[i, j], mask, other)` reads a tensor parameter. Each index is an integer scalar or an integer tile, and each tile adds its axes to the result in order: with tiles `rows: [BM]` and `cols: [BN]`, `x[rows, cols]` is a `[BM, BN]` tile. Where the optional `bool` mask, of the result's shape or a scalar, is false, the element is `other` (default `0`) and memory is not read.
- `store(y[i, j], value, mask)` writes a result: `value` is a tile of the indexed shape or a scalar, written where the optional mask is true. `atomic_add`, `atomic_max` and `atomic_min`, of the same form, combine the value with what the result holds, atomically, so that programs may write the same elements; the result then starts at the operation's identity (zero, the lowest value, the highest). They write `f32`, `i32`, `u32`, `i64` or `u64`, and `atomic_add` also `f16`. A result takes one kind of write. These are the calls that stand as statements.
- A tensor parameter or result appears only in `load` or `store`. Tiles use the rest of the language: arithmetic, index notation, reductions, `iota`, `fill`, `for` and `while` loops, `fn` calls. A kernel returns nothing and is not called.

An op's `kernel name<args>` binds the kernel's generics in terms of the op's. The kernel takes the op's parameters, in order and of the same types, and writes what the op returns, one result or a tuple. A backend MAY launch the kernel in place of the op's body where the kernel's `where` clause holds for the bound generics; the body remains the op's definition. A kernel that disagrees with it is a program error the compiler cannot detect.

