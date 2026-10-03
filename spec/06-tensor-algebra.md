# 6. Tensor algebra and index notation

## 6.1 Motivation

Index notation follows einsum notation, with indices as source syntax and every reduction explicit.

## 6.2 Tensor comprehension

An indexed `let` binding introduces a tensor comprehension:

```text
let c[m, n] =
    sum[k] a[m, k] * b[k, n]
```

Left-hand-side indices are free output indices. Only a reduction expression introduces reduction indices.

## 6.3 No implicit summation

Invalid, because `j` is neither an output index nor bound by a reduction:

```text
let y[i] = a[i, j] * b[j]
```

Valid:

```text
let y[i] = sum[j] a[i, j] * b[j]
```

## 6.4 Index domains

An index takes its domain from the tensor axes it indexes.

```text
let c[m, n] = sum[k] a[m, k] * b[k, n]
```

implies:

- `m` has the domain of `a` axis 0;
- `k` must have a single compatible domain for `a` axis 1 and `b` axis 0;
- `n` has the domain of `b` axis 1.

Conflicting domains are a static shape error.

An index position may hold a computed integer, such as `labels[b]` in `x[b, labels[b]]`. It reads the position it holds and gives its axis no domain (§5.8).

## 6.5 Repeated indices in one tensor

Repeated indices in one tensor expression select a diagonal, not a reduction:

```text
let d[i] = a[i, i]
let t = sum[i] a[i, i]
```

The corresponding dimensions MUST be provably equal.

## 6.6 Shape-pack indices

Inside a comprehension, a lowercase `*name` is a variadic index pack:

```text
let y[*s, o] =
    sum[i] x[*s, i] * weight[o, i]
```

The pack stands for the statically known axes of a generic shape pack.

## 6.7 Reductions

Built-in reductions:

```text
sum[i] expr
prod[i] expr
max[i] expr
min[i] expr
any[i] expr
all[i] expr
```

Multiple axes:

```text
sum[i, j] expr
```

The reduced expression extends as far right as possible: `x + sum[i] a[i] * b[i]` reduces the whole product.

Reduction names are contextual: `sum`, `prod`, `max`, `min`, `any`, and `all` begin a reduction only when immediately followed by an index list (`sum[i]`) or an accumulator dtype and index list (`sum<f32>[i]`). Elsewhere they are identifiers, so `max(A, B)` is a call, and a value with one of these names cannot be indexed directly.

## 6.8 Accumulation dtype

A numeric reduction may specify an accumulator dtype:

```text
sum<f32>[d] cast<f32>(x[d]) * cast<f32>(y[d])
```

Without one, accumulation uses the expression dtype. With one, the reduced expression MUST already have that dtype. `sum<T>` and `prod<T>` produce dtype `T`.

`sum`, `prod`, `max`, and `min` reduce `Numeric` values; `any` and `all` reduce `bool` values.

An index variable may appear only as a tensor index; use `iota(N)[i]` for the position itself.

## 6.9 Common examples

Matrix multiplication:

```text
let c[m, n] = sum[k] a[m, k] * b[k, n]
```

Batched matrix multiplication:

```text
let c[*b, m, n] = sum[k] a[*b, m, k] * b[*b, k, n]
```

Outer product:

```text
let c[i, j] = a[i] * b[j]
```

Transpose:

```text
let y[j, i] = x[i, j]
```

Trace:

```text
let t = sum[i] a[i, i]
```

Attention scores:

```text
let score[b, h, q, k] =
    sum<f32>[d]
        cast<f32>(query[b, h, q, d]) *
        cast<f32>(key[b, h, k, d])
```

## 6.10 Core IR lowering

Comprehensions lower to a contraction form in Core Tensor IR. Source does not prescribe loop order, memory layout, tiling, or kernels; a backend or optimizer may choose any equivalent implementation the active numeric-equivalence policy permits.

## 6.11 Compatibility `einsum`

A future compatibility library MAY provide a string-based `einsum` parser. It is not canonical syntax and MUST lower immediately to the same typed tensor-algebra representation.
