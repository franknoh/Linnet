# 6. Tensor Algebra and Index Notation

## 6.1 Motivation

Linnet adopts the useful part of Einstein/einsum notation while rejecting string-encoded mini-languages and implicit reductions.

The canonical source syntax exposes free indices and reduction indices directly to the parser and type checker.

## 6.2 Tensor comprehension

A tensor comprehension is introduced by an indexed `let` binding:

```text
let c[m, n] =
    sum[k] a[m, k] * b[k, n]
```

The indices on the left-hand side are free output indices.

Reduction indices are explicitly introduced by a reduction expression.

## 6.3 No implicit summation

The following is invalid:

```text
let y[i] = a[i, j] * b[j]
```

because `j` is neither an output index nor explicitly bound by a reduction.

The valid form is:

```text
let y[i] = sum[j] a[i, j] * b[j]
```

## 6.4 Index domains

An index obtains its domain from tensor axes it indexes.

```text
let c[m, n] = sum[k] a[m, k] * b[k, n]
```

implies:

- `m` has the domain of `a` axis 0;
- `k` must have a single compatible domain for `a` axis 1 and `b` axis 0;
- `n` has the domain of `b` axis 1.

Conflicting domains are a static shape error.

## 6.5 Repeated indices in one tensor

Repeated indices in a single tensor expression select a diagonal; they do not imply reduction.

```text
let d[i] = a[i, i]
let t = sum[i] a[i, i]
```

The corresponding dimensions MUST be provably equal.

## 6.6 Shape-pack indices

A variadic index pack is written with lowercase `*name` inside a comprehension:

```text
let y[*s, o] =
    sum[i] x[*s, i] * weight[o, i]
```

The pack corresponds to a statically known sequence of tensor axes derived from a generic shape pack.

## 6.7 Reductions

Initial built-in reduction forms:

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

## 6.8 Accumulation dtype

A numeric reduction may explicitly specify accumulator dtype:

```text
sum<f32>[d] cast<f32>(x[d]) * cast<f32>(y[d])
```

Without an explicit accumulator dtype, accumulation uses the expression dtype.

The result dtype of `sum<T>` and `prod<T>` is `T` unless explicitly converted afterward.

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

Transpose-like remapping:

```text
let y[j, i] = x[i, j]
```

Trace:

```text
let t = sum[i] a[i, i]
```

Attention score contraction:

```text
let score[b, h, q, k] =
    sum<f32>[d]
        cast<f32>(query[b, h, q, d]) *
        cast<f32>(key[b, h, k, d])
```

## 6.10 Core IR lowering

Tensor comprehensions lower to a generalized contraction/reduction representation in Core Tensor IR. The source syntax does not prescribe loop order, memory layout, tiling, or backend kernel choice.

A backend or optimizer is free to choose any semantically equivalent implementation permitted by the active numeric-equivalence policy.

## 6.11 Compatibility `einsum`

A future compatibility library MAY provide a string-based `einsum` parser, but it is not canonical language syntax and MUST lower immediately to the same typed tensor-algebra representation.
