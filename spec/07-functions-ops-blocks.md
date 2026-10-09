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
