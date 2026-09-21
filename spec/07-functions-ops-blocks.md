# 7. Functions, Semantic Operations, Blocks, and Entries

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

A compiler may inline, duplicate, eliminate, or otherwise transform a `fn` without preserving a semantic boundary.

Recursive functions are rejected in the initial language version.

## 7.2 `op`

An `op` is a pure function with a stable semantic identity and canonical decomposition:

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

A backend MAY preserve the op, inline its body, or replace it with a proven equivalent native implementation.

The op body is the normative fallback semantics unless the op is explicitly declared `extern` in a future extension.

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

A block is not itself a runtime tensor value. It defines a parameter/state namespace and callable methods.

## 7.5 Sub-blocks

```text
sub q_proj: Linear<Hidden, QDim, T>
sub layers: [DecoderLayer<Hidden, T>; Layers]
```

Nested block declarations define deterministic parameter paths.

## 7.6 Entries

An entry is a public callable intended as a backend-visible model/program entry point:

```text
pub entry forward<B: Dim, S: Dim>(
    tokens: Tensor[B, S; i32],
) -> Tensor[B, S, Vocab; bf16] {
    ...
}
```

A block or module MAY expose multiple entries such as `prefill`, `decode`, or `classify`.

The word `forward` has no intrinsic language meaning.

## 7.7 Generic defaults

Generic parameters may define defaults where unambiguous:

```text
T: Float = bf16
```

A default generic parameter MUST satisfy its declared constraint.

## 7.8 Overloading

User-defined name overloading by argument type is not part of the initial language version. A module cannot declare multiple functions or ops with the same name.

This restriction simplifies diagnostics, tooling, import resolution, and backend semantic IDs.
