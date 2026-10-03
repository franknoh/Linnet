# GPT-2

GPT-2 with its distinctive details written out: learned positions, LayerNorm
with a bias passed as an optional, a fused QKV projection, the tanh GELU,
and an output head tied to the token embedding.

## Commands

```bash
linnet check --std stdlib examples/06-gpt2/gpt2.linnet
linnet onnx --std stdlib --bind Vocab=50257 --bind MaxPositions=1024 --bind H=768 \
            --bind Heads=12 --bind Layers=12 --bind T=f32 --bind B=1 --bind S=64 \
            examples/06-gpt2/gpt2.linnet > gpt2.onnx.txt
```

## Key ideas

### Positions from `iota`

```linnet
let x0 = embedding(tokens, wte) + embedding(iota<i32>(S), wpe)
```

Both lookups are the same library op. `where S <= MaxPositions` on
`Model.forward` keeps the second one in bounds.

### Dimension arithmetic

The `qkv` projection is one `Conv1D<H, 3 * H, T>`, and
`projected[:, :, H:2 * H]` selects the keys. The checker evaluates slice
bounds as dimension expressions, so the result type is exactly
`Tensor[B, S, H; T]`.

### Tied output head

The logits are `sum[i] h[b, s, i] * wte[v, i]`: the embedding matrix used
as the output projection, in index notation rather than a transpose.
