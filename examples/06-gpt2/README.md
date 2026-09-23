# GPT-2

GPT-2 with its distinctive details written out: learned positions looked up
by `iota`, LayerNorm with a bias passed as an optional, a fused QKV projection
split by slicing with dimension arithmetic, the tanh GELU, and an output head
tied to the token embedding through a contraction.

## Positions from `iota`

```linnet
let x0 = embedding(tokens, wte) + embedding(iota<i32>(S), wpe)
```

Both lookups are the same library operation; `where S <= MaxPositions` on
the entry is what makes the second one in bounds.

## Slicing with dimension arithmetic

The QKV projection is one `Linear<H, 3 * H, T>`; `projected[:, :, H:2 * H]`
selects the keys. Slice bounds are dimension expressions the checker
evaluates, so the result type is `Tensor[B, S, H; T]` exactly.

## A tied head

The logits are `sum[i] h[b, s, i] * wte[v, i]` — the embedding matrix used
as the output projection, in index notation rather than a transpose.

## Run it

```bash
linnet check --std stdlib examples/06-gpt2/gpt2.linnet
linnet onnx --std stdlib --bind Vocab=50257 --bind MaxPositions=1024 --bind H=768 \
            --bind Heads=12 --bind Layers=12 --bind T=f32 --bind B=1 --bind S=64 \
            examples/06-gpt2/gpt2.linnet > gpt2.onnx.txt
```
