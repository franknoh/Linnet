# Examples

Every example checks with `linnet lint --std stdlib examples/<name>`, formats
cleanly, materializes in PyTorch and JAX, and exports to StableHLO; the
tests under `python/linnet_torch/tests` compare the model examples against
hand-written PyTorch references.

| Example | What it shows |
| --- | --- |
| `01-linear` | A generic `op` in index notation and the block that owns its weights. |
| `04-attention` | Softmax and attention written as comprehensions with shape packs. |
| `05-block-and-weights` | A block hierarchy, sub arrays, `static for`, and the parameter manifest. |
| `09-tiny-transformer` | A package built only from the standard library, with rotary tables as inputs. |
| `10-llama` | A Llama-style decoder as a multi-module package: `crate.` imports, a `pub const` base frequency, rotary tables computed at compile time from `iota`, grouped-query attention whose `where` clause states the divisibility the reshapes rely on, `broadcast_to` for repeating key/value heads, two entries (`forward` and `next_token`, which slices the last position with `S - 1`), and a dtype generic defaulting to `bf16`. |
| `11-gpt2` | GPT-2: learned positions looked up with `embedding(iota<i32>(S), wpe)` under `where S <= MaxPositions`, LayerNorm with a bias passed as an optional, a fused QKV projection split by slicing with dimension arithmetic (`H:2 * H`), the tanh GELU, and an output head tied to the token embedding through a contraction. |
| `12-vit` | A Vision Transformer: patches cut with `reshape` and `permute` whose element counts the checker proves from `Height % Patch == 0`, a class token concatenated after `broadcast_to`, a position table sized by `(Height / Patch) * (Width / Patch) + 1`, unmasked attention (`none`), and a pooling strategy chosen by matching a compile-time `enum` constant. |

Weights bind by parameter path (`linnet inspect --parameters`); the models
have no implicit initialization, so the tests fill them with random tensors
through SafeTensors.
