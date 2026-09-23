# Examples

Every example checks with `linnet lint --std stdlib examples/<name>`, formats
cleanly, materializes in PyTorch and JAX, and exports to StableHLO; the
tests under `python/linnet_torch/tests` compare the model examples against
hand-written PyTorch references.

| Example | What it shows |
| --- | --- |
| `01-linear` | A generic `op` in index notation and the block that owns its weights. |
| `02-attention` | Softmax and attention written as comprehensions with shape packs. |
| `03-block-and-weights` | A block hierarchy, sub arrays, `static for`, and the parameter manifest. |
| `04-tiny-transformer` | A package built only from the standard library, with rotary tables as inputs. |
| `05-llama` | A Llama-style decoder as a multi-module package: `crate.` imports, a `pub const` base frequency, rotary tables computed at compile time from `iota`, grouped-query attention whose `where` clause states the divisibility the reshapes rely on, `broadcast_to` for repeating key/value heads, four entries (`forward`; `next_token`, which slices the last position with `S - 1`; `decode`, which generates one token at a time from per-layer KV caches held in `state` members and written with a masked `select` at the runtime position; and `generate<Steps>`, greedy decoding inside the graph with `static for i in 0..Steps` and `std.nn.decoding::argmax`), and a dtype generic defaulting to `bf16`. |
| `06-gpt2` | GPT-2: learned positions looked up with `embedding(iota<i32>(S), wpe)` under `where S <= MaxPositions`, LayerNorm with a bias passed as an optional, a fused QKV projection split by slicing with dimension arithmetic (`H:2 * H`), the tanh GELU, and an output head tied to the token embedding through a contraction. |
| `07-vit` | A Vision Transformer: patches cut with `reshape` and `permute` whose element counts the checker proves from `Height % Patch == 0`, a class token concatenated after `broadcast_to`, a position table sized by `(Height / Patch) * (Width / Patch) + 1`, unmasked attention (`none`), and a pooling strategy chosen by matching a compile-time `enum` constant. |
| `08-clip` | A CLIP-style dual encoder as a package whose towers are separate modules sharing one encoder module: the vision tower runs it without a mask (`none`), the text tower with `some(causal_mask(...))`, and the root block exposes three entries over the same parameters — `embed_image`, `embed_text`, and `similarity`, an L2-normalized contraction scaled by a learned temperature stored as a `Tensor[1; f32]`. |

Weights bind by parameter path (`linnet inspect --parameters`); the models
have no implicit initialization, so the tests fill them with random tensors
through SafeTensors.
