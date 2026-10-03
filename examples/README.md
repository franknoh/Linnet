# Examples

Worked examples, from one linear layer up to Llama, GPT-2, ViT and CLIP.
Each directory's README walks through its source and the commands to run it.

| Example | What it shows |
| --- | --- |
| `01-linear` | A generic `op` in index notation and the block that owns its weights. |
| `02-attention` | Softmax and attention written as comprehensions with shape packs. |
| `03-block-and-weights` | A block hierarchy, sub arrays, `static for`, and the parameter manifest. |
| `04-tiny-transformer` | A package built only from the standard library, with rotary tables as inputs. |
| `05-llama` | A Llama-style decoder package: grouped-query attention, a KV cache in `state`, and six entries from a forward pass to generation loops. |
| `06-gpt2` | GPT-2: learned positions, a fused QKV projection split by slicing, and an output head tied to the token embedding. |
| `07-vit` | A Vision Transformer: patches cut by reshapes the checker proves, and pooling chosen by a compile-time `enum` constant. |
| `08-clip` | A CLIP-style dual encoder: two towers share one encoder module, and three entries share one parameter set. |
| `09-functions` | A loss, preprocessing and rewards as module-level entries, and a block entry that calls the loss. |

Every example checks with `linnet lint --std stdlib examples/<name>` and
formats cleanly. Those with an entry (03 to 09) also materialize in PyTorch
and JAX and export to StableHLO.
The tests under `python/linnet/tests/torch` compare the model examples
against hand-written PyTorch references.

Weights bind by parameter path (`linnet inspect --parameters`). The models
have no implicit initialization, so the tests fill them with random tensors
through SafeTensors.
