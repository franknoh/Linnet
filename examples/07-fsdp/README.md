# Full fine-tuning across GPUs

Every weight of Llama 3.1 8B trained across four GPUs with fully sharded
data parallelism: each GPU holds a quarter of every weight, gradient and
AdamW state, and gathers a layer's weights only while the layer runs. The
model is the same card in PyTorch and in JAX.

## Run

```bash
torchrun --nproc_per_node=4 examples/07-fsdp/sft_torch.py
python examples/07-fsdp/sft_jax.py            # one process, every GPU as a mesh
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python examples/07-fsdp/sft_jax.py --no-remat
```

The JAX run recomputes each layer in the backward pass by default.
`--no-remat` keeps them, which is faster but needs more than XLA's default
75% of the GPU.

On some H100 hosts NCCL fails to set up NVLink SHARP; set
`NCCL_NVLS_ENABLE=0` if it does.

On four H100s, 30 steps of one 4096-token row per GPU:

| | Step | Tokens/s | Peak a GPU | Held-out loss |
| --- | --- | --- | --- | --- |
| Linnet, PyTorch | 0.55 s | 29.4K | 48 GiB | 1.91 to 1.342 |
| Linnet, JAX, `--no-remat` | 0.53 s | 30.9K | 62 GiB | 1.91 to 1.345 |
| Linnet, JAX | 0.66 s | 24.6K | 39 GiB | 1.91 to 1.344 |
| TRL, FSDP2 | 0.57 s | 28.6K | 52 GiB | |

## What it shows

- **Sharding needs nothing in the source.** `linnet.torch.fully_shard`
  splits each layer's weights across the processes in f32, and compiles the
  entries to gather a layer's weights in bf16 where it first uses them. The
  model is one generated function, which PyTorch's own `fully_shard` cannot
  hook into.
- **JAX from the same card.** `linnet.jax.train(..., mesh=mesh)` runs each
  device on its own batch (`shard_map`). The generated code gathers each
  layer's weights where it runs (`linnet jax --fully-shard`), and reduces
  gradients into the parts in f32.
- **Memory for compute.** `remat=True` wraps each layer in `jax.checkpoint`:
  the backward pass gathers and computes the layer again, and a step keeps
  each layer's input alone. Peak memory falls from 62 to 39 GiB a GPU for a
  quarter more time a step.
- **Checkpoints.** `train(..., checkpoint=dir, checkpoint_every=100)`
  resumes from the latest one, in either framework; `save_weights` gathers
  the parts and writes one SafeTensors file.
