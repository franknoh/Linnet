"""Full fine-tuning of Llama 3.1 8B across GPUs with fully sharded data
parallelism (FSDP): each process holds a quarter of every weight, gradient
and AdamW state (on four GPUs), and gathers a layer's weights only while the
layer runs. Run under torchrun:

    torchrun --nproc_per_node=4 examples/07-fsdp/sft_torch.py
"""

import argparse
import os
import statistics

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from linnet import nest
from linnet.torch import fully_shard
from linnet.train import Example, cosine_schedule, pack, train

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--card", default="llama-3.1-8b-instruct")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--out", default=None, help="write the trained weights here (bf16)")
args = parser.parse_args()
TOKENS = 4096  # positions in a packed row

torch.distributed.init_process_group("nccl")
rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
device = f"cuda:{torch.cuda.current_device()}"
tokenizer = AutoTokenizer.from_pretrained(nest.resolve(args.card).weights.repo)


def example(row: dict) -> Example:
    user = row["instruction"] + (f"\n\n{row['input']}" if row["input"] else "")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False
    )
    prompt = tokenizer(text, add_special_tokens=False)["input_ids"][-1024:]
    answer = tokenizer(row["output"] + "<|eot_id|>", add_special_tokens=False)["input_ids"]
    return Example.prompted(prompt, answer[: 2048 - len(prompt)])


rows = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0).select(range(8400))
batches = list(pack([example(row) for row in rows.select(range(200, len(rows)))], TOKENS))

model = nest.load(
    args.card,
    backend="torch",
    device=device,
    numerics="fast",
    compile="inductor",
    generics={"Batch": 1, "MaxSeq": TOKENS, "T": "bf16"},
    cast_dtype=True,
    trainable=True,
)
# Splits every layer's weights across the processes, kept in f32, and
# compiles the entries to gather each layer's in bf16 where it runs.
fully_shard(model)
trained = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trained, lr=args.lr, weight_decay=0.0)
schedule = cosine_schedule(optimizer, 3, args.steps, floor=0.0)


def report(step):
    if rank == 0:
        print(f"step {step.step}: loss {step.loss:.3f}, {step.seconds:.2f} s", flush=True)


# One packed row a process a step: the processes take turns over the batches.
history = train(
    model,
    iter(batches[rank::world]),
    optimizer=optimizer,
    steps=args.steps,
    schedule=schedule,
    on_step=report,
)
peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device=device)
torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
if rank == 0:
    step = statistics.median(s.seconds for s in history.steps[2:])
    print(
        f"{step:.2f} s a step ({world * TOKENS / step:.0f} positions/s on {world} GPUs), "
        f"peak {float(peak):.1f} GiB a GPU"
    )
if args.out is not None:
    # Every process calls it: the parts are gathered, and the first writes.
    model.save_weights(args.out, dtype=torch.bfloat16)
torch.distributed.destroy_process_group()
