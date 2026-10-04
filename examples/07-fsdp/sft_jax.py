"""The same full fine-tuning in JAX, from the same card: one process over
every local GPU as a mesh. Each device takes its own packed row and holds
part of every weight; each layer gathers its weights where it runs, and the
backward pass gathers and computes each layer again instead of keeping it
(`--no-remat` keeps it: faster, in 65 GiB a GPU rather than 39, beyond XLA's
default memory fraction, so set XLA_PYTHON_CLIENT_MEM_FRACTION=0.95)."""

import argparse
import statistics

import jax
import numpy as np
import optax
from datasets import load_dataset
from jax.sharding import Mesh
from transformers import AutoTokenizer

from linnet import nest
from linnet.jax.train import train
from linnet.packing import Example, pack

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--card", default="llama-3.1-8b-instruct")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument(
    "--no-remat", dest="remat", action="store_false", help="keep each layer for backward"
)
args = parser.parse_args()
TOKENS = 4096  # positions in a packed row

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
batches = pack((example(row) for row in rows.select(range(200, len(rows)))), TOKENS)

# The card's `loss_packed` entry as generated `jax.numpy`, f32 master weights
# computed in bf16.
model = nest.load(
    args.card,
    backend="jax_source",
    entry="loss_packed",
    numerics="fast",
    generics={"Batch": 1, "MaxSeq": TOKENS, "T": "bf16"},
    cast_dtype=True,
)
mesh = Mesh(np.array(jax.devices()), ("data",))
schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, 3, args.steps, 0.0)


def report(step):
    print(f"step {step.step}: loss {step.loss:.3f}, {step.seconds:.2f} s", flush=True)


params, history = train(
    model,
    batches,
    optimizer=optax.adamw(schedule, weight_decay=0.0),
    steps=args.steps,
    mesh=mesh,
    remat=args.remat,
    on_step=report,
)
step = statistics.median(s.seconds for s in history.steps[2:])
peak = max(d.memory_stats().get("peak_bytes_in_use", 0) for d in jax.devices()) / 2**30
print(
    f"{step:.2f} s a step ({mesh.size * TOKENS / step:.0f} positions/s on {mesh.size} GPUs), "
    f"peak {peak:.1f} GiB a GPU"
)
