"""Supervised fine-tuning of Llama 3.1 8B with LoRA adapters on one GPU:
Alpaca packed into 4096-token rows, trained by `linnet.train`. Prints each
step and the held-out loss before and after, and saves the adapters."""

import argparse
import statistics

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from linnet import nest
from linnet.train import Example, cosine_schedule, pack, train

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--card", default="llama-3.1-8b-instruct")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--accumulate", type=int, default=4, help="packed rows a step")
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--rank", type=int, default=16)
parser.add_argument("--out", default="adapters.safetensors")
args = parser.parse_args()
TOKENS = 4096  # positions in a packed row

tokenizer = AutoTokenizer.from_pretrained(nest.resolve(args.card).weights.repo)


def example(row: dict) -> Example:
    """An Alpaca row as a chat turn: the prompt is context, the answer is learned."""
    user = row["instruction"] + (f"\n\n{row['input']}" if row["input"] else "")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False
    )
    prompt = tokenizer(text, add_special_tokens=False)["input_ids"][-1024:]
    answer = tokenizer(row["output"] + "<|eot_id|>", add_special_tokens=False)["input_ids"]
    return Example.prompted(prompt, answer[: 2048 - len(prompt)])


rows = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0).select(range(8400))
held = list(pack([example(row) for row in rows.select(range(200))], TOKENS))
batches = pack((example(row) for row in rows.select(range(200, len(rows)))), TOKENS)

# The card's source, compiled for one packed row; Inductor fuses the step.
model = nest.load(
    args.card,
    backend="torch",
    device="cuda",
    numerics="fast",
    compile="inductor",
    generics={"Batch": 1, "MaxSeq": TOKENS, "T": "bf16"},
    cast_dtype=True,
)
# Adapters beside every attention and MLP projection; only they train.
model.add_lora(
    ["layers.*.attention.*_proj.weight", "layers.*.mlp.*.weight"],
    rank=args.rank,
    alpha=2 * args.rank,
)


def held_out() -> float:
    count = sum(batch.count for batch in held)
    with torch.no_grad():
        return sum(
            float(model.run_entry("loss_packed", batch.inputs(count, "cuda"))) for batch in held
        )


before = held_out()
trained = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trained, lr=args.lr, weight_decay=0.0)
schedule = cosine_schedule(optimizer, 3, args.steps, floor=0.0)


def report(step):
    print(f"step {step.step}: loss {step.loss:.3f}, {step.seconds:.2f} s", flush=True)


history = train(
    model,
    batches,
    optimizer=optimizer,
    steps=args.steps,
    accumulate=args.accumulate,
    schedule=schedule,
    on_step=report,
)
step = statistics.median(s.seconds for s in history.steps[2:])
print(
    f"held-out loss {before:.3f} -> {held_out():.3f}; {step:.2f} s a step "
    f"({args.accumulate * TOKENS / step:.0f} positions/s), "
    f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
)
model.save_weights(args.out, names="linnet", include=["*.lora_a", "*.lora_b"])
print(f"adapters written to {args.out}")
