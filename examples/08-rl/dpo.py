"""Preference optimization (DPO) for Llama 3.1 8B with LoRA, on one GPU:
UltraFeedback's chosen and rejected answers. The reference model's
log-probabilities are computed first, by the same model before it trains."""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from linnet import nest
from linnet.packing import Pair
from linnet.train.dpo import dpo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--card", default="llama-3.1-8b-instruct")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--pairs", type=int, default=32, help="pairs a step")
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--beta", type=float, default=0.1)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(nest.resolve(args.card).weights.repo)


def answer(text: str) -> list[int]:
    return tokenizer(text + "<|eot_id|>", add_special_tokens=False)["input_ids"][:512]


pairs = []
for row in load_dataset("trl-lib/ultrafeedback_binarized", split="train").shuffle(seed=0):
    conversation = row["chosen"][:-1]
    if not conversation or conversation[-1]["role"] != "user":
        continue
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    prompt = tokenizer(text, add_special_tokens=False)["input_ids"][-512:]
    pairs.append(
        Pair(prompt, answer(row["chosen"][-1]["content"]), answer(row["rejected"][-1]["content"]))
    )
    if len(pairs) == args.steps * args.pairs:
        break

model = nest.load(
    args.card,
    backend="torch",
    device="cuda",
    numerics="fast",
    compile="inductor",
    generics={"Batch": 1, "MaxSeq": 2048, "T": "bf16"},
    cast_dtype=True,
)
model.add_lora(["layers.*.attention.*_proj.weight", "layers.*.mlp.*.weight"], rank=16, alpha=32)


def report(step):
    print(
        f"step {step.step}: loss {step.loss:.3f}, accuracy {step.accuracy:.2f}, "
        f"margin {step.margin:.3f}, {step.seconds:.2f} s",
        flush=True,
    )


trained = [p for p in model.parameters() if p.requires_grad]
dpo(
    model,
    pairs,
    optimizer=torch.optim.AdamW(trained, lr=args.lr, weight_decay=0.0),
    steps=args.steps,
    pairs_per_step=args.pairs,
    beta=args.beta,
    tokens=4096,
    on_step=report,
)
print(f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
