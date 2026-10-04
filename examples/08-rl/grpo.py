"""Reinforcement learning (GRPO) for Llama 3.1 8B with LoRA, on one GPU: the
model learns to answer two-number multiplications with the number alone. A
`linnet.serve` Engine samples a group of answers to each prompt from a second
copy of the model, which takes the policy's weights before every step; the
policy learns from each answer's reward against its group's."""

import argparse
import itertools
import random
import re

import torch
from transformers import AutoTokenizer

from linnet import nest
from linnet.packing import Prompt
from linnet.serve import Engine
from linnet.train.grpo import grpo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--card", default="llama-3.1-8b-instruct")
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--prompts", type=int, default=16, help="prompts a step")
parser.add_argument("--group", type=int, default=8, help="answers sampled per prompt")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--rank", type=int, default=16)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(nest.resolve(args.card).weights.repo)
eos = tokenizer.convert_tokens_to_ids(["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"])


def prompts():
    """Endless multiplications as chat prompts, each carrying its answer."""
    draw = random.Random(0)
    for _ in itertools.count():
        a, b = draw.randrange(10, 100), draw.randrange(10, 100)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"What is {a} * {b}?"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        yield Prompt(tokenizer(text, add_special_tokens=False)["input_ids"], a * b)


def reward(prompt: Prompt, completion: list[int]) -> float:
    """1 for the number alone, 0.5 for the right number among other text."""
    text = tokenizer.decode(completion, skip_special_tokens=True).strip()
    if text == str(prompt.data):
        return 1.0
    numbers = re.findall(r"\d[\d,]*", text)
    return 0.5 if numbers and numbers[-1].replace(",", "") == str(prompt.data) else 0.0


# The policy trains its adapters; the engine samples from a copy of the model
# compiled for serving: 64 cache rows, each step one CUDA graph.
policy = nest.load(
    args.card,
    backend="torch",
    device="cuda",
    numerics="fast",
    compile="inductor",
    generics={"Batch": 1, "MaxSeq": 1024, "T": "bf16"},
    cast_dtype=True,
)
policy.add_lora(
    ["layers.*.attention.*_proj.weight", "layers.*.mlp.*.weight"],
    rank=args.rank,
    alpha=2 * args.rank,
)
engine = Engine(
    nest.load(
        args.card,
        backend="torch",
        device="cuda",
        numerics="fast",
        compile=True,
        generics={"Batch": 64, "MaxSeq": 1024, "T": "bf16"},
        cast_dtype=True,
    )
)


def report(step):
    print(
        f"step {step.step}: reward {step.reward:.3f}, answers {step.length:.1f} tokens, "
        f"sample {step.sample_seconds:.2f} s, train {step.train_seconds:.2f} s",
        flush=True,
    )


trained = [p for p in policy.parameters() if p.requires_grad]
grpo(
    policy,
    engine,
    prompts(),
    reward,
    optimizer=torch.optim.AdamW(trained, lr=args.lr),
    steps=args.steps,
    group=args.group,
    prompts_per_step=args.prompts,
    max_new_tokens=128,
    temperature=1.0,
    eos=eos,
    correction_cap=2.0,
    on_step=report,
)
print(f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
