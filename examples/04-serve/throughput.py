"""Serves Llama 3.1 8B from Nest with `linnet.serve` under the benchmarks'
load: 256 requests of 128 to 512 prompt tokens, at most 64 in flight, each
generating 128 tokens. Prints the generated tokens per second."""

import argparse
import random
import time

from linnet import nest
from linnet.serve import Engine, Request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--card", default="llama-3.1-8b-instruct")
parser.add_argument("--backend", choices=["torch", "jax"], default="torch")
parser.add_argument("--requests", type=int, default=256)
parser.add_argument("--rows", type=int, default=64, help="requests in flight")
parser.add_argument("--new", type=int, default=128, help="tokens each request generates")
args = parser.parse_args()

rng = random.Random(0)
prompts = [
    [rng.randrange(100, 20000) for _ in range(rng.randint(128, 512))] for _ in range(args.requests)
]
# One cache row per request in flight, long enough for the longest prompt
# and its completion.
generics = {"Batch": args.rows, "MaxSeq": 512 + args.new, "T": "bf16"}
if args.backend == "torch":
    # Generated PyTorch, each step replayed as one CUDA graph.
    model = nest.load(
        args.card,
        backend="torch",
        device="cuda",
        numerics="fast",
        compile=True,
        generics=generics,
        cast_dtype=True,
    )
else:
    # Every entry over one copy of the weights, compiled by XLA.
    model = nest.load(
        args.card, backend="jax_model", numerics="fast", generics=generics, cast_dtype=True
    )

engine = Engine(model)
start = time.perf_counter()
engine.warmup(len(prompt) for prompt in prompts)
print(f"compiled in {time.perf_counter() - start:.0f} s")

# No end token: every request generates all its tokens, as in the benchmarks.
done, stats = engine.run([Request(prompt=prompt, max_new_tokens=args.new) for prompt in prompts])
rate = stats.generated_tokens / stats.seconds
print(f"{stats.generated_tokens} tokens in {stats.seconds:.2f} s: {rate:.0f} tokens/s")
