"""Load test for an OpenAI-compatible server: `linnet serve`, vLLM, SGLang.

Sends the same seeded chat requests to each server, new ones at each
concurrency (none passes from a prefix cache), all streamed, and reports per
run:

- throughput: requests, output tokens and all tokens per second, over the
  run from its first request to its last token;
- TTFT: from sending a request to its first token's text;
- TPOT: a request's time after its first token over its other tokens;
- ITL: the gaps between streamed tokens, every request's together;
- E2E: from sending a request to its last token.

Each latency as mean, P50, P90, P95 and P99, in milliseconds. Output tokens
are the server's count (`stream_options.include_usage`), or the streamed
chunks where it gives none. Standard library only.

    python bench/serve_load.py \\
        --server linnet=http://localhost:8100 --server vllm=http://localhost:8000 \\
        --model llama-3.1-8b-instruct --concurrency 1 8 32 --requests 8 64 128 \\
        --json results.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Plain words for prompts: about 1.3 tokens each in Llama 3's vocabulary.
_WORDS = [
    "the",
    "river",
    "light",
    "stone",
    "garden",
    "morning",
    "city",
    "window",
    "quiet",
    "music",
    "winter",
    "road",
    "paper",
    "engine",
    "harbor",
    "forest",
    "signal",
    "market",
    "evening",
    "letter",
    "bridge",
    "mountain",
    "silver",
    "orange",
    "careful",
    "distant",
    "gentle",
    "sudden",
    "ancient",
    "open",
    "narrow",
    "bright",
    "walk",
    "carry",
    "listen",
    "remember",
    "build",
    "follow",
    "notice",
    "return",
    "explain",
    "gather",
]


@dataclass
class Result:
    """One request: when it was sent, its tokens' arrival times, and its
    token counts."""

    sent: float
    arrivals: list[float] = field(default_factory=list[float])
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    @property
    def ttft(self) -> float:
        return self.arrivals[0] - self.sent

    @property
    def e2e(self) -> float:
        return self.arrivals[-1] - self.sent

    @property
    def tpot(self) -> float | None:
        others = self.output_tokens - 1
        return (self.arrivals[-1] - self.arrivals[0]) / others if others > 0 else None


def prompts(count: int, low: int, high: int, seed: int) -> list[str]:
    """`count` prompts of `low` to `high` tokens or so, the same for a seed."""
    rng = random.Random(seed)
    made: list[str] = []
    for index in range(count):
        words = round(rng.randint(low, high) / 1.3)
        body = " ".join(rng.choice(_WORDS) for _ in range(words))
        made.append(f"Request {index}. Continue this text as a long story: {body}")
    return made


def send(url: str, model: str, prompt: str, max_tokens: int, ignore_eos: bool) -> Result:
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True  # vLLM's and SGLang's extension
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    result = Result(sent=time.perf_counter())
    chunks = 0
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw in response:
                line = raw.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                usage = chunk.get("usage")
                if usage:
                    result.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    result.output_tokens = int(usage.get("completion_tokens") or 0)
                for choice in chunk.get("choices") or []:
                    if (choice.get("delta") or {}).get("content"):
                        result.arrivals.append(time.perf_counter())
                        chunks += 1
    except Exception as error:  # reported with the run
        result.error = f"{type(error).__name__}: {error}"
    if not result.output_tokens:
        result.output_tokens = chunks
    if not result.arrivals and not result.error:
        result.error = "no tokens"
    return result


def percentiles(values: list[float]) -> dict[str, float]:
    """Mean, P50, P90, P95 and P99 of `values` (seconds), in milliseconds."""
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> float:
        # Linear between the closest ranks, as NumPy's default.
        position = q * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "mean": 1000 * statistics.fmean(ordered),
        "p50": 1000 * at(0.50),
        "p90": 1000 * at(0.90),
        "p95": 1000 * at(0.95),
        "p99": 1000 * at(0.99),
    }


def run(
    url: str,
    model: str,
    texts: list[str],
    concurrency: int,
    max_tokens: int,
    ignore_eos: bool,
) -> dict[str, object]:
    """Sends `texts` with at most `concurrency` in flight; the run's metrics."""
    lock = threading.Lock()
    results: list[Result] = []

    def one(text: str) -> None:
        result = send(url, model, text, max_tokens, ignore_eos)
        with lock:
            results.append(result)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, texts))
    good = [r for r in results if not r.error]
    end = max((r.arrivals[-1] for r in good), default=time.perf_counter())
    span = end - start
    output = sum(r.output_tokens for r in good)
    prompt = sum(r.prompt_tokens for r in good)
    gaps = [b - a for r in good for a, b in zip(r.arrivals, r.arrivals[1:], strict=False)]
    return {
        "concurrency": concurrency,
        "requests": len(texts),
        "failed": len(results) - len(good),
        "errors": sorted({r.error for r in results if r.error})[:3],
        "seconds": span,
        "request_per_s": len(good) / span,
        "output_tok_per_s": output / span,
        "total_tok_per_s": (output + prompt) / span,
        "output_tokens_mean": output / len(good) if good else 0.0,
        "prompt_tokens_mean": prompt / len(good) if good else 0.0,
        "ttft_ms": percentiles([r.ttft for r in good]),
        "tpot_ms": percentiles([t for r in good if (t := r.tpot) is not None]),
        "itl_ms": percentiles(gaps),
        "e2e_ms": percentiles([r.e2e for r in good]),
    }


def table(rows: list[tuple[str, dict[str, object]]]) -> str:
    """The runs as a Markdown table."""
    head = (
        "| server | conc | req/s | out tok/s | total tok/s | out tok | "
        "TTFT mean/p50/p90/p95/p99 | TPOT mean/p50/p90/p95/p99 | "
        "ITL p50/p90/p95/p99 | E2E mean/p50/p90/p95/p99 | failed |"
    )
    lines = [head, "|" + " --- |" * 11]

    def five(stats: object, keys: tuple[str, ...]) -> str:
        values = stats if isinstance(stats, dict) else {}
        return " / ".join(f"{values.get(k, float('nan')):.0f}" for k in keys)

    every = ("mean", "p50", "p90", "p95", "p99")
    for name, m in rows:
        lines.append(
            f"| {name} | {m['concurrency']} | {m['request_per_s']:.2f} | "
            f"{m['output_tok_per_s']:.1f} | {m['total_tok_per_s']:.1f} | "
            f"{m['output_tokens_mean']:.0f} | {five(m['ttft_ms'], every)} | "
            f"{five(m['tpot_ms'], every)} | {five(m['itl_ms'], every[1:])} | "
            f"{five(m['e2e_ms'], every)} | {m['failed']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--server",
        action="append",
        required=True,
        metavar="NAME=URL",
        help="repeat per server",
    )
    parser.add_argument("--model", required=True, help="the model name each server serves")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument(
        "--requests",
        type=int,
        nargs="+",
        help="requests per concurrency (default: 4 times it)",
    )
    parser.add_argument("--prompt-tokens", type=int, nargs=2, default=[128, 512], metavar="N")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="ask for every max token (vLLM, SGLang)",
    )
    parser.add_argument("--warmup", type=int, default=2, help="requests before each server")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="write every run's metrics here")
    args = parser.parse_args(argv)
    counts = args.requests or [4 * c for c in args.concurrency]
    if len(counts) != len(args.concurrency):
        parser.error("give --requests once per --concurrency")
    servers = [tuple(s.split("=", 1)) for s in args.server]
    rows: list[tuple[str, dict[str, object]]] = []
    for name, url in servers:
        low, high = args.prompt_tokens
        for text in prompts(args.warmup, low, high, args.seed + 1):
            send(url, args.model, text, 16, False)
        for step, (concurrency, count) in enumerate(zip(args.concurrency, counts, strict=True)):
            # Prompts of their own for each run, the same for every server: a
            # prompt sent before would pass from a prefix cache.
            texts = prompts(count, low, high, args.seed + 1000 * (step + 1))
            metrics = run(url, args.model, texts, concurrency, args.max_tokens, args.ignore_eos)
            metrics["server"] = name
            rows.append((name, metrics))
            print(table([(name, metrics)]).splitlines()[-1], flush=True)
    print()
    print(table(rows))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump([m for _, m in rows], out, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
