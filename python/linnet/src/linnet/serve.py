"""Continuous batching: many requests decoded together, each at its own
length, joining the batch as soon as a row is free and leaving it when done.

A decoder that serves this way has two entries (the zoo's decoder cards
all do):

- `prefill_slot<S>(tokens: [1, S], slot, length) -> [1, Vocab]` writes one
  request's prompt into row `slot` of its KV caches and returns the logits
  after the prompt's last token;
- `decode_rows(tokens: [Batch, 1], positions: [Batch]) -> [Batch, Vocab]`
  advances every row by one token, each at its own position.

`Engine` schedules requests over them. Between two decoding steps it admits
waiting requests into free rows (one prompt pass each, padded to one of a
few compiled lengths), then takes one step for the whole batch; a request
leaves when it reaches its token budget, an end-of-sequence token, or the
cache's length. Rows are fixed slots of a cache sized by the model's `Batch`
and `MaxSeq` generics -- no paging -- so `Batch` is the most requests in
flight and `MaxSeq` the longest prompt plus completion.

    from linnet.torch import load
    from linnet.serve import Engine, Request

    model = load("model.linnet", generics={..., "Batch": 32, "MaxSeq": 2048},
                 weights="model.safetensors", device="cuda")
    engine = Engine(model)
    done = engine.run([Request(prompt=ids, max_new_tokens=128) for ids in prompts])

With the PyTorch backend the step is replayed as a CUDA graph and prompts
run as generated source; with `linnet.jax.load_model` both are XLA programs.
Decoding is greedy.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Request:
    """One prompt to complete: `prompt` is token ids, at least one."""

    prompt: Sequence[int]
    max_new_tokens: int
    eos: frozenset[int] = frozenset()
    id: int = -1


@dataclass
class Completion:
    """What a request produced, and when (seconds since `run` started)."""

    request: Request
    tokens: list[int] = field(default_factory=list[int])
    admitted: float = 0.0  # its prompt pass began
    first_token: float = 0.0  # its first token existed
    finished: float = 0.0
    reason: str = ""  # "length", "eos", or "cache"

    @property
    def ttft(self) -> float:
        return self.first_token

    @property
    def inter_token(self) -> float:
        """Mean seconds between its tokens after the first."""
        extra = len(self.tokens) - 1
        return (self.finished - self.first_token) / extra if extra > 0 else 0.0


@dataclass
class Stats:
    """A run's totals: throughput counts generated tokens only."""

    requests: int
    prompt_tokens: int
    generated_tokens: int
    seconds: float
    steps: int
    prefills: int

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds if self.seconds > 0 else 0.0


class _Backend(Protocol):
    slots: int
    max_seq: int

    def prefill(self, tokens: list[int], slot: int, length: int) -> int: ...
    def decode(self, tokens: list[int], positions: list[int]) -> list[int]: ...


class Engine:
    """Continuous batching over a loaded decoder (see the module docs).

    `model` is a `linnet.torch` module or a `linnet.jax.LinnetModel` with
    `prefill_slot` and `decode_rows` entries. `buckets` are the prompt
    lengths compiled (each prompt is padded to the smallest that holds it);
    by default powers of two from 16 up to `MaxSeq`. `graphs=True` replays
    the PyTorch step as a CUDA graph.
    """

    def __init__(
        self,
        model: Any,
        *,
        buckets: Sequence[int] | None = None,
        graphs: bool = True,
        pad: int = 0,
    ) -> None:
        self.backend: _Backend = _backend_for(model, graphs)
        limit = self.backend.max_seq
        if buckets is None:
            buckets = []
            length = 16
            while length < limit:
                buckets.append(length)
                length *= 2
            buckets.append(limit)
        self.buckets = sorted(b for b in buckets if b <= limit)
        self.pad = pad

    @property
    def slots(self) -> int:
        return self.backend.slots

    def warmup(self, prompt_lengths: Iterable[int] = ()) -> None:
        """Compiles the step and the prompt shapes `prompt_lengths` need, so
        the first requests do not pay for it."""
        lengths = {self._bucket(n) for n in prompt_lengths} or {self.buckets[0]}
        for bucket in sorted(lengths):
            self.backend.prefill([self.pad] * bucket, 0, 1)
        for _ in range(3):
            self.backend.decode([self.pad] * self.slots, [0] * self.slots)

    def run(self, requests: Iterable[Request]) -> tuple[list[Completion], Stats]:
        """Completes every request, all of them waiting from the start, and
        returns them in the order given with the run's totals."""
        waiting = deque(Completion(r) for r in requests)
        order = list(waiting)
        for completion in order:
            request = completion.request
            if not request.prompt:
                raise ValueError("a request needs at least one prompt token")
            if len(request.prompt) >= self.backend.max_seq:
                raise ValueError(
                    f"a prompt of {len(request.prompt)} tokens does not fit a cache of "
                    f"{self.backend.max_seq} positions"
                )
        rows: list[Completion | None] = [None] * self.slots
        positions = [0] * self.slots
        last = [self.pad] * self.slots
        steps = prefills = 0
        start = time.perf_counter()
        while waiting or any(row is not None for row in rows):
            # Admit whoever fits: each prompt fills a free row of the caches.
            for slot in range(self.slots):
                if rows[slot] is not None or not waiting:
                    continue
                completion = waiting.popleft()
                prompt = list(completion.request.prompt)
                completion.admitted = time.perf_counter() - start
                bucket = self._bucket(len(prompt))
                token = self.backend.prefill(
                    prompt + [self.pad] * (bucket - len(prompt)), slot, len(prompt)
                )
                prefills += 1
                completion.first_token = time.perf_counter() - start
                completion.tokens.append(token)
                rows[slot], positions[slot], last[slot] = completion, len(prompt), token
                if self._finished(completion, positions[slot]):
                    completion.finished = completion.first_token
                    rows[slot] = None
            if not any(row is not None for row in rows):
                continue
            # One step for the whole batch; empty rows compute along.
            produced = self.backend.decode(last, positions)
            steps += 1
            now = time.perf_counter() - start
            for slot, completion in enumerate(rows):
                if completion is None:
                    continue
                token = produced[slot]
                completion.tokens.append(token)
                positions[slot] += 1
                last[slot] = token
                if self._finished(completion, positions[slot]):
                    completion.finished = now
                    rows[slot] = None
        seconds = time.perf_counter() - start
        stats = Stats(
            requests=len(order),
            prompt_tokens=sum(len(c.request.prompt) for c in order),
            generated_tokens=sum(len(c.tokens) for c in order),
            seconds=seconds,
            steps=steps,
            prefills=prefills,
        )
        return order, stats

    def _bucket(self, length: int) -> int:
        for bucket in self.buckets:
            if bucket >= length:
                return bucket
        raise ValueError(f"no compiled prompt length holds {length} tokens")

    def _finished(self, completion: Completion, position: int) -> bool:
        request = completion.request
        if completion.tokens[-1] in request.eos:
            completion.reason = "eos"
        elif len(completion.tokens) >= request.max_new_tokens:
            completion.reason = "length"
        elif position + 1 >= self.backend.max_seq:
            completion.reason = "cache"  # the next token would have no row left
        else:
            return False
        return True


def _backend_for(model: Any, graphs: bool) -> _Backend:
    module = type(model).__module__
    if module.startswith("linnet.jax"):
        return _JaxBackend(model)
    return _TorchBackend(model, graphs)


def _cache_generics(model: Any) -> tuple[int, int]:
    generics = dict(model.generics)
    try:
        return int(generics["Batch"]), int(generics["MaxSeq"])
    except KeyError as missing:
        raise ValueError(
            f"the model's generics do not bind {missing}; serving needs both"
        ) from None


class _TorchBackend:
    def __init__(self, model: Any, graphs: bool) -> None:
        import torch

        for entry in ("prefill_slot", "decode_rows"):
            if entry not in model.entries:
                raise ValueError(f"the model has no `{entry}` entry, which serving needs")
        self.torch = torch
        self.model = model
        self.slots, self.max_seq = _cache_generics(model)
        self.device = next(iter(model.parameters())).device
        self.step_compile: bool | str = "reduce-overhead" if graphs else True
        self.tokens = torch.zeros(self.slots, 1, dtype=torch.int32, device=self.device)
        self.positions = torch.zeros(self.slots, dtype=torch.int32, device=self.device)

    def prefill(self, tokens: list[int], slot: int, length: int) -> int:
        torch = self.torch
        ids = torch.tensor([tokens], dtype=torch.int32, device=self.device)
        scalar = torch.tensor
        logits = self.model.run_entry(
            "prefill_slot",
            [
                ids,
                scalar(slot, dtype=torch.int32, device=self.device),
                scalar(length, dtype=torch.int32, device=self.device),
            ],
            compile=True,
        )
        return int(logits[0].argmax())

    def decode(self, tokens: list[int], positions: list[int]) -> list[int]:
        torch = self.torch
        # One host-to-device copy each; the step reads them where they are.
        self.tokens.copy_(torch.tensor(tokens, dtype=torch.int32).reshape(self.slots, 1))
        self.positions.copy_(torch.tensor(positions, dtype=torch.int32))
        logits = self.model.run_entry(
            "decode_rows", [self.tokens, self.positions], compile=self.step_compile
        )
        return logits.argmax(-1).tolist()


class _JaxBackend:
    def __init__(self, model: Any) -> None:
        import jax.numpy as jnp
        import numpy as np

        for entry in ("prefill_slot", "decode_rows"):
            if entry not in model.entries:
                raise ValueError(f"the model has no `{entry}` entry, which serving needs")
        self.jnp: Any = jnp
        self.np: Any = np
        self.model = model
        self.slots, self.max_seq = _cache_generics(model)

    def prefill(self, tokens: list[int], slot: int, length: int) -> int:
        jnp = self.jnp
        logits = self.model.run_entry(
            "prefill_slot",
            [jnp.asarray([tokens], dtype=jnp.int32), jnp.int32(slot), jnp.int32(length)],
        )
        return int(jnp.argmax(logits[0]))

    def decode(self, tokens: list[int], positions: list[int]) -> list[int]:
        np, jnp = self.np, self.jnp
        logits = self.model.run_entry(
            "decode_rows",
            [
                jnp.asarray(np.asarray(tokens, dtype=np.int32).reshape(self.slots, 1)),
                jnp.asarray(np.asarray(positions, dtype=np.int32)),
            ],
        )
        return np.asarray(jnp.argmax(logits, -1)).tolist()


__all__ = ["Completion", "Engine", "Request", "Stats"]
