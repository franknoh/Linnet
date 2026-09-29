"""Continuous batching: many requests decoded together, each at its own
length, joining the batch as soon as a row is free and leaving it when done.

A decoder that serves this way has two entries (the zoo's decoder cards
all do):

- `prefill_slots<M, S>(tokens: [M, S], slots: [M], lengths: [M]) -> [M, Vocab]`
  writes `M` requests' prompts into rows `slots` of its KV caches in one pass
  and returns the logits after each prompt's last token;
- `decode_rows(tokens: [Batch, 1], positions: [Batch]) -> [Batch, Vocab]`
  advances every row by one token, each at its own position.

`Engine` schedules requests over them. Between two decoding steps it admits
waiting requests into free rows -- their prompts in passes of up to 8, each
pass padded to one of a few compiled lengths -- then takes one step for the
whole batch; a request
leaves when it reaches its token budget, an end-of-sequence token, or the
cache's length. Rows are fixed slots of a cache sized by the model's `Batch`
and `MaxSeq` generics -- no paging -- so `Batch` is the most requests in
flight and `MaxSeq` the longest prompt plus completion.

The tokens a step produces stay on the device and feed the next step there:
the engine queues each step before it reads the previous one's tokens back,
so the accelerator is not left waiting while the host looks at them. A
request that finishes is seen one step late; its row computes that step
along and the token is dropped.

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
    prefills: int  # prompt passes, each of one or more prompts

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds if self.seconds > 0 else 0.0


class _Tokens(Protocol):
    """Token ids a queued pass produces; `tolist` waits for them."""

    def tolist(self) -> list[int]: ...


class _Backend(Protocol):
    """Holds each row's next token and position on the device. `prefill`
    queues a prompt pass and sets its rows to the token after each prompt, at
    the position after it; `decode` queues a step for every row and advances
    them all."""

    slots: int
    max_seq: int

    def prefill(self, tokens: list[list[int]], slots: list[int], lengths: list[int]) -> _Tokens: ...
    def decode(self) -> _Tokens: ...


@dataclass
class _Queued:
    """A pass the engine has queued and not yet read: `owners` are the
    `(index, row, completion)` its tokens belong to, as they were queued."""

    tokens: _Tokens
    owners: list[tuple[int, int, Completion]]
    prompts: bool


class Engine:
    """Continuous batching over a loaded decoder (see the module docs).

    `model` is a `linnet.torch` module, a `linnet.jax.LinnetModel`, or a
    `linnet.onnx.OnnxModel` with
    `prefill_slots` and `decode_rows` entries. `buckets` are the prompt
    lengths compiled (a pass is padded to the smallest that holds its longest
    prompt): by default 16, 32, 48, 64, then four to each doubling (80, 96,
    112, 128, 160, ...) up to `MaxSeq`, so padding adds less than a quarter
    to a prompt. Prompts are passed `max_group` at a time at most, in groups
    of powers of two, each a compiled shape. `graphs=True` replays the
    PyTorch step as a CUDA graph.
    """

    def __init__(
        self,
        model: Any,
        *,
        buckets: Sequence[int] | None = None,
        graphs: bool = True,
        pad: int = 0,
        max_group: int = 8,
    ) -> None:
        self.backend: _Backend = _backend_for(model, graphs)
        limit = self.backend.max_seq
        if buckets is None:
            buckets = []
            length = 16
            while length < limit:
                buckets.append(length)
                length += max(16, 1 << (length.bit_length() - 3))
            buckets.append(limit)
        self.buckets = sorted({b for b in buckets if b <= limit})
        self.pad = pad
        self.max_group = max(1, max_group)

    @property
    def slots(self) -> int:
        return self.backend.slots

    def warmup(self, prompt_lengths: Iterable[int] = ()) -> None:
        """Compiles the step and the prompt shapes `prompt_lengths` need, so
        the first requests do not pay for it."""
        lengths = {self._bucket(n) for n in prompt_lengths} or {self.buckets[0]}
        for bucket in sorted(lengths):
            group = 1
            while group <= min(self.max_group, self.slots):
                self.backend.prefill([[self.pad] * bucket] * group, list(range(group)), [1] * group)
                group *= 2
        for _ in range(2):
            self.backend.decode()
        self.backend.decode().tolist()

    def run(self, requests: Iterable[Request]) -> tuple[list[Completion], Stats]:
        """Completes every request, all of them waiting from the start, and
        returns them in the order given with the run's totals."""
        self.reset()
        order = [self.submit(request) for request in requests]
        while self.busy:
            self.step()
        seconds = time.perf_counter() - self._start
        stats = Stats(
            requests=len(order),
            prompt_tokens=sum(len(c.request.prompt) for c in order),
            generated_tokens=sum(len(c.tokens) for c in order),
            seconds=seconds,
            steps=self._steps,
            prefills=self._prefills,
        )
        return order, stats

    # ---- step by step, for a server whose requests arrive while it runs

    def reset(self) -> None:
        """Forgets every request, waiting or in a row, and restarts the clock
        the completions' times count from."""
        self._waiting: deque[Completion] = deque()
        self._rows: list[Completion | None] = [None] * self.slots
        self._positions = [0] * self.slots
        self._queued: deque[_Queued] = deque()
        self._steps = self._prefills = 0
        self._start = time.perf_counter()

    def submit(self, request: Request) -> Completion:
        """Queues a request; the next `step` admits it when a row is free.
        The returned completion fills in as the request runs."""
        if not request.prompt:
            raise ValueError("a request needs at least one prompt token")
        if len(request.prompt) >= self.backend.max_seq:
            raise ValueError(
                f"a prompt of {len(request.prompt)} tokens does not fit a cache of "
                f"{self.backend.max_seq} positions"
            )
        if not hasattr(self, "_waiting"):
            self.reset()
        completion = Completion(request)
        self._waiting.append(completion)
        return completion

    @property
    def busy(self) -> bool:
        """Whether any request is waiting or in a row, or a pass is still to
        be read."""
        return (
            bool(getattr(self, "_waiting", None))
            or any(row is not None for row in getattr(self, "_rows", []))
            or bool(getattr(self, "_queued", None))
        )

    def step(self) -> list[Completion]:
        """Admits waiting requests into free rows, their prompts in passes of
        a power of two, similar lengths together, then queues one decoding
        step for every row and reads what the passes before it produced.
        Returns the requests that finished."""
        rows = self._rows
        free = [slot for slot in range(self.slots) if rows[slot] is None]
        admitted = [self._waiting.popleft() for _ in range(min(len(free), len(self._waiting)))]
        admitted.sort(key=lambda c: len(c.request.prompt))
        placed = list(zip(free, admitted, strict=False))
        while placed:
            group = 1
            while group * 2 <= min(len(placed), self.max_group):
                group *= 2
            batch, placed = placed[:group], placed[group:]
            now = time.perf_counter() - self._start
            prompts = [list(c.request.prompt) for _, c in batch]
            bucket = self._bucket(max(len(p) for p in prompts))
            tokens = self.backend.prefill(
                [p + [self.pad] * (bucket - len(p)) for p in prompts],
                [slot for slot, _ in batch],
                [len(p) for p in prompts],
            )
            self._prefills += 1
            for slot, completion in batch:
                completion.admitted = now
                rows[slot] = completion
            owners = [(i, slot, c) for i, (slot, c) in enumerate(batch)]
            self._queued.append(_Queued(tokens, owners, prompts=True))
        ahead = 0
        if any(row is not None for row in rows):
            # One step for the whole batch; empty rows compute along. It
            # stays queued while the passes before it are read.
            owners = [(slot, slot, c) for slot, c in enumerate(rows) if c is not None]
            self._queued.append(_Queued(self.backend.decode(), owners, prompts=False))
            self._steps += 1
            ahead = 1
        finished: list[Completion] = []
        while len(self._queued) > ahead:
            finished += self._read(self._queued.popleft())
        return finished

    def _read(self, queued: _Queued) -> list[Completion]:
        produced = queued.tokens.tolist()
        now = time.perf_counter() - self._start
        finished: list[Completion] = []
        for index, slot, completion in queued.owners:
            if completion.reason:
                continue  # finished a step earlier; this token ran along
            completion.tokens.append(produced[index])
            if queued.prompts:
                completion.first_token = now
                self._positions[slot] = len(completion.request.prompt)
            else:
                self._positions[slot] += 1
            if self._finished(completion, self._positions[slot]):
                completion.finished = now
                self._rows[slot] = None
                finished.append(completion)
        return finished

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
    if module.startswith("linnet.onnx"):
        return _OnnxBackend(model)
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

        for entry in ("prefill_slots", "decode_rows"):
            if entry not in model.entries:
                raise ValueError(f"the model has no `{entry}` entry, which serving needs")
        self.torch = torch
        self.model = model
        self.slots, self.max_seq = _cache_generics(model)
        self.device = next(iter(model.parameters())).device
        self.step_compile: bool | str = "reduce-overhead" if graphs else True
        self.tokens = torch.zeros(self.slots, 1, dtype=torch.int32, device=self.device)
        self.positions = torch.zeros(self.slots, dtype=torch.int32, device=self.device)

    def _put(self, values: list[Any]) -> Any:
        host = self.torch.tensor(values, dtype=self.torch.int32)
        if self.device.type != "cuda":
            return host.to(self.device)
        # From pinned memory the copy is queued behind the work before it;
        # from pageable memory it would wait for that work to finish.
        return host.pin_memory().to(self.device, non_blocking=True)

    def prefill(self, tokens: list[list[int]], slots: list[int], lengths: list[int]) -> _Tokens:
        torch = self.torch
        rows, at = self._put(slots), self._put(lengths)
        logits = self.model.run_entry("prefill_slots", [self._put(tokens), rows, at], compile=True)
        first = logits.argmax(-1)
        self.tokens[rows.long(), 0] = first.to(torch.int32)
        self.positions[rows.long()] = at
        return _TorchTokens(first)

    def decode(self) -> _Tokens:
        logits = self.model.run_entry(
            "decode_rows", [self.tokens, self.positions], compile=self.step_compile
        )
        produced = logits.argmax(-1)
        self.tokens.copy_(produced.reshape(self.slots, 1))
        # A row whose request has finished computes along until the engine
        # reads that it has; its position stays inside the cache.
        self.positions.add_(1).clamp_(max=self.max_seq - 1)
        return _TorchTokens(produced)


class _TorchTokens:
    def __init__(self, values: Any) -> None:
        import torch

        self.ready: Any = None
        self.values: Any
        if values.device.type == "cuda":
            self.values = torch.empty(values.shape, dtype=values.dtype, pin_memory=True)
            self.values.copy_(values, non_blocking=True)
            self.ready = torch.cuda.Event()
            self.ready.record()
        else:
            self.values = values

    def tolist(self) -> list[int]:
        if self.ready is not None:
            self.ready.synchronize()
        return self.values.tolist()


class _JaxBackend:
    def __init__(self, model: Any) -> None:
        import jax as jax_module
        import jax.numpy as jnp_module
        import numpy as np

        for entry in ("prefill_slots", "decode_rows"):
            if entry not in model.entries:
                raise ValueError(f"the model has no `{entry}` entry, which serving needs")
        jax: Any = jax_module
        jnp: Any = jnp_module
        self.jnp: Any = jnp
        self.np: Any = np
        self.model = model
        self.slots, self.max_seq = _cache_generics(model)
        self.tokens: Any = jnp.zeros((self.slots, 1), jnp.int32)
        self.positions: Any = jnp.zeros((self.slots,), jnp.int32)
        last = self.max_seq - 1

        def admit(tokens: Any, positions: Any, logits: Any, rows: Any, lengths: Any) -> Any:
            first = jnp.argmax(logits, -1).astype(jnp.int32)
            return tokens.at[rows, 0].set(first), positions.at[rows].set(lengths), first

        def advance(logits: Any, positions: Any) -> Any:
            produced = jnp.argmax(logits, -1).astype(jnp.int32)
            return produced.reshape(-1, 1), jnp.minimum(positions + 1, last), produced

        self._admit: Any = jax.jit(admit)
        self._advance: Any = jax.jit(advance)

    def _put(self, values: list[Any]) -> Any:
        return self.jnp.asarray(self.np.asarray(values, dtype=self.np.int32))

    def prefill(self, tokens: list[list[int]], slots: list[int], lengths: list[int]) -> _Tokens:
        rows, at = self._put(slots), self._put(lengths)
        logits = self.model.run_entry("prefill_slots", [self._put(tokens), rows, at])
        self.tokens, self.positions, first = self._admit(
            self.tokens, self.positions, logits, rows, at
        )
        return _JaxTokens(first)

    def decode(self) -> _Tokens:
        logits = self.model.run_entry("decode_rows", [self.tokens, self.positions])
        self.tokens, self.positions, produced = self._advance(logits, self.positions)
        return _JaxTokens(produced)


class _JaxTokens:
    def __init__(self, values: Any) -> None:
        self.values = values
        values.copy_to_host_async()

    def tolist(self) -> list[int]:
        import numpy as np

        return np.asarray(self.values).tolist()


class _OnnxBackend:
    """`linnet.onnx.load_model`: NumPy in, NumPy out, caches on the device,
    the step replayed as a CUDA graph on a GPU. Each call returns when its
    results exist, so nothing overlaps."""

    def __init__(self, model: Any) -> None:
        import numpy as np

        for entry in ("prefill_slots", "decode_rows"):
            if entry not in model.entries:
                raise ValueError(f"the model has no `{entry}` entry, which serving needs")
        self.np: Any = np
        self.model = model
        self.slots, self.max_seq = _cache_generics(model)
        self.tokens: Any = np.zeros((self.slots, 1), dtype=np.int32)
        self.positions: Any = np.zeros(self.slots, dtype=np.int32)

    # The argmax runs in the graph: only each row's token leaves the device.
    def prefill(self, tokens: list[list[int]], slots: list[int], lengths: list[int]) -> _Tokens:
        np = self.np
        first = self.model.run_entry(
            "prefill_slots",
            [
                np.asarray(tokens, dtype=np.int32),
                np.asarray(slots, dtype=np.int32),
                np.asarray(lengths, dtype=np.int32),
            ],
            argmax=True,
        ).reshape(-1)
        self.tokens[slots, 0] = first
        self.positions[slots] = lengths
        return first

    def decode(self) -> _Tokens:
        np = self.np
        produced = self.model.run_entry(
            "decode_rows", [self.tokens, self.positions], argmax=True, cuda_graph=True
        ).reshape(-1)
        self.tokens = produced.astype(np.int32).reshape(self.slots, 1)
        self.positions = np.minimum(self.positions + 1, self.max_seq - 1)
        return produced


__all__ = ["Completion", "Engine", "Request", "Stats"]
