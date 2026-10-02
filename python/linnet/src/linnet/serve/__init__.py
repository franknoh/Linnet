"""Continuous batching: many requests decoded together, each at its own
length, joining the batch as soon as a row is free and leaving it when done.

A decoder that serves this way has two entries (the zoo's decoder cards
all do):

- `prefill_slots<M, S>(tokens: [M, S], slots: [M], lengths: [M]) -> [M, Vocab]`
  writes `M` requests' prompts into rows `slots` of its KV caches in one pass
  and returns the logits after each prompt's last token;
- `decode_rows(tokens: [Batch, 1], positions: [Batch]) -> [Batch, Vocab]`
  advances every row by one token, each at its own position.

A card may also have `prefill_packed<P>(tokens: [P], rows: [P], positions:
[P], segments: [P], last: [Batch]) -> [Batch, Vocab]`, which takes several
prompts packed end to end in one pass of `P` tokens, each token told its
row, its position, and which prompt it belongs to. With PyTorch the engine
uses it: a pass then carries no padding between prompts, only after the last
one up to a compiled size. And a card may have `step_packed<P>(tokens, rows,
positions, segments, last, step_tokens: [Batch, 1], step_positions: [Batch])
-> [2 * Batch, Vocab]`, `prefill_packed`'s pass with `decode_rows`'s step
for every row in it: the logits after each prompt, then each row's step.
A step that admits prompts while other rows decode then reads the weights
once, not once for the prompts and again for the step.

`Engine` schedules requests over them. Between two decoding steps it admits
waiting requests into free rows -- their prompts packed into passes of up to
`pack` tokens, or in passes of up to 8 each padded to one of a few compiled
lengths -- then takes one step for the whole batch; a request
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

Decoding is greedy unless a request sets a `temperature`; then its tokens
are drawn on the device, from its `top_k` and `top_p` tokens when it sets
those (see `linnet.serve.sampling`), and a request with the same `seed`
draws the same tokens whatever else is in the batch. A request that sets
`logprobs` also gets each token's log-probability under the model, and the
`logprobs` most likely tokens with theirs; a pass computes them only when a
request in it asks. A request's `on_token` is called with its completion as
each token is read, which is how a server streams. `python -m linnet.serve` serves a Nest model over
HTTP (`linnet.serve.server`).
"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .sampling import Sampling, draw_jax, draw_numpy, draw_torch, mode


@dataclass
class Request:
    """One prompt to complete: `prompt` is token ids, at least one.

    `temperature` 0 decodes greedily; above it, tokens are drawn from the
    softmax of the logits over it, from the `top_k` most likely (0 for all)
    and then the fewest whose probabilities reach `top_p`. `seed` (its low
    32 bits) fixes the draws; without one the engine picks one. `on_token`
    is called with the completion each time a token of it is read, with
    `reason` set on its last. `logprobs` asks for each token's
    log-probability, and that many alternatives with theirs (0 for none):
    log-softmax of the model's logits, before temperature and filtering."""

    prompt: Sequence[int]
    max_new_tokens: int
    eos: frozenset[int] = frozenset()
    id: int = -1
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None
    on_token: Callable[[Completion], None] | None = None
    logprobs: int | None = None


@dataclass
class Completion:
    """What a request produced, and when (seconds since `run` started)."""

    request: Request
    tokens: list[int] = field(default_factory=list[int])
    admitted: float = 0.0  # its prompt pass began
    first_token: float = 0.0  # its first token existed
    finished: float = 0.0
    reason: str = ""  # "length", "eos", "cache", or what `Engine.cancel` gave
    sampling: Sampling = field(default_factory=Sampling)  # its seed chosen
    # With `Request.logprobs`: each token's log-probability, and the most
    # likely tokens at its position as (token, log-probability), best first.
    logprobs: list[float] = field(default_factory=list[float])
    top_logprobs: list[list[tuple[int, float]]] = field(
        default_factory=list[list[tuple[int, float]]]
    )

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
    """Token ids a queued pass produces; `tolist` waits for them.
    `logprobs` is each token's log-probability and the ids and
    log-probabilities of the most likely tokens, when the pass computed
    them."""

    def tolist(self) -> list[int]: ...
    def logprobs(self) -> _Logprobs | None: ...


# (chosen, top ids, top log-probabilities), one entry per row of a pass.
_Logprobs = tuple[list[float], list[list[int]], list[list[float]]]


class _Backend(Protocol):
    """Holds each row's next token, position, and sampling on the device.
    `prefill` queues a prompt pass, sets its rows' sampling, and sets them to
    the token drawn after each prompt, at the position after it; `decode`
    queues a step for every row and advances them all, `need` saying what
    drawing the rows in use takes (`sampling.mode`). `top` is -1 when no row
    of the pass wants log-probabilities, and otherwise the most
    alternatives one wants."""

    slots: int
    max_seq: int
    packs: bool  # `prefill_packed` runs
    mixes: bool  # `step_packed` runs

    def prefill(
        self,
        tokens: list[list[int]],
        slots: list[int],
        lengths: list[int],
        sampling: list[Sampling],
        top: int = -1,
    ) -> _Tokens: ...
    def prefill_packed(
        self,
        prompts: list[list[int]],
        slots: list[int],
        sampling: list[Sampling],
        size: int,
        pad: int,
        top: int = -1,
    ) -> _Tokens: ...
    def decode(self, need: int, top: int = -1) -> _Tokens: ...
    def step_packed(
        self,
        prompts: list[list[int]],
        slots: list[int],
        sampling: list[Sampling],
        size: int,
        pad: int,
        need: int,
        top: int = -1,
        step_top: int = -1,
    ) -> tuple[_Tokens, _Tokens]: ...


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

    A PyTorch model whose card has `prefill_packed` takes its prompts packed
    instead, first fit by decreasing length into passes of up to `pack`
    tokens (at least the longest prompt a row holds) and `Batch` prompts,
    each padded at its end to the smallest of `pack_sizes` that holds it: by
    default 512, 1024, 1536, 2048, 3072 and 4096, scaled to `pack`.
    `pack=0` keeps the grouped passes. A card that also has `step_packed`
    takes the step of the rows already decoding in the last such pass of a
    step (`mix=False` keeps them apart); the rows that pass admits take
    their first step with the next.
    """

    def __init__(
        self,
        model: Any,
        *,
        buckets: Sequence[int] | None = None,
        graphs: bool = True,
        pad: int = 0,
        max_group: int = 8,
        pack: int = 4096,
        mix: bool = True,
    ) -> None:
        self.backend: _Backend = _backend_for(model, graphs)
        limit = self.backend.max_seq
        self.pack = 0
        self.pack_sizes: list[int] = []
        if pack > 0 and self.backend.packs:
            unit = -(-max(pack, limit) // 8)
            # FlexAttention's 128-wide blocks divide every size worth its
            # kernel; a smaller pass runs as a plain masked attention.
            step = 128 if unit >= 128 else 16
            unit = -(-unit // step) * step
            self.pack_sizes = [unit * k for k in (1, 2, 3, 4, 6, 8)]
            self.pack = self.pack_sizes[-1]
        if buckets is None:
            buckets = []
            length = 16
            while length < limit:
                buckets.append(length)
                length += max(16, 1 << (length.bit_length() - 3))
            buckets.append(limit)
        self.buckets = sorted({b for b in buckets if b <= limit})
        self.mix = mix and bool(self.pack) and self.backend.mixes
        self.pad = pad
        self.max_group = max(1, max_group)
        self._seeds = random.Random()

    @property
    def slots(self) -> int:
        return self.backend.slots

    def warmup(self, prompt_lengths: Iterable[int] = ()) -> None:
        """Compiles the step and the prompt shapes `prompt_lengths` need, so
        the first requests do not pay for it."""
        if self.pack:
            # The sizes a pass can reach: up to the one that holds the longest
            # `slots` prompts together (every size, without lengths).
            longest = sorted(prompt_lengths)[-self.slots :]
            reach = sum(longest) if longest else self.pack
            for size in self.pack_sizes:
                self.backend.prefill_packed([[self.pad]], [0], [Sampling()], size, self.pad)
                if self.mix:
                    self.backend.step_packed(
                        [[self.pad]], [0], [Sampling()], size, self.pad, mode([])
                    )
                if size >= reach:
                    break
            for _ in range(2):
                self.backend.decode(mode([]))
            self.backend.decode(mode([])).tolist()
            return
        lengths = {self._bucket(n) for n in prompt_lengths} or {self.buckets[0]}
        for bucket in sorted(lengths):
            group = 1
            while group <= min(self.max_group, self.slots):
                self.backend.prefill(
                    [[self.pad] * bucket] * group,
                    list(range(group)),
                    [1] * group,
                    [Sampling()] * group,
                )
                group *= 2
        for _ in range(2):
            self.backend.decode(mode([]))
        self.backend.decode(mode([])).tolist()

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
        seed = request.seed if request.seed is not None else self._seeds.getrandbits(32)
        sampling = Sampling(request.temperature, request.top_k, request.top_p, seed)
        sampling.check()
        if not hasattr(self, "_waiting"):
            self.reset()
        completion = Completion(request, sampling=sampling)
        self._waiting.append(completion)
        return completion

    def cancel(self, completion: Completion, reason: str = "cancelled") -> None:
        """Ends a request that is waiting or in a row, with `reason`; its row
        is free for the next `step`. A request already finished is left as
        it is."""
        if completion.reason:
            return
        completion.reason = reason
        completion.finished = time.perf_counter() - self._start
        self._waiting = deque(c for c in self._waiting if c is not completion)
        for slot, row in enumerate(self._rows):
            if row is completion:
                self._rows[slot] = None

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
        decoding = any(row is not None for row in rows)
        free = [slot for slot in range(self.slots) if rows[slot] is None]
        admitted = [self._waiting.popleft() for _ in range(min(len(free), len(self._waiting)))]
        admitted.sort(key=lambda c: len(c.request.prompt))
        placed = list(zip(free, admitted, strict=False))
        passes = self._packed(placed)
        # With rows already decoding, the last pass carries their step.
        mixed = passes.pop() if passes and decoding and self.mix else None
        for batch in passes:
            now = time.perf_counter() - self._start
            prompts = [list(c.request.prompt) for _, c in batch]
            size = next(s for s in self.pack_sizes if s >= sum(len(p) for p in prompts))
            tokens = self.backend.prefill_packed(
                prompts,
                [slot for slot, _ in batch],
                [c.sampling for _, c in batch],
                size,
                self.pad,
                _top(c for _, c in batch),
            )
            self._prefills += 1
            for slot, completion in batch:
                completion.admitted = now
                rows[slot] = completion
            owners = [(i, slot, c) for i, (slot, c) in enumerate(batch)]
            self._queued.append(_Queued(tokens, owners, prompts=True))
        if self.pack:
            placed = []
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
                [c.sampling for _, c in batch],
                _top(c for _, c in batch),
            )
            self._prefills += 1
            for slot, completion in batch:
                completion.admitted = now
                rows[slot] = completion
            owners = [(i, slot, c) for i, (slot, c) in enumerate(batch)]
            self._queued.append(_Queued(tokens, owners, prompts=True))
        ahead = 0
        if mixed is not None:
            now = time.perf_counter() - self._start
            prompts = [list(c.request.prompt) for _, c in mixed]
            size = next(s for s in self.pack_sizes if s >= sum(len(p) for p in prompts))
            # Every row in use steps but those this pass fills, which have no
            # token to step from yet.
            stepping = [(slot, slot, c) for slot, c in enumerate(rows) if c is not None]
            first, stepped = self.backend.step_packed(
                prompts,
                [slot for slot, _ in mixed],
                [c.sampling for _, c in mixed],
                size,
                self.pad,
                mode(c.sampling for _, _, c in stepping),
                _top(c for _, c in mixed),
                _top(c for _, _, c in stepping),
            )
            self._prefills += 1
            self._steps += 1
            for slot, completion in mixed:
                completion.admitted = now
                rows[slot] = completion
            owners = [(i, slot, c) for i, (slot, c) in enumerate(mixed)]
            self._queued.append(_Queued(first, owners, prompts=True))
            self._queued.append(_Queued(stepped, stepping, prompts=False))
            ahead = 2
        elif any(row is not None for row in rows):
            # One step for the whole batch; empty rows compute along. It
            # stays queued while the passes before it are read.
            owners = [(slot, slot, c) for slot, c in enumerate(rows) if c is not None]
            need = mode(c.sampling for _, _, c in owners)
            top = _top(c for _, _, c in owners)
            self._queued.append(_Queued(self.backend.decode(need, top), owners, prompts=False))
            self._steps += 1
            ahead = 1
        finished: list[Completion] = []
        while len(self._queued) > ahead:
            finished += self._read(self._queued.popleft())
        return finished

    def _read(self, queued: _Queued) -> list[Completion]:
        produced = queued.tokens.tolist()
        logprobs = queued.tokens.logprobs()
        now = time.perf_counter() - self._start
        finished: list[Completion] = []
        for index, slot, completion in queued.owners:
            if completion.reason:
                continue  # finished a step earlier; this token ran along
            completion.tokens.append(produced[index])
            wanted = completion.request.logprobs
            if wanted is not None and logprobs is not None:
                chosen, ids, values = logprobs
                completion.logprobs.append(chosen[index])
                completion.top_logprobs.append(
                    list(zip(ids[index][:wanted], values[index][:wanted], strict=True))
                )
            if queued.prompts:
                completion.first_token = now
                self._positions[slot] = len(completion.request.prompt)
            else:
                self._positions[slot] += 1
            if self._finished(completion, self._positions[slot]):
                completion.finished = now
                self._rows[slot] = None
                finished.append(completion)
            if completion.request.on_token is not None:
                completion.request.on_token(completion)
        return finished

    def _packed(self, placed: list[tuple[int, Completion]]) -> list[list[tuple[int, Completion]]]:
        """The admitted prompts as packed passes: first fit, longest first,
        into passes of at most `pack` tokens and `slots` prompts."""
        if not self.pack:
            return []
        passes: list[list[tuple[int, Completion]]] = []
        room: list[int] = []
        for slot, completion in sorted(placed, key=lambda sc: -len(sc[1].request.prompt)):
            length = len(completion.request.prompt)
            for i, left in enumerate(room):
                if left >= length and len(passes[i]) < self.slots:
                    passes[i].append((slot, completion))
                    room[i] -= length
                    break
            else:
                passes.append([(slot, completion)])
                room.append(self.pack - length)
        return passes

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


def _top(completions: Iterable[Completion]) -> int:
    """What a pass computes of log-probabilities (see `_Backend`)."""
    wanted = [c.request.logprobs for c in completions if c.request.logprobs is not None]
    return max(wanted, default=-1)


def _logprobs_torch(logits: Any, produced: Any, top: int) -> tuple[Any, Any, Any]:
    """Each row's token's log-probability and its `top` most likely tokens,
    from the model's logits, before temperature and filtering."""
    scores = logits.float().log_softmax(-1)
    chosen = scores.gather(-1, produced.long().reshape(-1, 1)).reshape(-1)
    values, ids = scores.topk(max(top, 1), -1)
    return chosen, ids[:, :top], values[:, :top]


def _logprobs_numpy(logits: Any, produced: Any, top: int) -> _Logprobs:
    import numpy as numpy_module

    np: Any = numpy_module
    scores = logits.astype(np.float32)
    peak = scores.max(-1, keepdims=True)
    scores = scores - peak - np.log(np.exp(scores - peak).sum(-1, keepdims=True))
    rows = np.arange(scores.shape[0])
    chosen = scores[rows, np.asarray(produced).reshape(-1)]
    ids = np.argsort(-scores, axis=-1, kind="stable")[:, :top]
    values = np.take_along_axis(scores, ids, -1)
    return chosen.tolist(), ids.tolist(), values.tolist()


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
        self.packs = "prefill_packed" in model.entries
        self.mixes = self.packs and "step_packed" in model.entries
        self.step_compile: bool | str = "reduce-overhead" if graphs else True
        self.tokens = torch.zeros(self.slots, 1, dtype=torch.int32, device=self.device)
        self.positions = torch.zeros(self.slots, dtype=torch.int32, device=self.device)
        self.sampling = [Sampling()] * self.slots
        self._hold([], [])
        # Compiled, drawing reads the logits a few times, not once for each
        # step of the hash; the batch dimension varies without recompiling.
        self._draw: Any = draw_torch
        if self.device.type == "cuda":
            self._draw = torch.compile(draw_torch, dynamic=None)

    def _put(self, values: list[Any], dtype: Any = None) -> Any:
        host = self.torch.tensor(values, dtype=dtype or self.torch.int32)
        if self.device.type != "cuda":
            return host.to(self.device)
        # From pinned memory the copy is queued behind the work before it;
        # from pageable memory it would wait for that work to finish.
        return host.pin_memory().to(self.device, non_blocking=True)

    def _hold(self, slots: list[int], sampling: list[Sampling]) -> None:
        """Puts rows' sampling on the device, all of it, when any changed."""
        torch = self.torch
        if slots and all(self.sampling[s] == x for s, x in zip(slots, sampling, strict=True)):
            return
        for slot, row in zip(slots, sampling, strict=True):
            self.sampling[slot] = row
        rows = self.sampling
        self.temperature = self._put([r.temperature for r in rows], torch.float32)
        self.top_k = self._put([r.top_k for r in rows], torch.int64)
        self.top_p = self._put([r.top_p for r in rows], torch.float32)
        self.keys = self._put([r.key for r in rows], torch.int64)

    def prefill(
        self,
        tokens: list[list[int]],
        slots: list[int],
        lengths: list[int],
        sampling: list[Sampling],
        top: int = -1,
    ) -> _Tokens:
        torch = self.torch
        rows, at = self._put(slots), self._put(lengths)
        logits = self.model.run_entry("prefill_slots", [self._put(tokens), rows, at], compile=True)
        self._hold(slots, sampling)
        index = rows.long()
        first = self._draw(
            logits,
            self.temperature[index],
            self.top_k[index],
            self.top_p[index],
            self.keys[index],
            at,
            mode(sampling),
        )
        self.tokens[index, 0] = first.to(torch.int32)
        self.positions[index] = at
        return _TorchTokens(first, _logprobs_torch(logits, first, top) if top >= 0 else None)

    def _pack(self, prompts: list[list[int]], slots: list[int], size: int, pad: int) -> list[Any]:
        """A packed pass's `tokens`, `rows`, `positions`, `segments` and
        `last`, on the device."""
        tokens: list[int] = []
        rows: list[int] = []
        positions: list[int] = []
        segments: list[int] = []
        last: list[int] = []
        for m, (prompt, slot) in enumerate(zip(prompts, slots, strict=True)):
            tokens += prompt
            rows += [slot] * len(prompt)
            positions += range(len(prompt))
            segments += [m] * len(prompt)
            last.append(len(tokens) - 1)
        # Padding after the last prompt: a segment of its own, written at the
        # cache's last position, which a request stops before it reaches.
        extra = size - len(tokens)
        tokens += [pad] * extra
        rows += [slots[0]] * extra
        positions += [self.max_seq - 1] * extra
        segments += [-1] * extra
        last += [last[-1]] * (self.slots - len(last))
        packed = self._put([tokens, rows, positions, segments])
        return [packed[0], packed[1], packed[2], packed[3], self._put(last)]

    def _admit(
        self, logits: Any, prompts: list[list[int]], slots: list[int], sampling: list[Sampling]
    ) -> Any:
        """Draws the token after each prompt from its logits and sets its row
        to it, at the position after the prompt."""
        torch = self.torch
        lengths = [len(prompt) for prompt in prompts]
        rows_at, at = self._put(slots), self._put(lengths)
        self._hold(slots, sampling)
        index = rows_at.long()
        first = self._draw(
            logits,
            self.temperature[index],
            self.top_k[index],
            self.top_p[index],
            self.keys[index],
            at,
            mode(sampling),
        )
        self.tokens[index, 0] = first.to(torch.int32)
        self.positions[index] = at
        return first

    def prefill_packed(
        self,
        prompts: list[list[int]],
        slots: list[int],
        sampling: list[Sampling],
        size: int,
        pad: int,
        top: int = -1,
    ) -> _Tokens:
        # A few pass sizes, each compiled like the step (and replayed as a CUDA
        # graph): FlexAttention runs only under `torch.compile`, and a pass
        # of thousands of tokens has as many kernels as the step.
        logits = self.model.run_entry(
            "prefill_packed", self._pack(prompts, slots, size, pad), compile=self.step_compile
        )[: len(prompts)]
        first = self._admit(logits, prompts, slots, sampling)
        return _TorchTokens(first, _logprobs_torch(logits, first, top) if top >= 0 else None)

    def step_packed(
        self,
        prompts: list[list[int]],
        slots: list[int],
        sampling: list[Sampling],
        size: int,
        pad: int,
        need: int,
        top: int = -1,
        step_top: int = -1,
    ) -> tuple[_Tokens, _Tokens]:
        """`prefill_packed` and `decode` in one pass. The rows the prompts
        fill step along from no token of theirs: their step is written at
        the cache's last position, which a request stops before it reaches,
        and the prompts' tokens then set them."""
        index = self._put(slots).long()
        positions = self.positions.clone()
        positions[index] = self.max_seq - 1
        logits = self.model.run_entry(
            "step_packed",
            [*self._pack(prompts, slots, size, pad), self.tokens, positions],
            compile=self.step_compile,
        )
        stepped = self._advance(logits[self.slots :], need)
        prompted = logits[: len(prompts)]
        first = self._admit(prompted, prompts, slots, sampling)
        return (
            _TorchTokens(first, _logprobs_torch(prompted, first, top) if top >= 0 else None),
            _TorchTokens(
                stepped,
                _logprobs_torch(logits[self.slots :], stepped, step_top) if step_top >= 0 else None,
            ),
        )

    def _advance(self, logits: Any, need: int) -> Any:
        """Draws every row's next token from its step's logits and moves the
        row on to it."""
        produced = self._draw(
            logits, self.temperature, self.top_k, self.top_p, self.keys, self.positions + 1, need
        )
        self.tokens.copy_(produced.reshape(self.slots, 1))
        # A row whose request has finished computes along until the engine
        # reads that it has; its position stays inside the cache.
        self.positions.add_(1).clamp_(max=self.max_seq - 1)
        return produced

    def decode(self, need: int, top: int = -1) -> _Tokens:
        logits = self.model.run_entry(
            "decode_rows", [self.tokens, self.positions], compile=self.step_compile
        )
        produced = self._advance(logits, need)
        return _TorchTokens(produced, _logprobs_torch(logits, produced, top) if top >= 0 else None)


class _TorchTokens:
    def __init__(self, values: Any, logprobs: tuple[Any, Any, Any] | None = None) -> None:
        import torch

        self.ready: Any = None
        parts = [values, *(logprobs or ())]
        if values.device.type == "cuda":
            host = [torch.empty(p.shape, dtype=p.dtype, pin_memory=True) for p in parts]
            for target, part in zip(host, parts, strict=True):
                target.copy_(part, non_blocking=True)
            parts = host
            self.ready = torch.cuda.Event()
            self.ready.record()
        self.values: Any = parts[0]
        self.extra: list[Any] = parts[1:]

    def tolist(self) -> list[int]:
        if self.ready is not None:
            self.ready.synchronize()
        return self.values.tolist()

    def logprobs(self) -> _Logprobs | None:
        if not self.extra:
            return None
        if self.ready is not None:
            self.ready.synchronize()
        chosen, ids, values = self.extra
        return chosen.tolist(), ids.tolist(), values.tolist()


class _Grouped:
    """A backend whose prompt passes are grouped and padded (`prefill_slots`
    only): XLA and ONNX Runtime, where a packed pass's mask would cost its
    whole square."""

    packs = False
    mixes = False

    def prefill_packed(
        self,
        prompts: list[list[int]],
        slots: list[int],
        sampling: list[Sampling],
        size: int,
        pad: int,
        top: int = -1,
    ) -> _Tokens:
        raise NotImplementedError("packed prompt passes run with the PyTorch backend")

    def step_packed(
        self,
        prompts: list[list[int]],
        slots: list[int],
        sampling: list[Sampling],
        size: int,
        pad: int,
        need: int,
        top: int = -1,
        step_top: int = -1,
    ) -> tuple[_Tokens, _Tokens]:
        raise NotImplementedError("packed prompt passes run with the PyTorch backend")


class _JaxBackend(_Grouped):
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
        self.sampling = [Sampling()] * self.slots
        self._hold([], [])
        last = self.max_seq - 1

        def logprobs(logits: Any, produced: Any, top: int) -> Any:
            if top < 0:
                return ()
            scores = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            chosen = jnp.take_along_axis(scores, produced[:, None], -1)[:, 0]
            values, ids = jax.lax.top_k(scores, max(top, 1))
            return chosen, ids[:, :top], values[:, :top]

        # `held` is each row's temperature, top-k, top-p, and seed key.
        def admit(
            tokens: Any,
            positions: Any,
            logits: Any,
            rows: Any,
            lengths: Any,
            held: Any,
            need: int,
            top: int,
        ) -> Any:
            temperature, top_k, top_p, keys = held
            first = draw_jax(
                logits, temperature[rows], top_k[rows], top_p[rows], keys[rows], lengths, need
            )
            return (
                tokens.at[rows, 0].set(first),
                positions.at[rows].set(lengths),
                first,
                logprobs(logits, first, top),
            )

        def advance(logits: Any, positions: Any, held: Any, need: int, top: int) -> Any:
            temperature, top_k, top_p, keys = held
            produced = draw_jax(logits, temperature, top_k, top_p, keys, positions + 1, need)
            return (
                produced.reshape(-1, 1),
                jnp.minimum(positions + 1, last),
                produced,
                logprobs(logits, produced, top),
            )

        self._admit: Any = jax.jit(admit, static_argnames=("need", "top"))
        self._advance: Any = jax.jit(advance, static_argnames=("need", "top"))

    def _put(self, values: list[Any], dtype: Any = None) -> Any:
        return self.jnp.asarray(self.np.asarray(values, dtype=dtype or self.np.int32))

    def _hold(self, slots: list[int], sampling: list[Sampling]) -> None:
        """Puts rows' sampling on the device, all of it, when any changed."""
        np = self.np
        if slots and all(self.sampling[s] == x for s, x in zip(slots, sampling, strict=True)):
            return
        for slot, row in zip(slots, sampling, strict=True):
            self.sampling[slot] = row
        rows = self.sampling
        self.held = (
            self._put([r.temperature for r in rows], np.float32),
            self._put([r.top_k for r in rows], np.int32),
            self._put([r.top_p for r in rows], np.float32),
            self._put([r.key for r in rows], np.uint32),
        )

    def prefill(
        self,
        tokens: list[list[int]],
        slots: list[int],
        lengths: list[int],
        sampling: list[Sampling],
        top: int = -1,
    ) -> _Tokens:
        rows, at = self._put(slots), self._put(lengths)
        logits = self.model.run_entry("prefill_slots", [self._put(tokens), rows, at])
        self._hold(slots, sampling)
        self.tokens, self.positions, first, logprobs = self._admit(
            self.tokens, self.positions, logits, rows, at, self.held, need=mode(sampling), top=top
        )
        return _JaxTokens(first, logprobs)

    def decode(self, need: int, top: int = -1) -> _Tokens:
        logits = self.model.run_entry("decode_rows", [self.tokens, self.positions])
        self.tokens, self.positions, produced, logprobs = self._advance(
            logits, self.positions, self.held, need=need, top=top
        )
        return _JaxTokens(produced, logprobs)


class _JaxTokens:
    def __init__(self, values: Any, logprobs: tuple[Any, ...] = ()) -> None:
        self.values = values
        self.extra = logprobs
        for part in (values, *logprobs):
            part.copy_to_host_async()

    def tolist(self) -> list[int]:
        import numpy as np

        return np.asarray(self.values).tolist()

    def logprobs(self) -> _Logprobs | None:
        import numpy as np

        if not self.extra:
            return None
        chosen, ids, values = (np.asarray(part).tolist() for part in self.extra)
        return chosen, ids, values


class _OnnxBackend(_Grouped):
    """`linnet.onnx.load_model`: NumPy in, NumPy out, caches on the device,
    the step replayed as a CUDA graph on a GPU. Each call returns when its
    results exist, so nothing overlaps. Greedy passes take the argmax in the
    graph, so only each row's token leaves the device; a pass that samples
    takes the logits to the host and draws there."""

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
        self.sampling = [Sampling()] * self.slots

    def prefill(
        self,
        tokens: list[list[int]],
        slots: list[int],
        lengths: list[int],
        sampling: list[Sampling],
        top: int = -1,
    ) -> _Tokens:
        np = self.np
        need = mode(sampling)
        # The argmax in the graph unless the logits themselves are needed.
        on_device = not need and top < 0
        out = self.model.run_entry(
            "prefill_slots",
            [
                np.asarray(tokens, dtype=np.int32),
                np.asarray(slots, dtype=np.int32),
                np.asarray(lengths, dtype=np.int32),
            ],
            argmax=on_device,
        )
        first = out.reshape(-1) if on_device else draw_numpy(out, sampling, lengths, need)
        for slot, row in zip(slots, sampling, strict=True):
            self.sampling[slot] = row
        self.tokens[slots, 0] = first
        self.positions[slots] = lengths
        return _HostTokens(first, _logprobs_numpy(out, first, top) if top >= 0 else None)

    def decode(self, need: int, top: int = -1) -> _Tokens:
        np = self.np
        on_device = not need and top < 0
        out = self.model.run_entry(
            "decode_rows", [self.tokens, self.positions], argmax=on_device, cuda_graph=True
        )
        if on_device:
            produced = out.reshape(-1)
        else:
            produced = draw_numpy(out, self.sampling, self.positions + 1, need)
        self.tokens = produced.astype(np.int32).reshape(self.slots, 1)
        self.positions = np.minimum(self.positions + 1, self.max_seq - 1)
        return _HostTokens(produced, _logprobs_numpy(out, produced, top) if top >= 0 else None)


class _HostTokens:
    """A pass's tokens already on the host, and its log-probabilities when
    it computed them."""

    def __init__(self, values: Any, logprobs: _Logprobs | None) -> None:
        self.values = values
        self.computed = logprobs

    def tolist(self) -> list[int]:
        return self.values.tolist()

    def logprobs(self) -> _Logprobs | None:
        return self.computed


__all__ = ["Completion", "Engine", "Request", "Sampling", "Stats"]
