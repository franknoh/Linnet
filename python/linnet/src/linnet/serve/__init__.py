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
and `MaxSeq` generics, so `Batch` is the most requests in flight and `MaxSeq`
the longest prompt plus completion -- unless the engine serves from pages.

A card with `prefill_paged` and `decode_paged` entries (and `step_paged`,
which mixes as `step_packed` does) can serve from pages instead (`paged`).
Loaded with `Batch` 1, its caches' one row is a pool of `MaxSeq` positions
cut into pages of `PageSize`; each of the engine's `rows` takes pages as it
grows and gives them back when it ends, so a request holds the cache it
uses, not a whole row. When the pool runs short, the request admitted last
goes back to the queue with its tokens so far and passes again as a longer
prompt once pages are free.

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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from .sampling import Sampling, draw_jax, draw_numpy, draw_torch, mode

if TYPE_CHECKING:
    import jax
    import numpy as np
    import torch
    from jax.typing import ArrayLike
    from numpy.typing import DTypeLike, NDArray

    from ..jax.module import LinnetModel
    from ..onnx.runtime import OnnxModel
    from ..torch.module import LinnetModule

    # What `Engine` serves: a `linnet.torch` module, a `linnet.jax`
    # model, or a `linnet.onnx` model.
    _Model: TypeAlias = LinnetModule | LinnetModel | OnnxModel
    # Each row's temperature, top-k, top-p, and seed key, on the device (JAX).
    _Held: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    # A JAX pass's rows' next tokens and positions, the tokens it drew, and
    # their log-probabilities (empty unless asked for).
    _Moved: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]


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
    preempted: int = 0  # times it gave its pages back and waited again (paged)
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
    preempted: int = 0  # requests sent back to wait for pages (paged)

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


# Per prompt of a packed pass, the rows that share it: each takes the
# prompt's cache rows and draws its own first token, with its own sampling.
_Copies = list[list[tuple[int, Sampling]]]


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
    paged: bool  # rows take pages of a pool (`prefill_paged`, `decode_paged`)
    page_size: int
    pages: int  # the pool's, the first kept for writes nobody reads

    def assign(self, row: int, pages: list[int]) -> None:
        """Sets the pages a row's positions lie in, in order (paged)."""
        ...

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
        copies: _Copies | None = None,
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
        copies: _Copies | None = None,
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
    takes the step of the rows already decoding in the smallest such pass of
    a step, if it fits the second of `pack_sizes` (`mix=False` keeps them
    apart); the rows that pass admits take their first step with the next.

    `paged` serves from pages (see the module docs): by default when the
    card has the paged entries and the model was loaded with `Batch` 1.
    `rows` requests are then in flight at most, each `max_len` positions
    long at most (by default every page of the pool but one, up to 8192). A
    request is admitted while the pool keeps `reserve` pages free beside its
    prompt's (by default 1%).
    """

    def __init__(
        self,
        model: _Model,
        *,
        buckets: Sequence[int] | None = None,
        graphs: bool = True,
        pad: int = 0,
        max_group: int = 8,
        pack: int = 4096,
        mix: bool = True,
        share: bool = True,
        paged: bool | None = None,
        rows: int = 64,
        max_len: int | None = None,
        reserve: int | None = None,
    ) -> None:
        self.backend: _Backend = _backend_for(model, graphs, paged, rows, max_len)
        self.paged = self.backend.paged
        if self.paged and pack <= 0:
            raise ValueError("serving from pages packs the prompts: pack must be positive")
        # Pages kept free for the rows already decoding to grow into.
        self.reserve = max(1, self.backend.pages // 100) if reserve is None else max(0, reserve)
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
        # The largest pass that takes the decoding rows' step along: the
        # prompts that arrive while others decode are few, and each size
        # `step_packed` runs at is one more compiled pass at warmup.
        self.mix_size = self.pack_sizes[min(1, len(self.pack_sizes) - 1)] if self.mix else 0
        self.pad = pad
        self.max_group = max(1, max_group)
        self.share = share and bool(self.pack)
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
                # Twice: the first call compiles the pass, the second captures
                # it as a CUDA graph, which a pass of thousands of tokens takes
                # a while to do -- before the requests' clock starts.
                for _ in range(2):
                    self.backend.prefill_packed([[self.pad]], [0], [Sampling()], size, self.pad)
                    if self.mix and size <= self.mix_size:
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
            preempted=self._preempted,
        )
        return order, stats

    def load_weights(self, source: LinnetModule | Mapping[str, ArrayLike]) -> None:
        """Copies `source`'s weights into the model this engine serves, in
        place, between runs: a policy in training sampled with its latest
        weights. Compiled passes and CUDA graphs stay. PyTorch models only
        (`LinnetModule.copy_weights`)."""
        if self.busy:
            raise RuntimeError("the engine is still completing requests")
        model = getattr(self.backend, "model", None)
        copy = getattr(model, "copy_weights", None)
        if copy is None:
            raise ValueError("only a `linnet.torch` model's weights can be replaced")
        copy(source)

    # ---- step by step, for a server whose requests arrive while it runs

    def reset(self) -> None:
        """Forgets every request, waiting or in a row, and restarts the clock
        the completions' times count from."""
        self._waiting: deque[Completion] = deque()
        self._rows: list[Completion | None] = [None] * self.slots
        self._positions = [0] * self.slots
        self._queued: deque[_Queued] = deque()
        self._steps = self._prefills = self._preempted = 0
        self._start = time.perf_counter()
        # Paged: each row's pages in order, the positions its queued passes
        # will have written, and when it was admitted (the last goes first).
        # Page 0 takes the writes nobody reads.
        self._free_pages = list(range(self.backend.pages - 1, 0, -1)) if self.paged else []
        self._row_pages: list[list[int]] = [[] for _ in range(self.slots)]
        self._written = [0] * self.slots
        self._admission = [0] * self.slots
        self._admitted = 0
        if self.paged:
            for slot in range(self.slots):
                self.backend.assign(slot, [])

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
                self._release(slot)

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
        finished: list[Completion] = []
        if self.paged:
            # The rows decoding first: a page for each that steps into one.
            finished += self._grow()
        rows = self._rows
        decoding = any(row is not None for row in rows)
        free = [slot for slot in range(self.slots) if rows[slot] is None]
        count = min(len(free), len(self._waiting))
        if self.paged:
            count = self._fitting(count, decoding)
        admitted = [self._waiting.popleft() for _ in range(count)]
        admitted.sort(key=lambda c: len(_prompt(c)))
        placed = list(zip(free, admitted, strict=False))
        if self.paged:
            for slot, completion in placed:
                self._take_pages(slot, completion)
        # Prompts admitted together that are the same pass once; the rows
        # sharing one copy its cache rows (`share`).
        sharing: dict[int, list[tuple[int, Completion]]] = {}
        if self.share:
            first: dict[tuple[int, ...], int] = {}
            leaders: list[tuple[int, Completion]] = []
            for slot, completion in placed:
                key = tuple(_prompt(completion))
                if key in first:
                    sharing.setdefault(first[key], []).append((slot, completion))
                else:
                    first[key] = slot
                    leaders.append((slot, completion))
            placed = leaders
        passes = self._packed(placed)
        # With rows already decoding, the smallest pass carries their step,
        # if it is small enough.
        mixed = None
        if passes and decoding and self.mix:
            smallest = min(passes, key=lambda batch: sum(len(_prompt(c)) for _, c in batch))
            if sum(len(_prompt(c)) for _, c in smallest) <= self.mix_size:
                passes.remove(smallest)
                mixed = smallest
        for batch in passes:
            now = time.perf_counter() - self._start
            prompts = [_prompt(c) for _, c in batch]
            size = next(s for s in self.pack_sizes if s >= sum(len(p) for p in prompts))
            everyone, copies = _shared(batch, sharing)
            tokens = self.backend.prefill_packed(
                prompts,
                [slot for slot, _ in batch],
                [c.sampling for _, c in batch],
                size,
                self.pad,
                _top(c for _, c in everyone),
                copies,
            )
            self._prefills += 1
            for slot, completion in everyone:
                completion.admitted = now
                rows[slot] = completion
            owners = [(i, slot, c) for i, (slot, c) in enumerate(everyone)]
            self._queued.append(_Queued(tokens, owners, prompts=True))
        if self.pack:
            placed = []
        while placed:
            group = 1
            while group * 2 <= min(len(placed), self.max_group):
                group *= 2
            batch, placed = placed[:group], placed[group:]
            now = time.perf_counter() - self._start
            prompts = [_prompt(c) for _, c in batch]
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
            prompts = [_prompt(c) for _, c in mixed]
            size = next(s for s in self.pack_sizes if s >= sum(len(p) for p in prompts))
            # Every row in use steps but those this pass fills, which have no
            # token to step from yet.
            stepping = [(slot, slot, c) for slot, c in enumerate(rows) if c is not None]
            everyone, copies = _shared(mixed, sharing)
            prompted, stepped = self.backend.step_packed(
                prompts,
                [slot for slot, _ in mixed],
                [c.sampling for _, c in mixed],
                size,
                self.pad,
                mode(c.sampling for _, _, c in stepping),
                _top(c for _, c in everyone),
                _top(c for _, _, c in stepping),
                copies,
            )
            self._prefills += 1
            self._steps += 1
            for slot, completion in everyone:
                completion.admitted = now
                rows[slot] = completion
            owners = [(i, slot, c) for i, (slot, c) in enumerate(everyone)]
            self._queued.append(_Queued(prompted, owners, prompts=True))
            self._queued.append(_Queued(stepped, stepping, prompts=False))
            self._stepped(stepping)
            ahead = 2
        elif any(row is not None for row in rows):
            # One step for the whole batch; empty rows compute along. It
            # stays queued while the passes before it are read.
            owners = [(slot, slot, c) for slot, c in enumerate(rows) if c is not None]
            need = mode(c.sampling for _, _, c in owners)
            top = _top(c for _, _, c in owners)
            self._queued.append(_Queued(self.backend.decode(need, top), owners, prompts=False))
            self._steps += 1
            self._stepped(owners)
            ahead = 1
        while len(self._queued) > ahead:
            finished += self._read(self._queued.popleft())
        return finished

    # ---- pages

    def _fitting(self, count: int, decoding: bool) -> int:
        """How many of the first `count` waiting requests the free pages
        take, in order, keeping `reserve` free while rows decode."""
        budget = len(self._free_pages) - (self.reserve if decoding else 0)
        taken = 0
        for completion in list(self._waiting)[:count]:
            need = self._pages_for(len(_prompt(completion)))
            if need > budget:
                break
            budget -= need
            taken += 1
        return taken

    def _pages_for(self, length: int) -> int:
        """The pages a row needs for a prompt of `length` and its first step."""
        return length // self.backend.page_size + 1

    def _take_pages(self, slot: int, completion: Completion) -> None:
        length = len(_prompt(completion))
        pages = [self._free_pages.pop() for _ in range(self._pages_for(length))]
        self._row_pages[slot] = pages
        self._written[slot] = length
        self._admitted += 1
        self._admission[slot] = self._admitted
        self.backend.assign(slot, pages)

    def _stepped(self, owners: list[tuple[int, int, Completion]]) -> None:
        """A queued step writes one position of each of its rows; a row that
        reached the end, finished but not yet read, writes its last again,
        as the device keeps it there."""
        if self.paged:
            last = self.backend.max_seq - 1
            for _, slot, _ in owners:
                self._written[slot] = min(self._written[slot] + 1, last)

    def _grow(self) -> list[Completion]:
        """A page for every decoding row whose next step starts one. Short of
        pages, the passes still queued are read first (rows that finished
        give theirs back), and then the row admitted last waits again."""
        size = self.backend.page_size
        finished: list[Completion] = []
        while True:
            needy = [
                slot
                for slot, row in enumerate(self._rows)
                if row is not None and self._written[slot] // size >= len(self._row_pages[slot])
            ]
            if len(needy) <= len(self._free_pages):
                break
            if self._queued:
                while self._queued:
                    finished += self._read(self._queued.popleft())
                continue
            victim = max(
                (slot for slot, row in enumerate(self._rows) if row is not None),
                key=lambda slot: self._admission[slot],
            )
            self._preempt(victim)
        for slot in needy:
            self._row_pages[slot].append(self._free_pages.pop())
            self.backend.assign(slot, self._row_pages[slot])
        return finished

    def _preempt(self, slot: int) -> None:
        """Sends a row's request back to wait, first in line: its prompt and
        the tokens it has pass again once pages are free. Nothing queued may
        still read the row."""
        completion = self._rows[slot]
        assert completion is not None and not self._queued
        self._rows[slot] = None
        self._release(slot)
        completion.preempted += 1
        self._preempted += 1
        self._waiting.appendleft(completion)

    def _release(self, slot: int) -> None:
        """Gives a row's pages back; its steps write where nobody reads."""
        if not self.paged:
            return
        self._free_pages.extend(self._row_pages[slot])
        self._row_pages[slot] = []
        self._written[slot] = 0
        self.backend.assign(slot, [])

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
                if len(completion.tokens) == 1:
                    completion.first_token = now
                # After a preemption the tokens before passed with the prompt.
                self._positions[slot] = len(completion.request.prompt) + len(completion.tokens) - 1
            else:
                self._positions[slot] += 1
            if self._finished(completion, self._positions[slot]):
                completion.finished = now
                self._rows[slot] = None
                self._release(slot)
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
        for slot, completion in sorted(placed, key=lambda sc: -len(_prompt(sc[1]))):
            length = len(_prompt(completion))
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


def _prompt(completion: Completion) -> list[int]:
    """What a request passes as its prompt: the prompt, and after a
    preemption the tokens it had."""
    return [*completion.request.prompt, *completion.tokens]


def _shared(
    batch: list[tuple[int, Completion]], sharing: dict[int, list[tuple[int, Completion]]]
) -> tuple[list[tuple[int, Completion]], _Copies]:
    """A pass's rows, its prompts' then those sharing them, in the order the
    backend returns their tokens; and per prompt, the rows sharing it."""
    extra = [pair for slot, _ in batch for pair in sharing.get(slot, [])]
    copies = [[(s, c.sampling) for s, c in sharing.get(slot, [])] for slot, _ in batch]
    return [*batch, *extra], copies


def _record(held: list[Sampling], slots: list[int], sampling: list[Sampling]) -> bool:
    """Sets rows' sampling in `held`; whether any changed, which with no
    rows given (the first upload) it always has."""
    if slots and all(held[s] == x for s, x in zip(slots, sampling, strict=True)):
        return False
    for slot, row in zip(slots, sampling, strict=True):
        held[slot] = row
    return True


def _top(completions: Iterable[Completion]) -> int:
    """What a pass computes of log-probabilities (see `_Backend`)."""
    wanted = [c.request.logprobs for c in completions if c.request.logprobs is not None]
    return max(wanted, default=-1)


def _logprobs_torch(
    logits: torch.Tensor, produced: torch.Tensor, top: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each row's token's log-probability and its `top` most likely tokens,
    from the model's logits, before temperature and filtering."""
    scores = logits.float().log_softmax(-1)
    chosen = scores.gather(-1, produced.long().reshape(-1, 1)).reshape(-1)
    values, ids = scores.topk(max(top, 1), -1)
    return chosen, ids[:, :top], values[:, :top]


def _logprobs_numpy(logits: NDArray[np.generic], produced: ArrayLike, top: int) -> _Logprobs:
    import numpy as numpy_module

    np = numpy_module
    scores: np.ndarray[tuple[int, int], np.dtype[np.floating]] = logits.astype(np.float32)
    peak = scores.max(-1, keepdims=True)
    scores = scores - peak - np.log(np.exp(scores - peak).sum(-1, keepdims=True))
    rows = np.arange(scores.shape[0], dtype=np.intp)
    chosen = scores[rows, np.asarray(produced).reshape(-1)]
    ids = np.argsort(-scores, axis=-1, kind="stable")[:, :top]
    values = np.take_along_axis(scores, ids, -1)
    return chosen.tolist(), ids.tolist(), values.tolist()


class _Generics(Protocol):
    """A model, as far as `_cache_generics` reads it."""

    @property
    def generics(self) -> Mapping[str, int | str]: ...


def _backend_for(
    model: _Model, graphs: bool, paged: bool | None, rows: int, max_len: int | None
) -> _Backend:
    # Told apart by module rather than `isinstance`, which would import every
    # framework: hence the casts.
    module = type(model).__module__
    if module.startswith(("linnet.jax", "linnet.onnx")):
        if paged:
            raise ValueError("serving from pages runs with the PyTorch backend")
        return (
            _JaxBackend(cast("LinnetModel", model))
            if module.startswith("linnet.jax")
            else _OnnxBackend(cast("OnnxModel", model))
        )
    return _TorchBackend(cast("LinnetModule", model), graphs, paged, rows, max_len)


def _root_dim(model: LinnetModule, name: str) -> int | None:
    """The value of the root block's dimension generic `name`, its default
    when the model was loaded without one."""
    for generic in model.program.root.generics:
        if generic.name == name and generic.kind == "dim":
            return int(model.root.env.dims[generic.id])
    return None


def _cache_generics(model: _Generics) -> tuple[int, int]:
    generics = dict(model.generics)
    try:
        return int(generics["Batch"]), int(generics["MaxSeq"])
    except KeyError as missing:
        raise ValueError(
            f"the model's generics do not bind {missing}; serving needs both"
        ) from None


class _TorchBackend:
    def __init__(
        self,
        model: LinnetModule,
        graphs: bool,
        paged: bool | None,
        rows: int,
        max_len: int | None,
    ) -> None:
        import torch

        self.torch = torch
        self.model = model
        batch, length = _cache_generics(model)
        if paged is None:
            paged = batch == 1 and all(
                e in model.entries for e in ("prefill_paged", "decode_paged")
            )
        self.paged = paged
        needed = ("prefill_paged", "decode_paged") if paged else ("prefill_slots", "decode_rows")
        for entry in needed:
            if entry not in model.entries:
                raise ValueError(f"the model has no `{entry}` entry, which serving needs")
        self.device = next(iter(model.parameters())).device
        self.page_size = 1
        self.pages = 0
        if paged:
            if batch != 1:
                raise ValueError(
                    "serving from pages keeps them in the caches' one row: load with Batch 1"
                )
            size = _root_dim(model, "PageSize")
            if size is None or length % size:
                raise ValueError("serving from pages needs a `PageSize` that divides `MaxSeq`")
            self.page_size, self.pages = size, length // size
            # Every page but the first, which takes the writes nobody reads;
            # by default up to 8192 positions, as one prompt is one pass.
            widest = (self.pages - 1) * size
            longest = min(widest, 8192) if max_len is None else max_len
            if not 0 < longest <= widest:
                raise ValueError(f"max_len must be between 1 and {widest} for this pool")
            self.slots = rows
            self.max_seq = -(-longest // size) * size
            self.table = torch.zeros(
                rows, self.max_seq // size, dtype=torch.int32, device=self.device
            )
            self._pages: list[list[int]] = [[] for _ in range(rows)]
            self._changed: set[int] = set()
            self.packs = True
            self.mixes = "step_paged" in model.entries
        else:
            self.slots, self.max_seq = batch, length
            self.packs = "prefill_packed" in model.entries
            self.mixes = self.packs and "step_packed" in model.entries
        self.step_compile: bool | str = "reduce-overhead" if graphs else True
        self.tokens = torch.zeros(self.slots, 1, dtype=torch.int32, device=self.device)
        self.positions = torch.zeros(self.slots, dtype=torch.int32, device=self.device)
        self.sampling = [Sampling()] * self.slots
        self._row_states: list[tuple[torch.nn.Module, str]] | None = None
        self._hold([], [])
        # Compiled, drawing reads the logits a few times, not once for each
        # step of the hash; the batch dimension varies without recompiling.
        self._draw = draw_torch
        if self.device.type == "cuda":
            self._draw = torch.compile(draw_torch, dynamic=None)

    def _put(
        self, values: Sequence[float] | Sequence[Sequence[int]], dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        host = self.torch.tensor(values, dtype=dtype or self.torch.int32)
        if self.device.type != "cuda":
            return host.to(self.device)
        # From pinned memory the copy is queued behind the work before it;
        # from pageable memory it would wait for that work to finish.
        return host.pin_memory().to(self.device, non_blocking=True)

    def _hold(self, slots: list[int], sampling: list[Sampling]) -> None:
        """Puts rows' sampling on the device, all of it, when any changed."""
        torch = self.torch
        if not _record(self.sampling, slots, sampling):
            return
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
        rows, at = self._put(slots), self._put(lengths)
        # A serving entry has one result: the logits.
        logits = cast(
            "torch.Tensor",
            self.model.run_entry("prefill_slots", [self._put(tokens), rows, at], compile=True),
        )
        first = self._admit(logits, rows, at, slots, sampling)
        return _TorchTokens(first, _logprobs_torch(logits, first, top) if top >= 0 else None)

    def _pack(
        self, prompts: list[list[int]], slots: list[int], size: int, pad: int
    ) -> list[torch.Tensor]:
        """A packed pass's `tokens`, `rows`, `positions`, `segments` and
        `last`, on the device; paged, `tokens`, `positions`, `segments`,
        each token's place in the pool, and `last`."""
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
        # cache's last position, which a request stops before it reaches
        # (paged, at the start of the pool's first page, which nobody reads).
        extra = size - len(tokens)
        tokens += [pad] * extra
        rows += [slots[0]] * extra
        positions += [self.max_seq - 1] * extra
        segments += [-1] * extra
        last += [last[-1]] * (self.slots - len(last))
        if self.paged:
            places = [self._place(row, at) for row, at in zip(rows, positions, strict=True)]
            places[len(places) - extra :] = [0] * extra
            packed = self._put([tokens, positions, segments, places])
            return [*packed.unbind(0), self._put(last)]
        packed = self._put([tokens, rows, positions, segments])
        return [packed[0], packed[1], packed[2], packed[3], self._put(last)]

    def _place(self, row: int, at: int) -> int:
        """Where position `at` of a row lies in the pool (paged)."""
        size = self.page_size
        pages = self._pages[row]
        page = pages[at // size] if at // size < len(pages) else 0
        return page * size + at % size

    def assign(self, row: int, pages: list[int]) -> None:
        if self.paged:
            self._pages[row] = list(pages)
            self._changed.add(row)

    def _table(self) -> torch.Tensor:
        """The page table on the device, its changed rows written first, in
        the order of the passes queued."""
        if self._changed:
            rows = sorted(self._changed)
            width = self.table.shape[1]
            values = [self._pages[r] + [0] * (width - len(self._pages[r])) for r in rows]
            self.table.index_copy_(0, self._put(rows).long(), self._put(values))
            self._changed.clear()
        return self.table

    def _lengths(
        self, prompts: list[list[int]], slots: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prompts' rows and lengths, on the device."""
        return self._put(slots), self._put([len(prompt) for prompt in prompts])

    def _admit(
        self,
        logits: torch.Tensor,
        rows: torch.Tensor,
        at: torch.Tensor,
        slots: list[int],
        sampling: list[Sampling],
    ) -> torch.Tensor:
        """Draws the token after each prompt from its logits and sets its row
        (`rows`, `slots` on the device) to it, at the position after the
        prompt (`at`)."""
        torch = self.torch
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
        return first

    def prefill_packed(
        self,
        prompts: list[list[int]],
        slots: list[int],
        sampling: list[Sampling],
        size: int,
        pad: int,
        top: int = -1,
        copies: _Copies | None = None,
    ) -> _Tokens:
        # A few pass sizes, each compiled like the step (and replayed as a CUDA
        # graph): FlexAttention runs only under `torch.compile`, and a pass
        # of thousands of tokens has as many kernels as the step.
        logits = cast(
            "torch.Tensor",
            self.model.run_entry(
                "prefill_paged" if self.paged else "prefill_packed",
                self._pack(prompts, slots, size, pad),
                compile=self.step_compile,
            ),
        )[: len(prompts)]
        logits, prompts, slots, sampling = self._share(logits, prompts, slots, sampling, copies)
        first = self._admit(logits, *self._lengths(prompts, slots), slots, sampling)
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
        copies: _Copies | None = None,
    ) -> tuple[_Tokens, _Tokens]:
        """`prefill_packed` and `decode` in one pass. The rows the prompts
        fill step along from no token of theirs: their step is written at
        the cache's last position, which a request stops before it reaches,
        and the prompts' tokens then set them."""
        index = self._put(slots).long()
        positions = self.positions.clone()
        positions[index] = self.max_seq - 1
        packed = self._pack(prompts, slots, size, pad)
        if self.paged:
            entry, inputs = "step_paged", [*packed, self.tokens, positions, self._table()]
        else:
            entry, inputs = "step_packed", [*packed, self.tokens, positions]
        logits = cast(
            "torch.Tensor", self.model.run_entry(entry, inputs, compile=self.step_compile)
        )
        stepped = self._advance(logits[self.slots :], need)
        prompted, prompts, slots, sampling = self._share(
            logits[: len(prompts)], prompts, slots, sampling, copies
        )
        first = self._admit(prompted, *self._lengths(prompts, slots), slots, sampling)
        return (
            _TorchTokens(first, _logprobs_torch(prompted, first, top) if top >= 0 else None),
            _TorchTokens(
                stepped,
                _logprobs_torch(logits[self.slots :], stepped, step_top) if step_top >= 0 else None,
            ),
        )

    def _share(
        self,
        logits: torch.Tensor,
        prompts: list[list[int]],
        slots: list[int],
        sampling: list[Sampling],
        copies: _Copies | None,
    ) -> tuple[torch.Tensor, list[list[int]], list[int], list[Sampling]]:
        """The pass's prompts, then the rows that share them: each copies its
        prompt's cache rows and takes its logits."""
        if not copies or not any(copies):
            return logits, prompts, slots, sampling
        order = list(range(len(prompts)))
        sources: list[int] = []
        targets: list[int] = []
        shared = list(sampling)
        for i, extra in enumerate(copies):
            for slot, row in extra:
                order.append(i)
                sources.append(slots[i])
                targets.append(slot)
                shared.append(row)
        self._copy_rows(sources, targets)
        index = self._put(order).long()
        return logits[index], [prompts[i] for i in order], [*slots, *targets], shared

    def _copy_rows(self, sources: list[int], targets: list[int]) -> None:
        """Copies whole rows of every state whose first axis is the row (the
        KV caches), in place, so captured graphs keep their addresses; paged,
        each source row's pages into the target's."""
        if self.paged:
            self._copy_pages(sources, targets)
            return
        if self._row_states is None:
            self._row_states = [
                (module, leaf)
                for module in self.model.modules()
                for leaf in getattr(module, "state_names", ())
                if isinstance(getattr(module, leaf, None), self.torch.Tensor)
                and getattr(module, leaf).dim() > 0
                and getattr(module, leaf).shape[0] == self.slots
            ]
        source, target = self._put(sources).long(), self._put(targets).long()
        for module, leaf in self._row_states:
            state = getattr(module, leaf)
            state.index_copy_(0, target, state.index_select(0, source))

    def _copy_pages(self, sources: list[int], targets: list[int]) -> None:
        size = self.page_size
        pool = self.pages * size
        if self._row_states is None:
            self._row_states = [
                (module, leaf)
                for module in self.model.modules()
                for leaf in getattr(module, "state_names", ())
                if isinstance(getattr(module, leaf, None), self.torch.Tensor)
                and getattr(module, leaf).dim() == 4
                and getattr(module, leaf).shape[0] == 1
                and getattr(module, leaf).shape[2] == pool
            ]
        source: list[int] = []
        target: list[int] = []
        for row, other in zip(sources, targets, strict=True):
            for page, copy in zip(self._pages[row], self._pages[other], strict=False):
                source += range(page * size, (page + 1) * size)
                target += range(copy * size, (copy + 1) * size)
        if not source:
            return
        source_at, target_at = self._put(source).long(), self._put(target).long()
        for module, leaf in self._row_states:
            state = getattr(module, leaf)
            state.index_copy_(2, target_at, state.index_select(2, source_at))

    def _advance(self, logits: torch.Tensor, need: int) -> torch.Tensor:
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
        if self.paged:
            entry, inputs = "decode_paged", [self.tokens, self.positions, self._table()]
        else:
            entry, inputs = "decode_rows", [self.tokens, self.positions]
        logits = cast(
            "torch.Tensor", self.model.run_entry(entry, inputs, compile=self.step_compile)
        )
        produced = self._advance(logits, need)
        return _TorchTokens(produced, _logprobs_torch(logits, produced, top) if top >= 0 else None)


class _TorchTokens:
    def __init__(
        self,
        values: torch.Tensor,
        logprobs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        import torch

        self.ready: torch.cuda.Event | None = None
        parts = [values, *(logprobs or ())]
        if values.device.type == "cuda":
            host = [torch.empty(p.shape, dtype=p.dtype, pin_memory=True) for p in parts]
            for target, part in zip(host, parts, strict=True):
                target.copy_(part, non_blocking=True)
            parts = host
            self.ready = torch.cuda.Event()
            self.ready.record()
        self.values: torch.Tensor = parts[0]
        self.extra: list[torch.Tensor] = parts[1:]

    def tolist(self) -> list[int]:
        if self.ready is not None:
            self.ready.synchronize()
        # PyTorch's stubs give `tolist` an unparameterized `list`.
        return self.values.tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    def logprobs(self) -> _Logprobs | None:
        if not self.extra:
            return None
        if self.ready is not None:
            self.ready.synchronize()
        chosen, ids, values = self.extra
        return chosen.tolist(), ids.tolist(), values.tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


class _Grouped:
    """A backend whose prompt passes are grouped and padded (`prefill_slots`
    only): XLA and ONNX Runtime, where a packed pass's mask would cost its
    whole square."""

    packs = False
    mixes = False
    paged = False
    page_size = 1
    pages = 0

    def assign(self, row: int, pages: list[int]) -> None:
        pass

    def prefill_packed(
        self,
        prompts: list[list[int]],
        slots: list[int],
        sampling: list[Sampling],
        size: int,
        pad: int,
        top: int = -1,
        copies: _Copies | None = None,
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
        copies: _Copies | None = None,
    ) -> tuple[_Tokens, _Tokens]:
        raise NotImplementedError("packed prompt passes run with the PyTorch backend")


class _JaxBackend(_Grouped):
    def __init__(self, model: LinnetModel) -> None:
        import jax as jax_module
        import jax.numpy as jnp_module
        import numpy as np

        for entry in ("prefill_slots", "decode_rows"):
            if entry not in model.entries:
                raise ValueError(f"the model has no `{entry}` entry, which serving needs")
        jax = jax_module
        jnp = jnp_module
        self.jnp = jnp
        self.np = np
        self.model = model
        self.slots, self.max_seq = _cache_generics(model)
        # JAX's stubs leave `zeros`, `asarray` and `jit` partially unknown.
        self.tokens: jax.Array = jnp.zeros((self.slots, 1), jnp.int32)  # pyright: ignore[reportUnknownMemberType]
        self.positions: jax.Array = jnp.zeros((self.slots,), jnp.int32)  # pyright: ignore[reportUnknownMemberType]
        self.sampling = [Sampling()] * self.slots
        self._hold([], [])
        last = self.max_seq - 1

        def logprobs(logits: jax.Array, produced: jax.Array, top: int) -> tuple[jax.Array, ...]:
            if top < 0:
                return ()
            scores = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            chosen = jnp.take_along_axis(scores, produced[:, None], -1)[:, 0]
            values, ids = jax.lax.top_k(scores, max(top, 1))
            return chosen, ids[:, :top], values[:, :top]

        # `held` is each row's temperature, top-k, top-p, and seed key.
        def admit(
            tokens: jax.Array,
            positions: jax.Array,
            logits: jax.Array,
            rows: jax.Array,
            lengths: jax.Array,
            held: _Held,
            need: int,
            top: int,
        ) -> _Moved:
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

        def advance(
            logits: jax.Array, positions: jax.Array, held: _Held, need: int, top: int
        ) -> _Moved:
            temperature, top_k, top_p, keys = held
            produced = draw_jax(logits, temperature, top_k, top_p, keys, positions + 1, need)
            return (
                produced.reshape(-1, 1),
                jnp.minimum(positions + 1, last),
                produced,
                logprobs(logits, produced, top),
            )

        # Jitted, each takes the arguments of the function it wraps.
        self._admit: Callable[..., _Moved] = jax.jit(admit, static_argnames=("need", "top"))  # pyright: ignore[reportUnknownMemberType]
        self._advance: Callable[..., _Moved] = jax.jit(advance, static_argnames=("need", "top"))  # pyright: ignore[reportUnknownMemberType]

    def _put(
        self, values: Sequence[float] | Sequence[Sequence[int]], dtype: DTypeLike | None = None
    ) -> jax.Array:
        return self.jnp.asarray(self.np.asarray(values, dtype=dtype or self.np.int32))  # pyright: ignore[reportUnknownMemberType]

    def _hold(self, slots: list[int], sampling: list[Sampling]) -> None:
        """Puts rows' sampling on the device, all of it, when any changed."""
        np = self.np
        if not _record(self.sampling, slots, sampling):
            return
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
    def __init__(self, values: jax.Array, logprobs: tuple[jax.Array, ...] = ()) -> None:
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

    def __init__(self, model: OnnxModel) -> None:
        import numpy as np

        for entry in ("prefill_slots", "decode_rows"):
            if entry not in model.entries:
                raise ValueError(f"the model has no `{entry}` entry, which serving needs")
        self.np = np
        self.model = model
        self.slots, self.max_seq = _cache_generics(model)
        self.tokens: NDArray[np.integer] = np.zeros((self.slots, 1), dtype=np.int32)
        self.positions: NDArray[np.integer] = np.zeros(self.slots, dtype=np.int32)
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
        # One result: the logits, or with `argmax` each row's token.
        out = cast(
            "NDArray[np.generic]",
            self.model.run_entry(
                "prefill_slots",
                [
                    np.asarray(tokens, dtype=np.int32),
                    np.asarray(slots, dtype=np.int32),
                    np.asarray(lengths, dtype=np.int32),
                ],
                argmax=on_device,
            ),
        )
        first = (
            out.reshape(-1)
            if on_device
            else draw_numpy(cast("NDArray[np.floating]", out), sampling, lengths, need)
        )
        for slot, row in zip(slots, sampling, strict=True):
            self.sampling[slot] = row
        self.tokens[slots, 0] = first
        self.positions[slots] = lengths
        return _HostTokens(first, _logprobs_numpy(out, first, top) if top >= 0 else None)

    def decode(self, need: int, top: int = -1) -> _Tokens:
        np = self.np
        on_device = not need and top < 0
        out = cast(
            "NDArray[np.generic]",
            self.model.run_entry(
                "decode_rows", [self.tokens, self.positions], argmax=on_device, cuda_graph=True
            ),
        )
        if on_device:
            produced = out.reshape(-1)
        else:
            produced = draw_numpy(
                cast("NDArray[np.floating]", out), self.sampling, self.positions + 1, need
            )
        self.tokens = produced.astype(np.int32).reshape(self.slots, 1)
        self.positions = np.minimum(self.positions + 1, self.max_seq - 1)
        return _HostTokens(produced, _logprobs_numpy(out, produced, top) if top >= 0 else None)


class _HostTokens:
    """A pass's tokens already on the host, and its log-probabilities when
    it computed them."""

    def __init__(self, values: NDArray[np.generic], logprobs: _Logprobs | None) -> None:
        self.values = values
        self.computed = logprobs

    def tolist(self) -> list[int]:
        return self.values.tolist()

    def logprobs(self) -> _Logprobs | None:
        return self.computed


__all__ = ["Completion", "Engine", "Request", "Sampling", "Stats"]
