"""Embeddings served together: the inputs of every request waiting pass
through a card's embedding entry in shared batches.

A card that serves embeddings has one of

- `embed_padded<B, S>(tokens: [B, S], token_types: [B, S], lengths: [B])
  -> [B, D]`: sequence `b` is its first `lengths[b]` tokens, the rest
  padding it neither attends to nor pools (the zoo's all-MiniLM-L6-v2);
- `embed<B, S>(tokens: [B, S], token_types: [B, S]) -> [B, D]`, which
  pools every position, so only inputs of one length share a batch.

Either may leave out `token_types` (a BERT's segments; every input here is
one segment, all zeros), and each pools and normalizes as the checkpoint's
own pipeline does. With `embed_padded`, inputs are padded to a few lengths
(powers of two up to `max_len`) and batches to powers of two up to `batch`,
so the backend compiles a handful of shapes, all of them in `warmup`. With
`embed`, each length and batch size compiles when first seen.

    from linnet.serve.embeddings import Embedder

    embedder = Embedder(model, max_len=512)
    embedder.warmup()
    embedder.start()
    vectors = embedder.embed([[101, 7592, 102], [101, 2088, 102]])
"""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..torch import LinnetModule


class EmbeddingError(ValueError):
    """An input the embedder refuses, with the reason."""


@dataclass
class _Job:
    """One request's inputs, and their vectors once the thread has them."""

    inputs: list[list[int]]
    vectors: list[list[float]] = field(default_factory=list["list[float]"])
    error: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


def _powers(limit: int, start: int = 1) -> list[int]:
    """Powers of two from `start` below `limit`, then `limit`."""
    sizes: list[int] = []
    size = start
    while size < limit:
        sizes.append(size)
        size *= 2
    return [*sizes, limit]


class Embedder:
    """Runs `model`'s embedding entry for requests from many threads, on one
    thread of its own: `embed` hands it a request's inputs and waits for
    their vectors. Inputs are token ids, at most `max_len` of them; a pass
    takes up to `batch` of them."""

    def __init__(self, model: LinnetModule, *, max_len: int, batch: int = 32) -> None:
        if "embed_padded" in model.entries:
            self.entry, self.padded = "embed_padded", True
        elif "embed" in model.entries:
            self.entry, self.padded = "embed", False
        else:
            raise EmbeddingError(
                f"the model has no `embed_padded` or `embed` entry "
                f"(it has {', '.join(model.entries)})"
            )
        inputs = len(model.entries[self.entry].params) - self.padded
        if inputs not in (1, 2):
            raise EmbeddingError(f"`{self.entry}` takes token ids, and optionally token types")
        if batch < 1 or max_len < 1:
            raise EmbeddingError("`batch` and `max_len` are at least 1")
        self.model = model
        self.max_len = max_len
        self.batch = batch
        self.token_types = inputs == 2
        self.lengths = _powers(max_len, start=min(8, max_len))
        self.sizes = _powers(batch)
        self.failed: str | None = None  # the model's error, once it has one
        self._inbox: queue.Queue[_Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="embedder", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._inbox.put(None)

    def warmup(self) -> None:
        """Compiles every padded shape (with `embed`, one input of 8
        tokens), so requests do not wait for the backend to."""
        if not self.padded:
            self._pass([[0] * min(8, self.max_len)])
            return
        for length in self.lengths:
            for size in self.sizes:
                self._pass([[0] * length] * size)

    def check(self, inputs: Sequence[Sequence[int]]) -> None:
        """Refuses what the entry cannot take, before it is queued."""
        if not inputs:
            raise EmbeddingError("`input` is empty")
        for index, ids in enumerate(inputs):
            if not ids:
                raise EmbeddingError(f"input {index} is empty")
            if len(ids) > self.max_len:
                raise EmbeddingError(
                    f"input {index} has {len(ids)} tokens, more than the {self.max_len} served"
                )

    def embed(self, inputs: Sequence[Sequence[int]]) -> list[list[float]]:
        """The vectors of `inputs`, in their order, passed with whatever
        other requests are waiting."""
        if self.failed is not None:
            raise RuntimeError("the embedder has failed:\n" + self.failed)
        self.check(inputs)
        job = _Job([list(ids) for ids in inputs])
        self._inbox.put(job)
        job.done.wait()
        if job.error is not None:
            raise RuntimeError("the embedder has failed:\n" + job.error)
        return job.vectors

    # ---- the embedder's thread

    def _run(self) -> None:
        while True:
            job = self._inbox.get()
            if job is None:
                return
            jobs = [job]
            # Everything waiting goes in the same passes.
            while True:
                try:
                    more = self._inbox.get_nowait()
                except queue.Empty:
                    break
                if more is None:
                    self._inbox.put(None)
                    break
                jobs.append(more)
            try:
                self._serve(jobs)
            except Exception:
                # A model that failed once (the device lost, say) is not
                # trusted again: these requests and every later one get the
                # error.
                self.failed = traceback.format_exc()
            for one in jobs:
                if self.failed is not None:
                    one.error = self.failed
                one.done.set()

    def _serve(self, jobs: list[_Job]) -> None:
        # Inputs that share a pass: one padded length, or one exact length.
        groups: dict[int, list[tuple[_Job, int]]] = {}
        for job in jobs:
            job.vectors = [[] for _ in job.inputs]
            for index, ids in enumerate(job.inputs):
                length = self._padded_length(len(ids)) if self.padded else len(ids)
                groups.setdefault(length, []).append((job, index))
        for places in groups.values():
            for start in range(0, len(places), self.batch):
                group = places[start : start + self.batch]
                vectors = self._pass([job.inputs[index] for job, index in group])
                for (job, index), vector in zip(group, vectors, strict=True):
                    job.vectors[index] = vector

    def _padded_length(self, length: int) -> int:
        return next(size for size in self.lengths if size >= length)

    def _pass(self, inputs: list[list[int]]) -> list[list[float]]:
        """One pass of the entry over `inputs`: with `embed_padded`, padded
        to the next length and batch size (the extra rows one token long,
        their vectors dropped)."""
        import torch

        device = self.model.interpreter.device
        count = len(inputs)
        rows = inputs
        if self.padded:
            length = self._padded_length(max(len(ids) for ids in inputs))
            size = next(size for size in self.sizes if size >= count)
            rows = [ids + [0] * (length - len(ids)) for ids in inputs]
            rows += [[0] * length] * (size - count)
        tokens = torch.tensor(rows, dtype=torch.int32, device=device)
        operands = [tokens, torch.zeros_like(tokens)] if self.token_types else [tokens]
        if self.padded:
            lengths = [len(ids) for ids in inputs] + [1] * (len(rows) - count)
            operands.append(torch.tensor(lengths, dtype=torch.int32, device=device))
        with torch.inference_mode():
            vectors = self.model.run_entry(self.entry, operands)
        assert isinstance(vectors, torch.Tensor)
        host = vectors[:count].float().cpu().numpy()
        return cast("list[list[float]]", host.tolist())
