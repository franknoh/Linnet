"""An HTTP front for `Engine` that speaks OpenAI's API: `GET /v1/models`,
`POST /v1/completions`, `POST /v1/chat/completions` (streamed as server-sent
events with `"stream": true`), and `GET /health`.

    python -m linnet.serve llama-3.1-8b-instruct --batch 32 --max-seq 4096

One thread runs the engine; each request's handler thread hands it the
request and reads its tokens back as they are produced. A tokenizer turns
text into token ids and back -- one from Transformers, or anything with the
same `encode`, `decode`, and `apply_chat_template`.

Requests take `max_tokens` (`max_completion_tokens` in a chat), `temperature`
(1 unless given, as OpenAI has it), `top_p`, `top_k` (0 or -1 for every
token), `seed`, `stop` (a string or up to four), `n` (choices, each its own
request; with a `seed`, choice `i` draws with `seed + i`), and `stream`, with
`stream_options.include_usage`. Log probabilities come as OpenAI has them:
`logprobs` (a number of alternatives, up to 20) for completions, `logprobs`
and `top_logprobs` for a chat, of the model's distribution before
temperature and filtering. `echo` puts the prompt before a completion's
text; with `logprobs` it is refused, as the prompt's own log probabilities
are not computed. Other fields are ignored.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol, TypeAlias, TypeVar, cast

from . import Completion, Engine, Request
from .sampling import Sampling

# A JSON value: a request's body as `json.loads` reads it, or a response's
# as `json.dumps` writes it.
_Json: TypeAlias = "bool | int | float | str | Sequence[_Json] | Mapping[str, _Json] | None"

_T = TypeVar("_T")


class Tokenizer(Protocol):
    """The part of a Transformers tokenizer the server uses."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...
    def apply_chat_template(
        self, conversation: list[_Json], *, tokenize: bool = ..., add_generation_prompt: bool = ...
    ) -> object: ...


class RequestError(ValueError):
    """A request the server refuses, with the reason sent back."""


_FINISH = {"eos": "stop", "stop": "stop", "length": "length", "cache": "length"}


# What the engine thread sends a handler for each token read: the choice it
# belongs to, the token, the request's finish reason (empty until its last),
# and its log-probability with the alternatives when they were asked for. If
# the engine failed, the token is "" and the reason is the error.
_Event = tuple[int, int | str, str, tuple[float, list[tuple[int, float]]] | None]


@dataclass
class _Pending:
    """A request between a handler and the engine thread, one per choice;
    the choices of a request share `events`."""

    request: Request
    events: queue.Queue[_Event]
    index: int = 0
    completion: Completion | None = None


class Server:
    """Serves `engine` over HTTP at `host`:`port`, as the model `name`.
    `eos` ends every request (the tokenizer's end-of-sequence token and the
    model's generation config, say)."""

    def __init__(
        self,
        engine: Engine,
        tokenizer: Tokenizer,
        *,
        name: str,
        eos: Iterable[int] = (),
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.name = name
        self.eos = frozenset(eos)
        self._inbox: queue.Queue[tuple[str, _Pending]] = queue.Queue()
        self._active: list[_Pending] = []
        self.failed: str | None = None  # the engine's error, once it has one
        self._stopping = threading.Event()
        self._http = ThreadingHTTPServer((host, port), _handler(self))
        self._http.daemon_threads = True
        self._engine_thread = threading.Thread(target=self._run, name="engine", daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._http.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        """Serves until `shutdown` (or an interrupt)."""
        self._engine_thread.start()
        try:
            self._http.serve_forever()
        finally:
            self._stopping.set()
            self._http.server_close()

    def shutdown(self) -> None:
        """Stops `serve_forever`, from another thread."""
        self._stopping.set()
        self._http.shutdown()

    # ---- the engine thread

    def _run(self) -> None:
        engine = self.engine
        while not self._stopping.is_set():
            try:
                self._take(wait=not engine.busy)
                if engine.busy:
                    engine.step()
                    self._active = [p for p in self._active if not _done(p)]
            except Exception:
                # An engine that failed once (the device lost, say) is not
                # trusted again: every request waiting on it gets the error.
                self.failed = traceback.format_exc()
                for pending in self._active:
                    pending.events.put((pending.index, "", self.failed, None))
                self._active.clear()
                return

    def _take(self, wait: bool) -> None:
        """Submits or cancels what handlers sent; waits for something if
        `wait`."""
        try:
            kind, pending = self._inbox.get(timeout=0.2) if wait else self._inbox.get_nowait()
        except queue.Empty:
            return
        while True:
            if kind == "submit":
                pending.completion = self.engine.submit(pending.request)
                self._active.append(pending)
            elif pending.completion is not None:
                self.engine.cancel(pending.completion)
            try:
                kind, pending = self._inbox.get_nowait()
            except queue.Empty:
                return

    # ---- handlers

    def generate(
        self, prompt: list[int], body: Mapping[str, _Json], limit: int, chat: bool
    ) -> _Generation:
        """Hands a request to the engine, one per choice; their tokens and
        text come from the returned generation as they are produced."""
        if self.failed is not None:
            raise RuntimeError("the engine has failed:\n" + self.failed)
        if not prompt:
            raise RequestError("the prompt is empty")
        room = self.engine.backend.max_seq - len(prompt)
        if room <= 0:
            raise RequestError(
                f"the prompt's {len(prompt)} tokens leave no room in a cache of "
                f"{self.engine.backend.max_seq} positions"
            )
        choices = int(_given(body, "n", 1))
        if not 1 <= choices <= 128:
            raise RequestError("`n` is between 1 and 128")
        wanted = _logprobs_wanted(body, chat)
        if body.get("echo") and wanted is not None:
            raise RequestError(
                "`echo` with `logprobs` needs the prompt's log probabilities, "
                "which are not computed"
            )
        stop = cast("str | list[_Json]", body.get("stop") or [])
        stops: list[_Json] = [stop] if isinstance(stop, str) else list(stop)
        if len(stops) > 4 or not all(isinstance(s, str) and s for s in stops):
            raise RequestError("`stop` is a string or up to four")
        seed = cast("int | None", body.get("seed"))
        events: queue.Queue[_Event] = queue.Queue()
        pending: list[_Pending] = []
        for index in range(choices):

            def on_token(completion: Completion, index: int = index) -> None:
                logprob = None
                if wanted is not None and completion.logprobs:
                    logprob = (completion.logprobs[-1], completion.top_logprobs[-1])
                events.put((index, completion.tokens[-1], completion.reason, logprob))

            request = Request(
                prompt=prompt,
                max_new_tokens=min(int(_given(body, "max_tokens", limit)), room),
                eos=self.eos,
                temperature=float(_given(body, "temperature", 1.0)),
                top_k=max(int(_given(body, "top_k", 0)), 0),
                top_p=float(_given(body, "top_p", 1.0)),
                seed=None if seed is None else int(seed) + index,
                on_token=on_token,
                logprobs=wanted,
            )
            if request.max_new_tokens < 1:
                raise RequestError("`max_tokens` must be at least 1")
            try:
                Sampling(request.temperature, request.top_k, request.top_p).check()
            except ValueError as error:
                raise RequestError(str(error)) from None
            pending.append(_Pending(request, events, index))
        for one in pending:
            self._inbox.put(("submit", one))
        texts = [_Text(self.tokenizer, prompt) for _ in pending]
        return _Generation(self, pending, texts, cast("list[str]", stops), events)

    def cancel(self, pending: _Pending) -> None:
        self._inbox.put(("cancel", pending))


def _done(pending: _Pending) -> bool:
    return pending.completion is not None and bool(pending.completion.reason)


def _given(body: Mapping[str, _Json], key: str, default: _T) -> _T:
    value = body.get(key)
    # Typed as `default` is, as the API has it: the caller's `int` or `float`
    # converts it and refuses a field that does not convert (a 400).
    return default if value is None else cast("_T", value)


class _Text:
    """The text of a growing list of tokens, a piece at a time. Each piece is
    decoded with a few tokens before it, so a tokenizer that drops a word's
    leading space at the start of a text keeps it, and a token that ends
    inside a character waits for the rest of it."""

    def __init__(self, tokenizer: Tokenizer, prompt: Sequence[int]) -> None:
        self.tokenizer = tokenizer
        self.tokens = list(prompt[-4:])  # context only
        self.prefix = 0
        self.read = len(self.tokens)

    def add(self, token: int) -> str:
        self.tokens.append(token)
        decode = self.tokenizer.decode
        before = decode(self.tokens[self.prefix : self.read], skip_special_tokens=True)
        after = decode(self.tokens[self.prefix :], skip_special_tokens=True)
        if len(after) <= len(before) or after.endswith("�"):
            return ""
        self.prefix, self.read = self.read, len(self.tokens)
        return after[len(before) :]


@dataclass
class _Logprob:
    """A produced token's log-probability, and the most likely tokens at its
    position with theirs."""

    token: int
    logprob: float
    top: list[tuple[int, float]]


@dataclass
class _Choice:
    """One choice's output so far: `written` is its text, `sent` how much of
    it has been yielded, `shown` how many of its log-probabilities."""

    pending: _Pending
    text: _Text
    written: str = ""
    sent: int = 0
    reason: str = ""
    tokens: int = 0
    logprobs: list[_Logprob] = field(default_factory=list["_Logprob"])
    shown: int = 0


class _Generation:
    """A request's choices as they arrive: `pieces` yields text as it firms
    up, holding back what could be the start of a stop string, with the
    log-probabilities of the tokens behind it; each choice's `reason` and
    `tokens` are set when it ends."""

    def __init__(
        self,
        server: Server,
        pending: list[_Pending],
        texts: list[_Text],
        stops: list[str],
        events: queue.Queue[_Event],
    ) -> None:
        self.server = server
        self.choices = [_Choice(p, t) for p, t in zip(pending, texts, strict=True)]
        self.stops = stops
        self.events = events
        self.hold = max((len(s) for s in stops), default=1) - 1

    @property
    def tokens(self) -> int:
        return sum(choice.tokens for choice in self.choices)

    def pieces(self) -> Generator[tuple[int, str, list[_Logprob], str]]:
        """`(choice, text, log-probabilities, reason)`, the reason set on the
        last of each choice."""
        try:
            while not all(choice.reason for choice in self.choices):
                index, token, reason, logprob = self.events.get()
                if isinstance(token, str):
                    raise RuntimeError("the engine has failed:\n" + reason)
                choice = self.choices[index]
                if choice.reason:
                    continue  # read after a stop string ended it
                choice.tokens += 1
                if reason != "eos":
                    choice.written += choice.text.add(token)
                    if logprob is not None:
                        choice.logprobs.append(_Logprob(token, *logprob))
                cut = _first_stop(choice.written, self.stops, max(0, choice.sent - self.hold))
                if cut is not None:
                    choice.written = choice.written[:cut]
                    choice.reason = "stop"
                    self.server.cancel(choice.pending)
                elif reason:
                    choice.reason = _FINISH.get(reason, "stop")
                end = len(choice.written) if choice.reason else len(choice.written) - self.hold
                if end > choice.sent or choice.reason:
                    piece = choice.written[choice.sent : max(end, choice.sent)]
                    shown = choice.logprobs[choice.shown :]
                    choice.sent = max(end, choice.sent)
                    choice.shown = len(choice.logprobs)
                    yield index, piece, shown, choice.reason
        except GeneratorExit:
            # The client left: its rows are freed.
            for choice in self.choices:
                if not choice.reason:
                    self.server.cancel(choice.pending)
            raise


def _first_stop(text: str, stops: list[str], start: int) -> int | None:
    found = [i for i in (text.find(s, start) for s in stops) if i >= 0]
    return min(found) if found else None


# ---- HTTP


def _handler(server: Server) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "linnet"

        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200 if server.failed is None else 500, {})
            elif self.path == "/v1/models":
                model = {"id": server.name, "object": "model", "created": 0, "owned_by": "linnet"}
                self._json(200, {"object": "list", "data": [model]})
            else:
                self._error(404, f"no route {self.path}")

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
                parsed: _Json = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(parsed, dict):
                    raise RequestError("the body is a JSON object")
                body = parsed
                if self.path == "/v1/completions":
                    self._complete(body)
                elif self.path == "/v1/chat/completions":
                    self._chat(body)
                else:
                    self._error(404, f"no route {self.path}")
            except (TypeError, ValueError) as error:  # a RequestError, or a field's type
                self._error(400, str(error))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                self._error(500, str(error))

        def _complete(self, body: Mapping[str, _Json]) -> None:
            prompt: _Json = body.get("prompt")
            if isinstance(prompt, list) and len(prompt) == 1:
                prompt = prompt[0]
            if isinstance(prompt, str):
                ids = server.tokenizer.encode(prompt, add_special_tokens=True)
                echo = prompt
            elif isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
                ids = [int(t) for t in cast("list[int]", prompt)]
                echo = server.tokenizer.decode(ids, skip_special_tokens=True)
            else:
                raise RequestError("`prompt` is a string or a list of token ids")
            generation = server.generate(ids, body, limit=16, chat=False)
            self._respond(
                body, generation, len(ids), chat=False, echo=echo if body.get("echo") else ""
            )

        def _chat(self, body: Mapping[str, _Json]) -> None:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise RequestError("`messages` is a list of messages")
            text = server.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = server.tokenizer.encode(str(text), add_special_tokens=False)
            if body.get("max_completion_tokens") is not None:
                body = {**body, "max_tokens": body["max_completion_tokens"]}
            generation = server.generate(ids, body, limit=server.engine.backend.max_seq, chat=True)
            self._respond(body, generation, len(ids), chat=True)

        def _respond(
            self,
            body: Mapping[str, _Json],
            generation: _Generation,
            prompt: int,
            chat: bool,
            echo: str = "",
        ) -> None:
            identity = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
            head: dict[str, _Json] = {
                "id": identity,
                "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()),
                "model": server.name,
            }
            count = len(generation.choices)
            wanted = generation.choices[0].pending.request.logprobs is not None
            logprobs = _Logprobs(server.tokenizer, count)
            if not body.get("stream"):
                texts = [echo] * count
                taken: list[list[_Logprob]] = [[] for _ in range(count)]
                for index, piece, shown, _ in generation.pieces():
                    texts[index] += piece
                    taken[index] += shown
                choices: list[dict[str, _Json]] = []
                for index, choice in enumerate(generation.choices):
                    entry: dict[str, _Json] = {"index": index, "finish_reason": choice.reason}
                    formatted = logprobs.format(index, taken[index], chat) if wanted else None
                    if chat:
                        entry["message"] = {"role": "assistant", "content": texts[index]}
                    else:
                        entry["text"] = texts[index]
                    entry["logprobs"] = formatted
                    choices.append(entry)
                usage = _usage(prompt, generation.tokens)
                self._json(200, {**head, "choices": choices, "usage": usage})
                return
            head["object"] = "chat.completion.chunk" if chat else "text_completion"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def chunk(
                index: int,
                piece: str | None,
                reason: str | None,
                first: bool = False,
                shown: list[_Logprob] | None = None,
            ) -> None:
                choice: dict[str, _Json] = {"index": index, "finish_reason": reason}
                formatted = logprobs.format(index, shown or [], chat) if wanted and shown else None
                if chat:
                    delta: dict[str, _Json] = {"role": "assistant"} if first else {}
                    if piece is not None:
                        delta["content"] = piece
                    choice["delta"] = delta
                    choice["logprobs"] = formatted
                else:
                    choice |= {"text": piece or "", "logprobs": formatted}
                self._event({**head, "choices": [choice]})

            pieces = generation.pieces()
            try:
                for index in range(count):
                    if chat:
                        chunk(index, "", None, first=True)
                    elif echo:
                        chunk(index, echo, None)
                for index, piece, shown, reason in pieces:
                    if piece or shown:
                        chunk(index, piece, None, shown=shown)
                    if reason:
                        chunk(index, None, reason)
                options = body.get("stream_options")
                if isinstance(options, dict) and options.get("include_usage"):
                    usage = _usage(prompt, generation.tokens)
                    self._event({**head, "choices": [], "usage": usage})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            finally:
                pieces.close()

        def _event(self, data: Mapping[str, _Json]) -> None:
            self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
            self.wfile.flush()

        def _json(self, status: int, data: Mapping[str, _Json]) -> None:
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: int, message: str) -> None:
            kind = "invalid_request_error" if status < 500 else "server_error"
            self._json(status, {"error": {"message": message, "type": kind}})

    return Handler


def _logprobs_wanted(body: Mapping[str, _Json], chat: bool) -> int | None:
    """How many alternatives a request wants with each token's
    log-probability, or None when it wants none: a chat's `logprobs` and
    `top_logprobs`, a completion's `logprobs`."""
    if chat:
        if not body.get("logprobs"):
            if body.get("top_logprobs"):
                raise RequestError("`top_logprobs` needs `logprobs` set to true")
            return None
        wanted = int(_given(body, "top_logprobs", 0))
    else:
        value = cast("bool | int | None", body.get("logprobs"))
        if value is None or value is False:
            return None
        wanted = 0 if value is True else int(value)
    if not 0 <= wanted <= 20:
        raise RequestError("the number of log-probability alternatives is between 0 and 20")
    return wanted


class _Logprobs:
    """Log-probabilities as OpenAI writes them: a chat's `content` list, or a
    completion's parallel lists with each token's offset in the choice's
    text, which continues across a stream's chunks."""

    def __init__(self, tokenizer: Tokenizer, choices: int) -> None:
        self.tokenizer = tokenizer
        self.offsets = [0] * choices

    def text(self, token: int) -> str:
        return self.tokenizer.decode([token], skip_special_tokens=False)

    def format(self, index: int, entries: list[_Logprob], chat: bool) -> dict[str, _Json]:
        if chat:
            content: list[dict[str, _Json]] = []
            for entry in entries:
                text = self.text(entry.token)
                alternatives = [
                    {"token": t, "logprob": v, "bytes": list(t.encode())}
                    for t, v in ((self.text(token), value) for token, value in entry.top)
                ]
                content.append(
                    {
                        "token": text,
                        "logprob": entry.logprob,
                        "bytes": list(text.encode()),
                        "top_logprobs": alternatives,
                    }
                )
            return {"content": content, "refusal": None}
        tokens: list[str] = []
        offsets: list[int] = []
        for entry in entries:
            text = self.text(entry.token)
            tokens.append(text)
            offsets.append(self.offsets[index])
            self.offsets[index] += len(text)
        return {
            "tokens": tokens,
            "token_logprobs": [entry.logprob for entry in entries],
            "top_logprobs": [
                {self.text(token): value for token, value in entry.top} for entry in entries
            ],
            "text_offset": offsets,
        }


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
