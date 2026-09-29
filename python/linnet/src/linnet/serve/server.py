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
token), `seed`, `stop` (a string or up to four), and `stream`, with
`stream_options.include_usage`. A request asking for more than one choice
or for log probabilities is refused; other fields are ignored.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol, cast

from . import Completion, Engine, Request
from .sampling import Sampling


class Tokenizer(Protocol):
    """The part of a Transformers tokenizer the server uses."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...
    def apply_chat_template(self, conversation: Any, **options: Any) -> Any: ...


class RequestError(ValueError):
    """A request the server refuses, with the reason sent back."""


_FINISH = {"eos": "stop", "stop": "stop", "length": "length", "cache": "length"}


@dataclass
class _Pending:
    """A request between a handler and the engine thread: `events` gets
    `(token, reason)` for every token read, or `("", error)` if the engine
    failed."""

    request: Request
    events: queue.Queue[tuple[int | str, str]]
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
                    pending.events.put(("", self.failed))
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

    def generate(self, prompt: list[int], body: dict[str, Any], limit: int) -> _Generation:
        """Hands a request to the engine; its tokens and text come from the
        returned generation as they are produced."""
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
        if _given(body, "n", 1) != 1:
            raise RequestError("only one choice (`n` = 1) is supported")
        if body.get("logprobs") or body.get("top_logprobs") or body.get("echo"):
            raise RequestError("`logprobs` and `echo` are not supported")
        stop: Any = body.get("stop") or []
        stops: list[Any] = [stop] if isinstance(stop, str) else list(stop)
        if len(stops) > 4 or not all(isinstance(s, str) and s for s in stops):
            raise RequestError("`stop` is a string or up to four")
        seed = body.get("seed")
        events: queue.Queue[tuple[int | str, str]] = queue.Queue()

        def on_token(completion: Completion) -> None:
            events.put((completion.tokens[-1], completion.reason))

        request = Request(
            prompt=prompt,
            max_new_tokens=min(int(_given(body, "max_tokens", limit)), room),
            eos=self.eos,
            temperature=float(_given(body, "temperature", 1.0)),
            top_k=max(int(_given(body, "top_k", 0)), 0),
            top_p=float(_given(body, "top_p", 1.0)),
            seed=None if seed is None else int(seed),
            on_token=on_token,
        )
        if request.max_new_tokens < 1:
            raise RequestError("`max_tokens` must be at least 1")
        try:
            Sampling(request.temperature, request.top_k, request.top_p).check()
        except ValueError as error:
            raise RequestError(str(error)) from None
        pending = _Pending(request, events)
        self._inbox.put(("submit", pending))
        return _Generation(self, pending, _Text(self.tokenizer, prompt), stops)

    def cancel(self, pending: _Pending) -> None:
        self._inbox.put(("cancel", pending))


def _done(pending: _Pending) -> bool:
    return pending.completion is not None and bool(pending.completion.reason)


def _given(body: dict[str, Any], key: str, default: Any) -> Any:
    value = body.get(key)
    return default if value is None else value


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


class _Generation:
    """One request's output as it arrives: `pieces` yields text as it firms
    up, holding back what could be the start of a stop string; `reason`
    and `tokens` are set when it ends."""

    def __init__(self, server: Server, pending: _Pending, text: _Text, stops: list[str]) -> None:
        self.server = server
        self.pending = pending
        self.text = text
        self.stops = stops
        self.hold = max((len(s) for s in stops), default=1) - 1
        self.tokens = 0
        self.reason = ""

    def pieces(self) -> Generator[str]:
        written = ""  # the text so far, some of it perhaps not yet yielded
        sent = 0
        try:
            while not self.reason:
                token, reason = self.pending.events.get()
                if isinstance(token, str):
                    raise RuntimeError("the engine has failed:\n" + reason)
                self.tokens += 1
                if reason != "eos":
                    written += self.text.add(token)
                cut = _first_stop(written, self.stops, max(0, sent - self.hold))
                if cut is not None:
                    written = written[:cut]
                    self.reason = "stop"
                    self.server.cancel(self.pending)
                elif reason:
                    self.reason = _FINISH.get(reason, "stop")
                end = len(written) if self.reason else len(written) - self.hold
                if end > sent:
                    yield written[sent:end]
                    sent = end
        except GeneratorExit:
            # The client left: its row is freed.
            if not self.reason:
                self.server.cancel(self.pending)
            raise


def _first_stop(text: str, stops: list[str], start: int) -> int | None:
    found = [i for i in (text.find(s, start) for s in stops) if i >= 0]
    return min(found) if found else None


# ---- HTTP


def _handler(server: Server) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "linnet"

        def log_message(self, format: str, *args: Any) -> None:
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
                parsed: Any = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(parsed, dict):
                    raise RequestError("the body is a JSON object")
                body = cast("dict[str, Any]", parsed)
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

        def _complete(self, body: dict[str, Any]) -> None:
            prompt: Any = body.get("prompt")
            if isinstance(prompt, list) and len(cast("list[Any]", prompt)) == 1:
                prompt = cast("list[Any]", prompt)[0]
            if isinstance(prompt, str):
                ids = server.tokenizer.encode(prompt, add_special_tokens=True)
            elif isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):  # pyright: ignore[reportUnknownVariableType]
                ids = [int(t) for t in cast("list[int]", prompt)]
            else:
                raise RequestError("`prompt` is a string or a list of token ids")
            generation = server.generate(ids, body, limit=16)
            self._respond(body, generation, len(ids), chat=False)

        def _chat(self, body: dict[str, Any]) -> None:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise RequestError("`messages` is a list of messages")
            text = server.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = server.tokenizer.encode(str(text), add_special_tokens=False)
            if body.get("max_completion_tokens") is not None:
                body = {**body, "max_tokens": body["max_completion_tokens"]}
            generation = server.generate(ids, body, limit=server.engine.backend.max_seq)
            self._respond(body, generation, len(ids), chat=True)

        def _respond(
            self, body: dict[str, Any], generation: _Generation, prompt: int, chat: bool
        ) -> None:
            identity = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
            head: dict[str, Any] = {
                "id": identity,
                "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()),
                "model": server.name,
            }
            if not body.get("stream"):
                text = "".join(generation.pieces())
                usage = _usage(prompt, generation.tokens)
                choice: dict[str, Any] = {"index": 0, "finish_reason": generation.reason}
                if chat:
                    choice["message"] = {"role": "assistant", "content": text}
                else:
                    choice |= {"text": text, "logprobs": None}
                self._json(200, {**head, "choices": [choice], "usage": usage})
                return
            head["object"] = "chat.completion.chunk" if chat else "text_completion"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def chunk(piece: str | None, reason: str | None, first: bool = False) -> None:
                choice: dict[str, Any] = {"index": 0, "finish_reason": reason}
                if chat:
                    delta: dict[str, Any] = {"role": "assistant"} if first else {}
                    if piece is not None:
                        delta["content"] = piece
                    choice["delta"] = delta
                else:
                    choice |= {"text": piece or "", "logprobs": None}
                self._event({**head, "choices": [choice]})

            pieces = generation.pieces()
            try:
                if chat:
                    chunk("", None, first=True)
                for piece in pieces:
                    chunk(piece, None)
                chunk(None, generation.reason)
                options: Any = body.get("stream_options")
                if isinstance(options, dict) and cast("dict[str, Any]", options).get(
                    "include_usage"
                ):
                    usage = _usage(prompt, generation.tokens)
                    self._event({**head, "choices": [], "usage": usage})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            finally:
                pieces.close()

        def _event(self, data: dict[str, Any]) -> None:
            self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
            self.wfile.flush()

        def _json(self, status: int, data: dict[str, Any]) -> None:
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


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
