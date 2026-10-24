"""A fake Ollama daemon: a real HTTP server on 127.0.0.1 with scripted replies.

Each scripted ``/api/chat`` reply is a ``(status, body)`` pair (``body`` a dict sent as
JSON, or a str sent as-is), or a number: sleep that many seconds, then answer 200 (for
timeout tests). Every request is recorded as ``(method, path, json_body)``. No real
network: the server binds the loopback interface on a free port. Invented text only."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest


def chat_reply(content: str, *, model: str = "invented-model", prompt_tokens: int = 120,
               eval_tokens: int = 30, done_reason: str = "stop") -> tuple[int, dict]:
    """An ``/api/chat`` success body (``stream: false``)."""
    return 200, {"model": model, "message": {"role": "assistant", "content": content}, "done": True,
                 "done_reason": done_reason, "prompt_eval_count": prompt_tokens, "eval_count": eval_tokens}


class FakeOllama:
    def __init__(self, models: tuple[str, ...] = ("invented-model:latest",)) -> None:
        self.models = list(models)
        self.chat_replies: deque[Any] = deque()
        self.requests: list[tuple[str, str, Any]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep test output quiet
                pass

            def _send(self, status: int, body: Any) -> None:
                data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:  # the client timed out and hung up
                    pass

            def do_GET(self) -> None:
                fake.requests.append(("GET", self.path, None))
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m, "model": m} for m in fake.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length).decode("utf-8") if length else ""
                body = json.loads(raw) if raw else None
                fake.requests.append(("POST", self.path, body))
                if self.path != "/api/chat":
                    self._send(404, {"error": "not found"})
                    return
                if not fake.chat_replies:
                    self._send(500, {"error": "no scripted reply"})
                    return
                reply = fake.chat_replies.popleft()
                if isinstance(reply, (int, float)) and not isinstance(reply, bool):
                    time.sleep(reply)
                    reply = chat_reply("{}")
                self._send(*reply)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def script(self, *replies: Any) -> None:
        self.chat_replies.extend(replies)

    @property
    def chats(self) -> list[dict]:
        return [body for method, path, body in self.requests if method == "POST" and path == "/api/chat"]

    def start(self) -> FakeOllama:
        self.thread.start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def closed_port_url() -> str:
    """A loopback URL on which nothing listens (connection refused)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture
def fake_ollama():
    server = FakeOllama().start()
    yield server
    server.close()
