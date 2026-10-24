"""The Ollama HTTP client (design §6): the one model client v3 keeps.

Reduced from v2 ``llm/providers/ollama.py`` plus ``_http.py`` (copied, not imported:
this module must not depend on the hosted layer, which P8 deletes).

- :func:`resolve_base_url` - an explicit URL, else ``SEEDGRAPH_OLLAMA_URL``, else
  ``OLLAMA_HOST`` (normalised: a bare ``host:port`` gets ``http://``, a ``0.0.0.0``
  bind address becomes ``127.0.0.1``), else :data:`DEFAULT_BASE_URL`.
- :class:`OllamaClient` - ``complete_json(model, system, user, schema)`` posts a
  non-streaming ``/api/chat`` with ``format`` set to the JSON schema (or ``"json"``)
  and returns an :class:`OllamaReply`; ``list_models()`` reads ``/api/tags``;
  ``check()`` is the reachability probe for ``doctor``.
- :func:`complete_json` - the design's module-level form, returning the text.

Every transport or HTTP failure raises :class:`OllamaError` with a ``code``:
``unreachable`` (connection refused, DNS), ``timeout``, ``model_missing`` (HTTP 404
on chat), ``http_error`` (any other non-2xx) or ``invalid_response`` (non-JSON body,
no ``message``). An empty answer is NOT an error here: the reply's ``text`` is empty
and the caller decides (the worker treats it as invalid output).

``httpx`` is imported inside the request helper, so importing this module needs no
network dependency.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "http://localhost:11434"
ENV_URL = "SEEDGRAPH_OLLAMA_URL"
ENV_MODEL = "SEEDGRAPH_OLLAMA_MODEL"
OLLAMA_HOST = "OLLAMA_HOST"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_PROBE_TIMEOUT_S = 3.0
DEFAULT_MAX_TOKENS = 1024


class OllamaError(Exception):
    """A typed Ollama failure; ``code`` is one of the codes in the module docstring."""

    def __init__(self, code: str, message: str, *, status_code: int | None = None) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class OllamaReply:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    done_reason: str | None = None

    @property
    def usage(self) -> dict[str, int]:
        out: dict[str, int] = {}
        if self.input_tokens is not None:
            out["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            out["output_tokens"] = self.output_tokens
        return out


@dataclass(frozen=True)
class Reachability:
    """``doctor``'s view: ``ok`` when ``/api/tags`` answered; ``models`` are the names."""

    ok: bool
    base_url: str
    models: list[str] = field(default_factory=list)
    error: str | None = None


def _normalise_host(value: str) -> str:
    value = value.strip()
    if "://" not in value:
        value = f"http://{value}"
    parts = urlsplit(value)
    host = parts.hostname or "localhost"
    if host in ("0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    port = parts.port or 11434
    return f"{parts.scheme}://{host}:{port}"


def resolve_base_url(base_url: str | None = None) -> str:
    """The Ollama base URL (no trailing slash); see the module docstring for the order."""
    if base_url:
        return base_url.rstrip("/")
    env = os.environ.get(ENV_URL)
    if env and env.strip():
        return env.strip().rstrip("/")
    host = os.environ.get(OLLAMA_HOST)
    if host and host.strip():
        return _normalise_host(host)
    return DEFAULT_BASE_URL


def default_model() -> str | None:
    """``SEEDGRAPH_OLLAMA_MODEL`` when set (the CLI's default for ``--model``)."""
    value = os.environ.get(ENV_MODEL)
    return value.strip() or None if value else None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class OllamaClient:
    """A small client for one Ollama daemon. ``transport`` (an httpx transport) is a
    test seam only; the default talks HTTP to ``base_url``."""

    def __init__(self, base_url: str | None = None, *, timeout: float = DEFAULT_TIMEOUT_S,
                 transport: Any = None) -> None:
        self.base_url = resolve_base_url(base_url)
        self.timeout = timeout
        self._transport = transport

    def _request(self, method: str, path: str, *, json_body: Mapping[str, Any] | None = None,
                 timeout: float | None = None) -> dict[str, Any]:
        import httpx

        kwargs: dict[str, Any] = {"timeout": timeout if timeout is not None else self.timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(**kwargs) as client:
                response = client.request(method, url, json=dict(json_body) if json_body is not None else None)
        except httpx.TimeoutException as exc:
            raise OllamaError("timeout", f"no answer from {url} within {kwargs['timeout']}s") from exc
        except httpx.HTTPError as exc:
            raise OllamaError("unreachable", f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            detail = ""
            try:
                body = response.json()
                if isinstance(body, dict) and isinstance(body.get("error"), str):
                    detail = f": {body['error']}"
            except ValueError:
                pass
            code = "model_missing" if response.status_code == 404 and path == "/api/chat" else "http_error"
            raise OllamaError(code, f"HTTP {response.status_code} from {url}{detail}",
                              status_code=response.status_code)
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError("invalid_response", f"{url} returned a non-JSON body") from exc
        if not isinstance(data, dict):
            raise OllamaError("invalid_response", f"{url} returned a non-object JSON body")
        return data

    def complete_json(self, model: str, system: str, user: str, schema: Mapping[str, Any] | None = None, *,
                      temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS) -> OllamaReply:
        """One non-streaming chat turn constrained to JSON (``schema`` when given)."""
        if not model:
            raise OllamaError("model_missing", "no Ollama model given")
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": dict(schema) if schema is not None else "json",
            # Thinking models (e.g. qwen3) otherwise spend the budget on hidden reasoning
            # and answer with empty content; non-thinking models ignore the flag.
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = self._request("POST", "/api/chat", json_body=body)
        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaError("invalid_response", "the Ollama reply has no message")
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if not text.strip():
            # An older daemon that ignored ``think: false`` may put the answer here.
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                text = thinking
        done = data.get("done_reason")
        return OllamaReply(text=text, model=str(data.get("model") or model),
                           input_tokens=_int_or_none(data.get("prompt_eval_count")),
                           output_tokens=_int_or_none(data.get("eval_count")),
                           done_reason=done if isinstance(done, str) else None)

    def list_models(self, *, timeout: float | None = None) -> list[str]:
        """The model names ``/api/tags`` lists."""
        data = self._request("GET", "/api/tags", timeout=timeout)
        models = data.get("models")
        if not isinstance(models, list):
            raise OllamaError("invalid_response", "/api/tags has no models list")
        return [str(m.get("name") or m.get("model")) for m in models
                if isinstance(m, dict) and (m.get("name") or m.get("model"))]

    def check(self, *, timeout: float = DEFAULT_PROBE_TIMEOUT_S) -> Reachability:
        """Reachability for ``doctor``: never raises."""
        try:
            return Reachability(True, self.base_url, self.list_models(timeout=timeout))
        except OllamaError as exc:
            return Reachability(False, self.base_url, [], str(exc))


def complete_json(model: str, system: str, user: str, schema: Mapping[str, Any] | None = None, *,
                  base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT_S) -> str:
    """The design's ``complete_json(model, system, user, schema) -> str``."""
    return OllamaClient(base_url, timeout=timeout).complete_json(model, system, user, schema).text


def check_reachable(base_url: str | None = None, *, timeout: float = DEFAULT_PROBE_TIMEOUT_S) -> Reachability:
    """``doctor``'s Ollama check (P8 wires it in)."""
    return OllamaClient(base_url).check(timeout=timeout)
