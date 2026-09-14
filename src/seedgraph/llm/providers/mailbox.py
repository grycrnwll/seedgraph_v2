"""Mailbox provider adapter (a human / Claude Code subagent answers by file).

No API, no network: each completion is written as a request FILE and the backend
blocks until an answer file appears, so any agent with filesystem access (e.g. a
Claude Code subagent) can stand in for a hosted model.

Protocol for the answering agent — poll ``<dir>/requests/`` by globbing
``requests/*.json`` (never list the directory: an in-flight request is a
``.json.tmp`` you must not read)::

    {"request_id", "created_at", "model", "temperature", "max_tokens",
     "system_prompt", "user_prompt"}

Answer the ``system_prompt``/``user_prompt`` pair exactly as a model would and
write the WHOLE answer (nothing else — no preamble, no fences) as the entire
content of ``<dir>/responses/<id>.txt``. Write it in ONE write (or write
``<id>.txt.tmp`` and rename), because the reader treats a file as complete once
its size is unchanged across two consecutive polls. A JSON answer file
``<id>.json`` (``{"text": ..., optional "finish_reason": ...}``) is also
accepted. The request file is moved to ``<dir>/done/`` once consumed.

The mailbox directory is the profile's ``base_url``, else
``SEEDGRAPH_MAILBOX_DIR``, else ``~/.seedgraph/mailbox``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from ..backend import LLMCompletion
from ..tokens import estimate_tokens
from ._http import ProviderError


class MailboxBackend:
    """File-mailbox chat backend (blocks/polls for an agent-written answer)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        *,
        poll_interval: float = 2.0,
        timeout_s: float = 7200.0,
        **_ignored,
    ) -> None:
        self.model = model
        self.root = Path(
            base_url or os.environ.get("SEEDGRAPH_MAILBOX_DIR") or Path.home() / ".seedgraph" / "mailbox"
        )
        self.poll_interval = poll_interval
        self.timeout_s = timeout_s

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMCompletion:
        requests_dir = self.root / "requests"
        responses_dir = self.root / "responses"
        done_dir = self.root / "done"
        try:
            for directory in (requests_dir, responses_dir, done_dir):
                directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProviderError(
                "provider_unavailable",
                f"mailbox directory {self.root} is not usable: {exc}",
                retryable=True,
                provider="mailbox",
            ) from exc

        request_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        payload = {
            "request_id": request_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model or self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        request_path = requests_dir / f"{request_id}.json"
        tmp_path = requests_dir / f"{request_id}.json.tmp"
        try:
            # Atomic publish: an agent globbing requests/*.json never sees a partial file.
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp_path, request_path)
        except OSError as exc:
            raise ProviderError(
                "provider_unavailable",
                f"could not write mailbox request {request_path}: {exc}",
                retryable=True,
                provider="mailbox",
            ) from exc

        text, finish_reason = self._await_response(request_id, responses_dir)
        try:
            os.replace(request_path, done_dir / request_path.name)
        except OSError:
            pass  # best effort — the answer is what matters
        return LLMCompletion(
            text=text,
            input_tokens=estimate_tokens(system_prompt) + estimate_tokens(user_prompt),
            output_tokens=estimate_tokens(text),
            finish_reason=finish_reason,
        )

    def _await_response(self, request_id: str, responses_dir: Path) -> tuple[str, str]:
        """Poll for ``<id>.txt`` / ``<id>.json``; return ``(text, finish_reason)``.

        An answer counts as complete only once its size is unchanged across two
        consecutive polls — agents may write non-atomically."""
        deadline = time.monotonic() + self.timeout_s
        last_size: Optional[int] = None
        while True:
            for path in (responses_dir / f"{request_id}.txt", responses_dir / f"{request_id}.json"):
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size == last_size:
                    return self._read_response(path)
                last_size = size
                break
            if time.monotonic() >= deadline:
                raise ProviderError(
                    "timeout",
                    f"no mailbox response for {request_id} after {self.timeout_s:g}s "
                    f"(expected {responses_dir / (request_id + '.txt')})",
                    retryable=True,
                    provider="mailbox",
                )
            time.sleep(self.poll_interval)

    def _read_response(self, path: Path) -> tuple[str, str]:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProviderError(
                "invalid_response",
                f"could not read mailbox response {path}: {exc}",
                retryable=False,
                provider="mailbox",
            ) from exc
        finish_reason = "stop"
        if path.suffix == ".json":
            try:
                data = json.loads(raw)
            except ValueError as exc:
                raise ProviderError(
                    "invalid_response",
                    f"mailbox response {path.name} is not valid JSON",
                    retryable=False,
                    provider="mailbox",
                ) from exc
            if not isinstance(data, dict):
                raise ProviderError(
                    "invalid_response",
                    f"mailbox response {path.name} is not a JSON object",
                    retryable=False,
                    provider="mailbox",
                )
            value = data.get("text")
            raw = value if isinstance(value, str) else ""
            reason = data.get("finish_reason")
            if isinstance(reason, str) and reason:
                finish_reason = reason
        if not raw.strip():
            raise ProviderError(
                "invalid_response",
                f"mailbox response {path.name} is empty",
                retryable=False,
                provider="mailbox",
            )
        return raw, finish_reason
