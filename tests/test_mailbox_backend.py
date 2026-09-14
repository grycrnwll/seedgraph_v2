"""Mailbox provider: a Claude Code subagent answers LLM requests by file.

Offline by construction — the "provider" is a directory, so these tests exercise
the real adapter end to end: a daemon thread plays the answering agent (globbing
``requests/*.json``, writing ``responses/<id>.txt``), plus the timeout, the
factory branch, and the routing hop through ``claude_code_subagent``.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from seedgraph.config.models import (
    ContentPolicy,
    LLMConfig,
    ProjectConfig,
    TaskRoute,
    default_profiles,
)
from seedgraph.llm.backend import default_backend
from seedgraph.llm.providers._http import ProviderError
from seedgraph.llm.providers.mailbox import MailboxBackend
from seedgraph.llm.routing import Route, resolve_route


def _answer(root, answered: list[str], text: str) -> None:
    """Stand-in for the answering agent (bounded so a failure never hangs pytest)."""
    requests_dir = root / "requests"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        for path in sorted(requests_dir.glob("*.json")) if requests_dir.exists() else []:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["system_prompt"] == "SYS"
            assert payload["user_prompt"] == "USER"
            request_id = payload["request_id"]
            answered.append(request_id)
            (root / "responses" / f"{request_id}.txt").write_text(text, encoding="utf-8")
            return
        time.sleep(0.01)


def test_round_trip_returns_agent_answer_and_files_the_request(tmp_path):
    answered: list[str] = []
    worker = threading.Thread(target=_answer, args=(tmp_path, answered, '{"ok": true}'), daemon=True)
    worker.start()

    backend = MailboxBackend(base_url=str(tmp_path), poll_interval=0.02, timeout_s=10.0)
    completion = backend.complete("SYS", "USER", model="claude-code-subagent")
    worker.join(timeout=5)

    assert completion.text == '{"ok": true}'
    assert completion.finish_reason == "stop"  # never "length" — no retry is triggered
    assert completion.input_tokens > 0 and completion.output_tokens > 0
    assert answered, "the fake agent never saw a request"
    request_id = answered[0]
    assert (tmp_path / "done" / f"{request_id}.json").exists()
    assert not (tmp_path / "requests" / f"{request_id}.json").exists()


def test_timeout_is_a_retryable_provider_error(tmp_path):
    backend = MailboxBackend(base_url=str(tmp_path), poll_interval=0.05, timeout_s=0.3)
    with pytest.raises(ProviderError) as excinfo:
        backend.complete("SYS", "USER")
    assert excinfo.value.code == "timeout"
    assert excinfo.value.retryable is True
    assert excinfo.value.provider == "mailbox"


def test_default_backend_resolves_the_mailbox_adapter(tmp_path):
    backend = default_backend(provider="mailbox", model="claude-code-subagent", base_url=str(tmp_path))
    assert isinstance(backend, MailboxBackend)
    assert backend.root == tmp_path


def test_resolve_route_selects_the_subagent_profile():
    cfg = ProjectConfig(
        slug="t",
        llm=LLMConfig(
            profiles=default_profiles(),
            routes={
                "note_extraction": TaskRoute(
                    task_type="note_extraction",
                    preferred_profile="claude_code_subagent",
                    requires_source_text=True,
                )
            },
        ),
        content_policy=ContentPolicy(),
    )
    route = resolve_route("note_extraction", "user_supplied_private", cfg)
    assert isinstance(route, Route)
    assert route.profile_id == "claude_code_subagent"
    assert route.provider == "mailbox"
    # Local by construction: private full text never trips the content gate.
    assert route.external_full_text is False
