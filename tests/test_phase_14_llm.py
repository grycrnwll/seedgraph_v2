"""Phase 14 (Track 1 Build Stage B) — policy-gated executor + fallback + taxonomy.

Fully offline/keyless: providers are driven by ``httpx.MockTransport`` and the
deterministic ``FakeLLMBackend``; ``executor._sleep`` is patched to a no-op so the
transport-retry test is instant. Covers the executor, the runtime fallback chain,
the policy/budget/no-llm blocks, the status taxonomy, the three no-crash bug
regressions, the no-body audit, and the registry-aware availability check.
"""

from __future__ import annotations

import sqlite3

import httpx
import pytest

from seedgraph.config.models import (
    GlobalConfig,
    LLMConfig,
    LLMProfile,
    TaskRoute,
    default_profiles,
)
from seedgraph.config.loader import load_llm_capabilities
from seedgraph.db.bootstrap import ensure_project_db
from seedgraph.db.connection import open_project_db
from seedgraph.llm import executor
from seedgraph.llm.backend import SUPPORTED_PROVIDERS, FakeLLMBackend, StubRealBackend
from seedgraph.llm.executor import LLMError, LLMResult, dispatch, run_llm, status_for_code
from seedgraph.llm.profiles import is_profile_available
from seedgraph.llm.providers import ProviderError
from seedgraph.llm.providers.anthropic import AnthropicBackend


@pytest.fixture(autouse=True)
def _instant_retry(monkeypatch):
    monkeypatch.setattr(executor, "_sleep", lambda _s: None)


def _conn():
    ensure_project_db("proj")
    return open_project_db("proj")


# --- backends used as test doubles ------------------------------------------

class _RaisingBackend:
    """A backend whose ``complete`` raises a fixed :class:`ProviderError`."""

    def __init__(self, code="provider_unavailable", retryable=True, calls=None):
        self.code = code
        self.retryable = retryable
        self.calls = calls if calls is not None else []

    def complete(self, system_prompt, user_prompt, *, model=None, temperature=0.0, max_tokens=4096):
        self.calls.append((system_prompt, user_prompt))
        raise ProviderError(self.code, "boom", retryable=self.retryable, provider="ollama")


# --- dispatch ---------------------------------------------------------------

def test_dispatch_success_with_parse():
    backend = FakeLLMBackend(response={"ok": True})

    def parse(text):
        import json

        try:
            return json.loads(text), None
        except Exception:
            return None, "bad"

    result = dispatch(backend, "sys", "user", model="m", provider="ollama", parse=parse)
    assert result.ok and result.status == "success"
    assert result.parsed == {"ok": True}
    assert result.prompt_hash and len(result.prompt_hash) == 64
    assert result.response_hash and len(result.response_hash) == 64


def test_dispatch_provider_error_degrades_not_raises():
    backend = _RaisingBackend(code="provider_unavailable")
    result = dispatch(backend, "sys", "user", model="m", provider="ollama")
    assert result.status == "extraction_failed"
    assert result.error is not None and result.error.code == "provider_unavailable"


def test_dispatch_stub_real_backend_is_skipped_no_llm():
    result = dispatch(StubRealBackend(provider="none"), "s", "u", model=None)
    assert result.status == "skipped_no_llm"
    assert result.error.code == "config_error"


# --- truncation terser retry (Build C chunk 2, designs D2/D3) -----------------

def _dict_parse(text):
    import json

    try:
        parsed = json.loads(text)
    except Exception:
        return None, "invalid json"
    return (parsed, None) if isinstance(parsed, dict) else (None, "not a dict")


def _repair(prev, errors):
    return f"repair:{prev}"


def test_fake_backend_finish_reasons_queue():
    backend = FakeLLMBackend(responses=["a", "b"], finish_reasons=["length"])
    assert backend.complete("s", "u").finish_reason == "length"
    assert backend.complete("s", "u").finish_reason is None  # exhausted -> fail-open


def test_dispatch_truncated_then_complete_appends_terser_suffix():
    """First completion cut at max_tokens -> the repair user message carries the
    terser suffix (repair_prompt signature untouched); second completes -> success."""
    backend = FakeLLMBackend(
        responses=['{"ok": tru', '{"ok": true}'],
        finish_reasons=["length", "stop"],
    )
    result = dispatch(backend, "sys", "user", model="m", provider="ollama",
                      parse=_dict_parse, repair_prompt=_repair)
    assert result.ok and result.parsed == {"ok": True}
    assert len(backend.calls) == 2
    second_user = backend.calls[1]["user"]
    assert second_user.startswith('repair:{"ok": tru')  # callback output preserved
    assert second_user.endswith(executor._TERSER_REPAIR_SUFFIX)
    assert "much more tersely" in second_user


def test_dispatch_persistent_truncation_names_truncation_in_terminal_message():
    backend = FakeLLMBackend(
        responses=['{"ok": tru', '{"still": tru'],
        finish_reasons=["length", "length"],
    )
    result = dispatch(backend, "sys", "user", model="m", provider="ollama",
                      parse=_dict_parse, repair_prompt=_repair)
    assert result.status == "extraction_failed"
    assert result.error.code == "schema_validation_failed"  # no new status code
    assert "output truncated at max_tokens after terser retry" in result.error.message
    assert result.completion.finish_reason == "length"


def test_dispatch_no_finish_reason_repair_path_unchanged():
    """Fail-open (D2): with no finish signal the repair message and the terminal
    error message are byte-identical to the pre-finish_reason behavior."""
    backend = FakeLLMBackend(responses=['{"bad": tru', '{"bad": tru'])
    result = dispatch(backend, "sys", "user", model="m", provider="ollama",
                      parse=_dict_parse, repair_prompt=_repair)
    assert result.status == "extraction_failed"
    assert result.error.message == "model output failed validation after repair"
    assert backend.calls[1]["user"] == 'repair:{"bad": tru'  # no suffix appended
    assert result.completion.finish_reason is None


def test_dispatch_truncated_but_parseable_does_not_retry():
    """The terser retry keys off parse failure AND length -- a length-flagged
    completion that still parses is a success with exactly one call."""
    backend = FakeLLMBackend(responses=['{"ok": true}'], finish_reasons=["length"])
    result = dispatch(backend, "sys", "user", model="m", provider="ollama",
                      parse=_dict_parse, repair_prompt=_repair)
    assert result.ok and result.parsed == {"ok": True}
    assert len(backend.calls) == 1


# --- runtime fallback chain -------------------------------------------------

def test_runtime_fallback_chain_one_hop(monkeypatch):
    """Primary provider down → next_hop's backend serves the call (capped one hop)."""
    primary = _RaisingBackend(code="provider_unavailable")
    secondary = FakeLLMBackend(response={"answer": 42})
    hops = {"n": 0}

    def next_hop(error: LLMError):
        hops["n"] += 1
        return executor.Hop(
            backend=secondary, model="claude-sonnet-4-6", provider="anthropic",
            access_mode="api_key", profile_id="anthropic_api_default", external_full_text=True,
        )

    def parse(text):
        import json

        return json.loads(text), None

    result = dispatch(primary, "sys", "user", model="llama3", provider="ollama",
                      parse=parse, next_hop=next_hop)
    assert hops["n"] == 1  # exactly one hop attempted
    assert result.ok and result.parsed == {"answer": 42}
    assert result.provider == "anthropic"  # the hop that actually produced the result
    assert result.external_full_text is True


def test_runtime_fallback_capped_at_one_hop():
    """If the hop also fails, the chain stops (no second hop) and degrades."""
    primary = _RaisingBackend(code="provider_unavailable")
    secondary = _RaisingBackend(code="provider_unavailable")

    def next_hop(error):
        return executor.Hop(backend=secondary, model="m", provider="anthropic")

    result = dispatch(primary, "s", "u", model="m", provider="ollama", next_hop=next_hop)
    assert result.status == "extraction_failed"
    assert result.provider == "anthropic"  # the last hop tried
    assert secondary.calls  # the hop ran once, then stopped


def test_run_llm_local_first_falls_back_to_anthropic(monkeypatch):
    """End-to-end: note_extraction prefers Ollama; when /api/chat ConnectErrors the
    chain re-routes to the hosted Anthropic profile (open_access → gate allows)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            raise httpx.ConnectError("ollama down", request=request)
        if request.url.path == "/v1/messages":
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "{}"}],
                      "usage": {"input_tokens": 4, "output_tokens": 2}},
            )
        return httpx.Response(404, json={})  # pragma: no cover

    conn = _conn()
    try:
        result = run_llm(
            "note_extraction", "sys", "user",
            access_class="open_access", config=GlobalConfig(),
            conn=conn, transport=httpx.MockTransport(handler),
        )
        assert result.ok
        assert result.provider == "anthropic"  # fell back off the down local provider
        row = conn.execute("SELECT status, provider FROM llm_usage_events").fetchone()
        assert row[0] == "success" and row[1] == "anthropic"
    finally:
        conn.close()


# --- policy / budget / no-llm blocks (still log a usage row) ----------------

def _external_only_config():
    profs = {
        "anthropic_api_default": default_profiles()["anthropic_api_default"],
        "no_llm": default_profiles()["no_llm"],
    }
    routes = {
        "note_extraction": TaskRoute(
            task_type="note_extraction", preferred_profile="anthropic_api_default",
            fallback_profile="no_llm", requires_source_text=True,
        )
    }
    return GlobalConfig(llm=LLMConfig(profiles=profs, routes=routes))


def test_policy_block_returns_skipped_policy_and_logs_usage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    conn = _conn()
    try:
        result = run_llm(
            "note_extraction", "sys", "user",
            access_class="user_supplied_private", config=_external_only_config(), conn=conn,
        )
        assert result.status == "skipped_policy"
        assert result.error.code == "policy_blocked"
        row = conn.execute("SELECT status, error_code FROM llm_usage_events").fetchone()
        assert tuple(row) == ("skipped_policy", "policy_blocked")
    finally:
        conn.close()


def test_budget_block_returns_skipped_budget(monkeypatch):
    cfg = _external_only_config()
    cfg.llm.routes["note_extraction"].preferred_profile = "local_ollama_default"
    cfg.llm.profiles["local_ollama_default"] = default_profiles()["local_ollama_default"]
    cfg.budget.usd_limit = 5.0  # arms the fail-closed unverified-pricing gate
    conn = _conn()
    try:
        result = run_llm(
            "note_extraction", "sys", "user",
            access_class="open_access", config=cfg,
            capabilities=load_llm_capabilities(), conn=conn,
        )
        assert result.status == "skipped_budget"
        assert result.error.code == "budget_exceeded"
        row = conn.execute("SELECT status FROM llm_usage_events").fetchone()
        assert row[0] == "skipped_budget"
    finally:
        conn.close()


def test_no_llm_route_returns_skipped_no_llm():
    cfg = GlobalConfig()
    cfg.llm.routes["note_extraction"].preferred_profile = "no_llm"
    cfg.llm.routes["note_extraction"].fallback_profile = None
    result = run_llm("note_extraction", "s", "u", access_class="open_access", config=cfg)
    assert result.status == "skipped_no_llm"
    assert result.error.code == "config_error"


# --- transport retry --------------------------------------------------------

def test_transport_retry_on_rate_limit_then_success():
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {}})
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "ok"}],
                  "usage": {"input_tokens": 1, "output_tokens": 1}},
        )

    backend = AnthropicBackend(model="m", api_key="k", transport=httpx.MockTransport(handler))
    result = dispatch(backend, "s", "u", model="m", provider="anthropic")
    assert result.ok
    assert state["n"] == 2  # retried once
    assert result.retry_count >= 1


def test_transport_does_not_retry_auth_error():
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(401, json={"error": {}})

    backend = AnthropicBackend(model="m", api_key="bad", transport=httpx.MockTransport(handler))
    result = dispatch(backend, "s", "u", model="m", provider="anthropic")
    assert result.status == "extraction_failed"
    assert result.error.code == "auth_error"
    assert state["n"] == 1  # auth errors are never retried


# --- status taxonomy --------------------------------------------------------

def test_status_taxonomy_code_to_status():
    assert status_for_code(None) == "success"
    assert status_for_code("policy_blocked") == "skipped_policy"
    assert status_for_code("config_error") == "skipped_no_llm"
    assert status_for_code("budget_exceeded") == "skipped_budget"
    for code in ("provider_unavailable", "rate_limited", "auth_error", "timeout",
                 "invalid_response", "schema_validation_failed"):
        assert status_for_code(code) == "extraction_failed"


def test_usage_status_maps_extraction_failed_to_error():
    conn = _conn()
    try:
        result = LLMResult(status="extraction_failed", provider="anthropic", model="m",
                           error=LLMError("provider_unavailable", "x"))
        executor.log_llm_usage(conn, task_type="note_extraction", result=result,
                               source_access_class="open_access")
        row = conn.execute("SELECT status, error_code FROM llm_usage_events").fetchone()
        assert tuple(row) == ("error", "provider_unavailable")
    finally:
        conn.close()


# --- no-crash bug regressions ----------------------------------------------

def _raising_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("transport boom")

    return httpx.MockTransport(handler)


def test_regression_ask_degrades_not_traceback(monkeypatch):
    """ask: a real key + a failing transport degrades to retrieval_only, not a crash."""
    from _phase8_helpers import build_fixture_project, make_answer_config, make_spec

    from seedgraph.answer import compose
    from seedgraph.answer.types import AnswerMode, RankedCandidate, RetrievedItem

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    project = build_fixture_project("ph14_ask")
    conn = sqlite3.connect(str(project.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        backend = AnthropicBackend(model="claude-sonnet-4-6", api_key="sk-test",
                                   transport=_raising_transport())
        item = RetrievedItem(
            item_id="s_a", kind="span", work_id="work_a", span_id="s_a", claim_id="c_a",
            text="Assumption 2 holds.", bm25_score=-3.0, access_class="open_access",
            epistemic_type="llm_extracted", assertion_status="stated",
        )
        cands = [RankedCandidate(item=item, rank_score=0.9, boosts={})]
        cfg = make_answer_config(preferred="anthropic_api_default")
        # chunk 1: generate returns (envelope, allowed, ComposeTrace) — this direct
        # caller sits outside the trace file-list but must track the 3-tuple return.
        env, _allowed, _ct = compose.generate(cands, make_spec("q"), cfg, project_conn=conn, backend=backend)
        assert env.mode == AnswerMode.RETRIEVAL_ONLY
        assert "model_unavailable" in env.warnings
        # one usage row, recording the failed call (no body, only status/hash).
        row = conn.execute("SELECT status, error_code FROM llm_usage_events").fetchone()
        assert row[0] == "error"
    finally:
        conn.close()


def test_regression_lens_error_text_not_note_extraction():
    """lens degradation no longer surfaces the old 'note extraction' message, and an
    executor dispatch over a failing backend degrades (the path run_lens routes to
    the deterministic fallback) instead of raising."""
    from seedgraph.errors import ConfigError

    with pytest.raises(ConfigError) as ei:
        StubRealBackend(provider="ollama").complete("s", "u")
    assert "note extraction" not in str(ei.value)
    assert "this task" in str(ei.value)

    # the mechanism run_lens uses: a failing backend yields a non-success result,
    # never an exception, so the runner falls through to deterministic_work.
    result = dispatch(_RaisingBackend(), "s", "u", model="m", provider="ollama")
    assert result.status != "success"


def test_regression_concepts_profile_selected_and_mode_deterministic(monkeypatch):
    """concepts --profile selects a REAL profile; when the external proposer applies
    no fold (offline degrade), the build is labeled deterministic — not mislabeled
    'llm' just because a profile was named."""
    from datetime import datetime, timezone

    from seedgraph.project import service
    from seedgraph.semantic import build_semantic_overlay

    def _now():
        return datetime.now(timezone.utc).isoformat()

    h = service.create_project("ph14_concepts")
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        # two token-overlapping found labels -> one candidate cluster -> the
        # proposer is actually invoked (and degrades to no fold offline).
        for cid, wid, label in (
            ("c1", "w1", "graph neural network"),
            ("c2", "w2", "graph neural networks"),
        ):
            conn.execute(
                "INSERT INTO works (work_id, canonical_title, created_at) VALUES (?,?,?)",
                (wid, wid, _now()),
            )
            run_w = "run_" + wid
            conn.execute(
                "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, "
                "markdown_hash, schema_version, prompt_version, access_class, run_status, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (run_w, wid, "md_" + wid, "h", "v1", "p1", "open_access", "success", _now()),
            )
            conn.execute(
                "INSERT INTO extracted_claims (claim_id, extraction_run_id, work_id, "
                "claim_type, field_key, normalized_label, claim_text, status, "
                "epistemic_type, access_class, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (cid, run_w, wid, "method", "f", label, label, "found",
                 "llm_extracted", "open_access", _now()),
            )
        conn.commit()

        cfg = GlobalConfig()
        profile = cfg.llm.profiles["local_ollama_default"]
        assert profile is not None  # a real profile is selected (review #3)
        # the injected backend returns junk -> proposer degrades to [] -> no fold.
        report = build_semantic_overlay(
            conn, run_id="r1", profile=profile, config=cfg,
            backend=FakeLLMBackend(response="not json at all"),
        )
        assert report.concept_mode == "deterministic"
    finally:
        conn.close()


# --- no-body audit ----------------------------------------------------------

def test_usage_persists_only_hashes_no_bodies():
    conn = _conn()
    try:
        result = dispatch(FakeLLMBackend(response="hello"), "secret system", "secret user",
                          model="m", provider="anthropic")
        executor.log_llm_usage(conn, task_type="answer_generation", result=result,
                               source_access_class="open_access")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(llm_usage_events)")}
        forbidden = {"prompt", "response", "prompt_text", "response_text", "prompt_body",
                     "response_body", "prompt_preview", "response_preview", "body"}
        assert forbidden.isdisjoint(cols)
        ph, rh = conn.execute(
            "SELECT prompt_hash, response_hash FROM llm_usage_events"
        ).fetchone()
        assert len(ph) == 64 and len(rh) == 64  # full sha256, never a body/preview
        # no body string is anywhere in the row.
        row_text = " ".join(
            str(v) for v in conn.execute("SELECT * FROM llm_usage_events").fetchone()
        )
        assert "secret" not in row_text
    finally:
        conn.close()


# --- registry-aware availability --------------------------------------------

def test_registry_rejects_unimplemented_provider_profile(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-mistral")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    # No mistral adapter ships: key present, still never "available"/selected.
    assert "mistral" not in SUPPORTED_PROVIDERS
    mistral_profile = LLMProfile(
        profile_id="mistral_api_default", provider="mistral", access_mode="api_key",
        model="mistral-x", env_var="MISTRAL_API_KEY",
    )
    assert is_profile_available(mistral_profile) is False
    # OpenAI shipped (ADR-0003): with a key it IS available.
    openai_profile = LLMProfile(
        profile_id="openai_api_default", provider="openai", access_mode="api_key",
        model="gpt-5.4-mini", env_var="OPENAI_API_KEY",
    )
    assert is_profile_available(openai_profile) is True
