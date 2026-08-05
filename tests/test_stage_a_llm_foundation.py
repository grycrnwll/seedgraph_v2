"""Stage A (LLM-backend foundation) — additive, offline/keyless.

Covers migration 0013 column-presence + no-body audit, hash_text, the
consolidated cost helpers (estimate_cost / preflight_estimate / monthly_spend),
and the new config fields. Provider dispatch / executor land in later stages.
"""

from __future__ import annotations

from seedgraph.config.models import (
    LLMProfile,
    TaskRoute,
    default_profiles,
    default_routes,
)
from seedgraph.db.bootstrap import ensure_project_db
from seedgraph.db.connection import open_project_db
from seedgraph.db.migrations import latest_version
from seedgraph.llm.cost import estimate_cost, monthly_spend, preflight_estimate
from seedgraph.llm.usage import UsageEvent, hash_text, log_usage


class _Cap:
    input_usd_per_mtok = 5.0
    output_usd_per_mtok = 15.0


# --- migration 0013 ---------------------------------------------------------

_NEW_COLUMNS = {
    "status",
    "error_code",
    "latency_ms",
    "request_id",
    "prompt_hash",
    "response_hash",
    "retry_count",
    "pricing_snapshot_date",
}


def test_latest_project_migration_is_pinned():
    # Guard pin: bump deliberately when a new numbered migration lands.
    # 0014 = Build A (reference_entries manual override); 0015 = Build B
    # (analysis/ranking: concepts.weight + project_graph_edges.shared_count);
    # 0016 = Build C (audit_records.payload for the canon decision log);
    # 0017 = Build D (works.abstract + works.oa_status — last of the program's
    # four project migrations).
    assert latest_version("project") == 17


def test_0013_adds_executor_columns_and_no_body_columns():
    ensure_project_db("proj")
    conn = open_project_db("proj")
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_usage_events)")}
        assert _NEW_COLUMNS <= cols
        # No body/preview columns are ever added (cross-cutting #5).
        forbidden = {
            "prompt",
            "response",
            "prompt_text",
            "response_text",
            "prompt_body",
            "response_body",
            "prompt_preview",
            "response_preview",
            "body",
        }
        assert forbidden.isdisjoint(cols)
    finally:
        conn.close()


def test_log_usage_persists_hashes_not_bodies():
    ensure_project_db("proj")
    conn = open_project_db("proj")
    try:
        ph = hash_text("the secret prompt body")
        rh = hash_text("the model response body")
        log_usage(
            conn,
            UsageEvent(
                task_type="note_extraction",
                provider="anthropic",
                model="claude-sonnet-4-6",
                status="success",
                prompt_hash=ph,
                response_hash=rh,
                retry_count=2,
                input_tokens=10,
                output_tokens=20,
                estimated_cost=0.0,
            ),
        )
        row = conn.execute(
            "SELECT status, prompt_hash, response_hash, retry_count FROM llm_usage_events"
        ).fetchone()
        assert tuple(row) == ("success", ph, rh, 2)
        assert len(ph) == 64 and len(rh) == 64  # full sha256 hex, never truncated
    finally:
        conn.close()


def test_retry_count_defaults_to_zero():
    ensure_project_db("proj")
    conn = open_project_db("proj")
    try:
        log_usage(conn, UsageEvent(task_type="note_extraction"))
        row = conn.execute("SELECT retry_count FROM llm_usage_events").fetchone()
        assert row[0] == 0  # NOT NULL DEFAULT 0
    finally:
        conn.close()


# --- cost helpers -----------------------------------------------------------

def test_estimate_cost_and_none_cap():
    assert estimate_cost(None, 1000, 1000) == 0.0
    # 1M in @ $5 + 1M out @ $15 = $20
    assert estimate_cost(_Cap(), 1_000_000, 1_000_000) == 20.0


def test_preflight_estimate_assumes_symmetric_output():
    # output unknown -> mirror the input (dry-run convention)
    assert preflight_estimate(_Cap(), 1_000_000) == estimate_cost(_Cap(), 1_000_000, 1_000_000)


def test_monthly_spend_sums_calendar_month():
    ensure_project_db("proj")
    conn = open_project_db("proj")
    try:
        # Two rows this month (created_at stamped now), one forced into another month.
        log_usage(conn, UsageEvent(task_type="t", estimated_cost=1.5))
        log_usage(conn, UsageEvent(task_type="t", estimated_cost=2.5))
        conn.execute(
            "UPDATE llm_usage_events SET created_at='2000-01-15T00:00:00+00:00', "
            "estimated_cost=99.0 WHERE rowid=(SELECT MIN(rowid) FROM llm_usage_events)"
        )
        conn.commit()
        ym = conn.execute(
            "SELECT substr(created_at,1,7) FROM llm_usage_events "
            "WHERE created_at != '2000-01-15T00:00:00+00:00' LIMIT 1"
        ).fetchone()[0]
        assert monthly_spend(conn, ym) == 2.5  # only the remaining current-month row
        assert monthly_spend(conn, "2000-01") == 99.0
    finally:
        conn.close()


# --- new config fields ------------------------------------------------------

def test_new_profile_and_route_fields_have_safe_defaults():
    p = LLMProfile(profile_id="x", provider="ollama")
    assert p.base_url is None
    assert p.key_source == "auto"  # ADR-0002: keyring → env chain by default
    assert p.key_name is None

    r = TaskRoute(task_type="t", preferred_profile="x")
    assert r.allowed_access_classes is None
    assert r.allow_external_fragments is False
    assert r.requires_structured_output is False
    assert r.max_input_tokens is None
    assert r.max_output_tokens is None
    assert r.fallback_behavior == "degrade"


def test_openai_profile_in_defaults_but_never_routed_by_default():
    # ADR-0003: the OpenAI adapter ships, so a default profile exists — but
    # ADR-0006 keeps every default route local-first, so nothing references it
    # until a user opts in via ``llm route set``.
    profs = default_profiles()
    openai = profs["openai_api_default"]
    assert openai.provider == "openai"
    assert openai.model == "gpt-5.4-mini"
    assert openai.env_var == "OPENAI_API_KEY"
    routes = default_routes()
    for route in routes.values():
        assert route.preferred_profile != "openai_api_default"
        assert route.fallback_profile != "openai_api_default"
    assert routes["answer_generation"].preferred_profile in profs
    assert routes["answer_generation"].fallback_profile in profs
