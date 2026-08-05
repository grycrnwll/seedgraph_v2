"""Usage logging (doc 13 §11) — no prompt/response bodies.

``external_full_text`` and ``source_access_class`` are captured per event so the
Phase-9 boundary audit can verify the content-access gate held. ``estimated_cost``
stays NULL until pricing enforcement is wired (decision 37).
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from ..ids import new_id


def hash_text(text: str) -> str:
    """Full SHA-256 hex of ``text`` (UTF-8). No bodies are ever stored in the DB —
    only these hashes (cross-cutting decision #5); never a truncated preview."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class UsageEvent:
    task_type: str
    provider: str | None = None
    model: str | None = None
    access_mode: str | None = None
    run_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost: float | None = None
    unit_price_json: str | None = None
    source_access_class: str | None = None
    external_full_text: bool = False
    # Track 1 (migration 0013) — policy-gated executor telemetry. No bodies: only
    # full sha256 hex hashes (prompt_hash/response_hash). retry_count is NOT NULL
    # in the schema (DEFAULT 0); the rest are nullable.
    status: str | None = None
    error_code: str | None = None
    latency_ms: int | None = None
    request_id: str | None = None
    prompt_hash: str | None = None
    response_hash: str | None = None
    retry_count: int = 0
    pricing_snapshot_date: str | None = None


def log_usage(conn: sqlite3.Connection, event: UsageEvent) -> str:
    """Insert a usage event into ``llm_usage_events``; return its id."""
    usage_event_id = new_id("ue")
    conn.execute(
        """
        INSERT INTO llm_usage_events (
            usage_event_id, run_id, task_type, provider, model, access_mode,
            input_tokens, output_tokens, estimated_cost, unit_price_json,
            source_access_class, external_full_text,
            status, error_code, latency_ms, request_id, prompt_hash,
            response_hash, retry_count, pricing_snapshot_date, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            usage_event_id,
            event.run_id,
            event.task_type,
            event.provider,
            event.model,
            event.access_mode,
            event.input_tokens,
            event.output_tokens,
            event.estimated_cost,
            event.unit_price_json,
            event.source_access_class,
            1 if event.external_full_text else 0,
            event.status,
            event.error_code,
            event.latency_ms,
            event.request_id,
            event.prompt_hash,
            event.response_hash,
            event.retry_count if event.retry_count is not None else 0,
            event.pricing_snapshot_date,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return usage_event_id
