"""Lens runner — one independent extraction pass per work (plan §5/§6.2/§7).

A lens is a *sibling extraction schema* (decision 22): the runner reuses the
phase_4 extraction plumbing and the phase_3/phase_5 span-anchoring path rather
than inventing storage. Per targeted work:

  1. resolve work->markdown via phase_5b ``resolve_work_markdown`` over the
     ``work_source_files`` bridge (UNIQUE(work_id), no `role` column, plain
     ``WHERE work_id = ?``). ``None`` -> skip + record in skipped_no_markdown[],
     never extraction_failed (plan §9 coverage-gap fix).
  2. idempotency guard (unless ``force``): skip works whose latest non-stale run
     already matches (definition_hash, markdown_hash) — no extraction_run, no
     LLM call (plan §7, must-fix 1).
  3. resolve the work's source `access_class` fail-closed via the bridge
     (work_id -> source_files.access_class, ATTACH cache.db read-only;
     missing/unknown -> user_supplied_private) (decision 60/76).
  4. route the `project_lens_extraction` task; validate records + enforce
     evidence_policy (schema.validate_record).
  5. anchor each verbatim quote to an `evidence_span` (reuse phase_3 ensure_span);
     project found records onto `extracted_claims` (object_type->claim_type rule,
     §4.4) + `claim_spans` (link_claim_spans, rank=0 primary) + `claim_fts`.
  6. write the `lens_outputs` row (found/not_found/ambiguous/extraction_failed,
     full verbatim fields_json) and the `extraction_run` stamped with
     lens_id + lens_definition_hash + markdown_id + markdown_hash.

D2 provenance on projected claims: origin `epistemic_type='llm_extracted'` (or
`deterministic` via the fallback path), author-assertion `assertion_status` from
the record. Schema authority is the 0009 migration (D6).

NOTE (schema_id): a lens run leaves ``extraction_runs.schema_id`` NULL (decision
22 / §4.3 — a lens is a *sibling* schema, mutually exclusive with the default
schema_id, which 0007 makes nullable); the lens identity is ``lens_id`` +
``lens_definition_hash`` and schema_id is never read for lens runs.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .. import cache_access
from ..acquisition.bridge import resolve_work_markdown
from ..db.adapter import raw_conn
from ..db.fts import ensure_fts_tables
from ..errors import ConfigError
from ..extraction.runner import resolve_source_access_class
from ..extraction.validator import extract_json_object
from ..ids import new_id
from ..llm.executor import dispatch as _llm_dispatch
from ..llm.routing import NoLlmRoute, resolve_route
from ..project.review import enqueue_in_session
from ..spans.store import ensure_span
from ..vocab import ReviewItemType, lens_claim_type
from . import registry
from .prompt import LENS_PROMPT_VERSION, build_lens_prompt

if TYPE_CHECKING:
    from sqlmodel import Session

    from ..config.models import LlmCapabilities, ProjectConfig
    from ..llm.backend import LLMBackend
    from .schema import LensDefinition

TASK_TYPE = "project_lens_extraction"
LENS_SCHEMA_VERSION = "lens_schema_v1"

# Offline test seam (mirrors extraction.runner._BACKEND_OVERRIDE): when set, the
# CLI path — which cannot thread a backend through Typer — dispatches through this
# backend instead of the lazy stub real client. Tests monkeypatch it; production
# leaves it None so the no-key path degrades to the deterministic fallback or the
# stub real backend's actionable error.
_BACKEND_OVERRIDE = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LensRouter:
    """Offline-injectable routing + backend bundle for the lens runner.

    The plan §6.2 names this ``LLMRouter``; concretely it bundles the project's
    routing :class:`~seedgraph.config.models.ProjectConfig` (consumed by
    ``resolve_route`` for the content-access gate) and the injected
    :class:`~seedgraph.llm.backend.LLMBackend` (a ``FakeLLMBackend`` in tests, so
    pytest never touches the network/a key). ``backend=None`` forces the no-LLM
    deterministic fallback path.
    """

    config: "ProjectConfig"
    backend: "LLMBackend | None" = None
    capabilities: "LlmCapabilities | None" = None


@dataclass
class LensRunResult:
    """Aggregate outcome of a `run_lens` pass (drives the manifest + CLI summary).

    Per-status counts plus the two honest-degradation work lists the manifest
    records separately (plan §7/§9): ``skipped_current`` (idempotency no-op) and
    ``skipped_no_markdown`` (no selected markdown). ``router_calls`` proves the
    idempotency acceptance check (a no-op re-run makes zero router calls).
    ``degraded`` / ``capability_note`` record the no-LLM reduced-capability gap
    (plan §8/§12) for the run manifest.
    """

    lens_id: str
    found: int = 0
    not_found: int = 0
    ambiguous: int = 0
    not_applicable: int = 0
    extraction_failed: int = 0
    extraction_run_ids: list[str] = field(default_factory=list)
    skipped_current: list[str] = field(default_factory=list)
    skipped_no_markdown: list[str] = field(default_factory=list)
    router_calls: int = 0
    degraded: bool = False
    capability_note: str | None = None


# --- idempotency / staleness ------------------------------------------------

def _latest_run(conn: sqlite3.Connection, lens_id: str, work_id: str):
    return conn.execute(
        "SELECT lens_definition_hash, markdown_hash FROM extraction_runs "
        "WHERE lens_id = ? AND work_id = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (lens_id, work_id),
    ).fetchone()


def _is_current(
    conn: sqlite3.Connection,
    lens_id: str,
    work_id: str,
    definition_hash: str,
    markdown_hash: str,
) -> bool:
    """A work is CURRENT when its LATEST lens run matches (definition_hash,
    markdown_hash) — skipped with NO extraction_run and NO router/backend call."""
    row = _latest_run(conn, lens_id, work_id)
    if row is None:
        return False
    return row[0] == definition_hash and row[1] == markdown_hash


# --- row writers ------------------------------------------------------------

_RUN_COLUMNS = (
    "extraction_run_id, work_id, markdown_id, markdown_hash, schema_id, lens_id, "
    "schema_version, prompt_version, model_name, provider, access_mode, temperature, "
    "access_class, external_full_text, run_status, input_tokens, output_tokens, "
    "estimated_cost, run_id, created_at, lens_definition_hash"
)


def _insert_lens_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    lens: "LensDefinition",
    work_id: str,
    markdown_id: str,
    markdown_hash: str,
    definition_hash: str,
    access_class: str,
    run_status: str,
    model_name: str | None = None,
    provider: str | None = None,
    access_mode: str | None = None,
    external_full_text: int = 0,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    build_run_id: str | None = None,
) -> None:
    """Insert one lens ``extraction_runs`` row (schema_id NULL; lens stamp).

    The omitted 4b columns (extraction_mode, ...) take their SQL DEFAULTs
    ('whole' / NULL); this phase never chunks.
    """
    conn.execute(
        f"INSERT INTO extraction_runs ({_RUN_COLUMNS}) "
        f"VALUES ({', '.join(['?'] * 21)})",
        (
            run_id,
            work_id,
            markdown_id,
            markdown_hash,
            None,  # schema_id NULL for lens runs (sibling schema; lens_id is the identity, decision 22 / §4.3)
            lens.lens_id,
            LENS_SCHEMA_VERSION,
            LENS_PROMPT_VERSION,
            model_name,
            provider,
            access_mode,
            0.0,
            access_class,
            external_full_text,
            run_status,
            input_tokens,
            output_tokens,
            None,  # estimated_cost (offline)
            build_run_id,
            _now(),
            definition_hash,
        ),
    )


def _insert_lens_output(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    lens_id: str,
    work_id: str,
    status: str,
    fields: dict,
    access_class: str,
    claim_id: str | None = None,
    confidence: float | None = None,
) -> str:
    """Insert one ``lens_outputs`` row; ``fields`` is round-tripped verbatim."""
    lens_output_id = new_id("lensout")
    conn.execute(
        "INSERT INTO lens_outputs (lens_output_id, extraction_run_id, lens_id, "
        "work_id, status, claim_id, confidence, fields_json, access_class, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            lens_output_id,
            run_id,
            lens_id,
            work_id,
            status,
            claim_id,
            confidence,
            json.dumps(fields, ensure_ascii=False),
            access_class,
            _now(),
        ),
    )
    return lens_output_id


def link_claim_spans(
    raw_conn: sqlite3.Connection,
    claim_id: str,
    span_ids: list[str],
) -> None:
    """Insert claim->span links into the canonical `claim_spans` junction
    (decision 70; the scalar `span_id` column was dropped). Primary span gets
    ``rank=0``; ``UNIQUE(claim_id, span_id)`` dedupes. Uses the raw DBAPI conn
    from ``db.adapter.raw_conn`` so it shares the ORM transaction (D7). This is
    the thin helper §4.4/§12 require — `claim_spans` is CONSUMED, never altered.
    """
    now = _now()
    for rank, span_id in enumerate(s for s in span_ids if s):
        raw_conn.execute(
            "INSERT OR IGNORE INTO claim_spans (claim_id, span_id, rank, created_at) "
            "VALUES (?, ?, ?, ?)",
            (claim_id, span_id, rank, now),
        )


# --- record field extraction (generic across lenses) ------------------------

def _record_quote(record: dict) -> str | None:
    """The verbatim quote a found record carries (plan §4.4 claim_text)."""
    for key in ("condition_text_verbatim", "exact_quote", "quote", "text_verbatim", "verbatim"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key, value in record.items():
        if isinstance(value, str) and value.strip() and key.endswith("verbatim"):
            return value
    return None


def _record_label(record: dict) -> str | None:
    """The normalized label a found record carries (plan §4.4 normalized_label)."""
    for key in ("normalized_condition_label", "normalized_label", "label"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


# --- found-record projection onto the claim spine ---------------------------

def _project_found_record(
    session: "Session",
    conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection,
    cache_root,
    lens: "LensDefinition",
    *,
    work_id: str,
    markdown_id: str,
    run_id: str,
    record: dict,
    field_index: int,
    access_class: str,
    result: LensRunResult,
) -> None:
    """Anchor + validate one found record, then project per its gated status."""
    quote = _record_quote(record)
    span_id = None
    if quote:
        span_id = ensure_span(
            conn,
            cache_conn,
            cache_root,
            markdown_id=markdown_id,
            work_id=work_id,
            exact_quote=quote,
            access_class=access_class,
            span_kind="manual",
        )
    record = dict(record)
    record["evidence_span_ids"] = [span_id] if span_id else []

    vr = lens.validate_record(record)
    now = _now()

    if vr.status == "found":
        claim_id = new_id("claim")
        claim_type = lens_claim_type(lens.object_type, lens.lens_id)
        label = _record_label(record)
        assertion_status = vr.assertion_status
        inferred_explanation = (
            (record.get("notes") or None) if assertion_status == "inferred" else None
        )
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, structured_note_id, "
            "extraction_run_id, work_id, claim_type, claim_subtype, field_key, "
            "normalized_label, claim_text, status, epistemic_type, assertion_status, "
            "inferred_explanation, confidence, access_class, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                claim_id,
                None,  # structured_note_id — lens claims mint no note header
                run_id,
                work_id,
                claim_type,
                None,
                f"{lens.object_type}[{field_index}]",
                label,
                quote,
                "found",
                "llm_extracted",
                assertion_status,
                inferred_explanation,
                vr.confidence,
                access_class,
                now,
            ),
        )
        link_claim_spans(conn, claim_id, [span_id] if span_id else [])
        conn.execute(
            "INSERT INTO claim_fts(claim_id, normalized_label, claim_text) VALUES (?, ?, ?)",
            (claim_id, label, quote),
        )
        _insert_lens_output(
            conn,
            run_id=run_id,
            lens_id=lens.lens_id,
            work_id=work_id,
            status="found",
            fields=record,
            access_class=access_class,
            claim_id=claim_id,
            confidence=vr.confidence,
        )
        result.found += 1
        return

    # ambiguous (span-less found) / extraction_failed (bad type/enum/inferred) ->
    # lens_outputs(claim_id NULL) + a lens_record review item (never silent drop).
    lo_id = _insert_lens_output(
        conn,
        run_id=run_id,
        lens_id=lens.lens_id,
        work_id=work_id,
        status=vr.status,
        fields=record,
        access_class=access_class,
        claim_id=None,
        confidence=vr.confidence,
    )
    reason = "; ".join(vr.issues) or (
        "unanchorable verbatim quote" if vr.status == "ambiguous" else "validation failed"
    )
    enqueue_in_session(
        session,
        ReviewItemType.lens_record.value,
        target_type="lens_output",
        target_id=lo_id,
        payload={
            "kind": "lens_record",
            "lens_id": lens.lens_id,
            "work_id": work_id,
            "status": vr.status,
            "reason": reason,
            "lens_output_id": lo_id,
            "raw": (quote or "")[:500],
        },
    )
    if vr.status == "ambiguous":
        result.ambiguous += 1
    else:
        result.extraction_failed += 1


# --- the LLM path (one work, one transaction) -------------------------------

def _process_llm_work(
    session: "Session",
    conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection,
    cache_root,
    lens: "LensDefinition",
    *,
    work_id: str,
    markdown_id: str,
    markdown_hash: str,
    definition_hash: str,
    access_class: str,
    route,
    completion_text: str,
    input_tokens: int,
    output_tokens: int,
    result: LensRunResult,
) -> None:
    run_id = new_id("extr")
    external_full_text = 1 if getattr(route, "external_full_text", False) else 0
    model_name = getattr(route, "profile_id", None)
    provider = getattr(route, "provider", None)
    access_mode = getattr(route, "access_mode", None)

    candidate = extract_json_object(completion_text or "")
    parsed = None
    if candidate is not None:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None

    if not isinstance(parsed, dict):
        # Invalid JSON overall -> extraction_failed + review item, one run (no drop).
        _insert_lens_run(
            conn,
            run_id=run_id,
            lens=lens,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            definition_hash=definition_hash,
            access_class=access_class,
            run_status="extraction_failed",
            model_name=model_name,
            provider=provider,
            access_mode=access_mode,
            external_full_text=external_full_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        fields = {
            "status": "extraction_failed",
            "error": "invalid JSON overall",
            "raw": (completion_text or "")[:2000],
        }
        lo_id = _insert_lens_output(
            conn,
            run_id=run_id,
            lens_id=lens.lens_id,
            work_id=work_id,
            status="extraction_failed",
            fields=fields,
            access_class=access_class,
        )
        enqueue_in_session(
            session,
            ReviewItemType.lens_record.value,
            target_type="lens_output",
            target_id=lo_id,
            payload={
                "kind": "lens_record",
                "lens_id": lens.lens_id,
                "work_id": work_id,
                "status": "extraction_failed",
                "reason": "invalid JSON overall",
                "lens_output_id": lo_id,
                "raw": (completion_text or "")[:500],
            },
        )
        result.extraction_failed += 1
        result.extraction_run_ids.append(run_id)
        session.commit()
        return

    _insert_lens_run(
        conn,
        run_id=run_id,
        lens=lens,
        work_id=work_id,
        markdown_id=markdown_id,
        markdown_hash=markdown_hash,
        definition_hash=definition_hash,
        access_class=access_class,
        run_status="success",
        model_name=model_name,
        provider=provider,
        access_mode=access_mode,
        external_full_text=external_full_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    ensure_fts_tables(session)

    records = parsed.get("records")
    records = records if isinstance(records, list) else []
    produced = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        _project_found_record(
            session,
            conn,
            cache_conn,
            cache_root,
            lens,
            work_id=work_id,
            markdown_id=markdown_id,
            run_id=run_id,
            record=record,
            field_index=produced,
            access_class=access_class,
            result=result,
        )
        produced += 1

    if produced == 0:
        # Explicit not_found row — full not-found shape round-tripped (decision 49).
        nf = parsed.get("not_found")
        nf = nf if isinstance(nf, dict) else {}
        fields = {
            "status": "not_found",
            "searched_sections": nf.get("searched_sections", []),
            "notes": nf.get("notes"),
            "confidence": nf.get("confidence"),
        }
        _insert_lens_output(
            conn,
            run_id=run_id,
            lens_id=lens.lens_id,
            work_id=work_id,
            status="not_found",
            fields=fields,
            access_class=access_class,
            confidence=nf.get("confidence"),
        )
        result.not_found += 1

    result.extraction_run_ids.append(run_id)
    session.commit()


# --- the deterministic (no-LLM) path (shared with fallback.py) --------------

def _find_anchor_quote(
    text: str, positive_anchors: list[str], negative_anchors: list[str]
) -> tuple[str, str] | None:
    """First verbatim line containing a positive anchor and no negative anchor.

    Returns ``(quote_line, matched_anchor)`` or ``None``. The quote is a verbatim
    slice of ``text`` so the phase_3 anchor path can locate it exactly.
    """
    lower = text.lower()
    negs = [n.lower() for n in negative_anchors]
    for anchor in positive_anchors:
        a = anchor.lower()
        idx = lower.find(a)
        while idx != -1:
            line_start = text.rfind("\n", 0, idx) + 1
            line_end = text.find("\n", idx)
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end].strip()
            if line and not any(neg in line.lower() for neg in negs):
                return (line, anchor)
            idx = lower.find(a, idx + 1)
    return None


def deterministic_work(
    session: "Session",
    conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection,
    cache_root,
    lens: "LensDefinition",
    *,
    work_id: str,
    markdown_id: str,
    markdown_hash: str,
    definition_hash: str,
    access_class: str,
    result: LensRunResult,
) -> None:
    """No-LLM honest degradation for ONE work (decision 38/58).

    Anchor/substring candidate match over the work's markdown using
    ``positive_anchors`` minus ``negative_anchors``. A match writes an
    ``epistemic_type='deterministic'`` saved-search span as a ``lens_outputs`` row
    — NOT a typed/confidence-bearing claim (no ``extracted_claims``), leaving
    ``assertion_status`` NULL (never fabricated). No match -> an honest not_found.
    """
    run_id = new_id("extr")
    _insert_lens_run(
        conn,
        run_id=run_id,
        lens=lens,
        work_id=work_id,
        markdown_id=markdown_id,
        markdown_hash=markdown_hash,
        definition_hash=definition_hash,
        access_class=access_class,
        run_status="success",
        model_name=None,  # no model — honest no-LLM degradation
    )
    result.degraded = True
    result.capability_note = (
        "no permitted LLM profile; deterministic anchor/FTS saved-search fallback "
        "(epistemic_type=deterministic, assertion_status NULL, no typed claims)"
    )

    md = cache_access.read_markdown(cache_conn, cache_root, markdown_id)
    match = None
    if md is not None:
        match = _find_anchor_quote(md.text, lens.positive_anchors, lens.negative_anchors)

    if match is not None:
        quote, anchor_term = match
        span_id = ensure_span(
            conn,
            cache_conn,
            cache_root,
            markdown_id=markdown_id,
            work_id=work_id,
            exact_quote=quote,
            access_class=access_class,
            span_kind="manual",
        )
        fields = {
            "status": "found",
            "epistemic_type": "deterministic",
            "assertion_status": None,  # NEVER fabricated for a non-claim span
            "matched_anchor": anchor_term,
            "candidate_quote": quote,
            "evidence_span_ids": [span_id] if span_id else [],
            "note": "deterministic saved-search candidate (no LLM)",
        }
        _insert_lens_output(
            conn,
            run_id=run_id,
            lens_id=lens.lens_id,
            work_id=work_id,
            status="found",
            fields=fields,
            access_class=access_class,
            claim_id=None,  # NOT a typed claim
            confidence=None,  # NEVER confidence-bearing
        )
        result.found += 1
    else:
        fields = {
            "status": "not_found",
            "epistemic_type": "deterministic",
            "assertion_status": None,
            "searched_sections": [],
            "notes": "no positive_anchor match (deterministic no-LLM pass)",
            "confidence": None,
        }
        _insert_lens_output(
            conn,
            run_id=run_id,
            lens_id=lens.lens_id,
            work_id=work_id,
            status="not_found",
            fields=fields,
            access_class=access_class,
        )
        result.not_found += 1

    result.extraction_run_ids.append(run_id)
    session.commit()


# --- service lifts (one source of truth for CLI + UI) -----------------------

def build_lens_router(config: "ProjectConfig") -> "LensRouter":
    """Build the :class:`LensRouter` for ``config`` (the CLI/UI shared factory).

    Resolves the backend through the offline test seam (``_BACKEND_OVERRIDE``) then
    the lazy ``default_backend`` registry, so both the CLI ``lens run/calibrate``
    path and the web lens screens construct routers the same way (no duplication)."""
    from ..llm.backend import default_backend

    backend = _BACKEND_OVERRIDE or default_backend(provider=None, model=None)
    return LensRouter(config=config, backend=backend)


def included_lens_work_ids(h) -> list[str]:
    """Work ids of the project's ``included`` corpus — the default lens target set."""
    from ..project.service import corpus_works

    return [w.work_id for w in corpus_works(h, ("included",))]


def lens_results(session: "Session", lens_id: str, status: str = "found") -> list[dict]:
    """Serializable lens-result rows shared by ``lens results --json`` and the UI.

    Lifts the CLI ``lens results --json`` payload construction over
    :func:`lenses.results.lens_results` so the JSON shape (the phase-6 success
    criterion) has one source of truth."""
    from .results import lens_results as _result_views

    out: list[dict] = []
    for v in _result_views(session, lens_id, status=status):
        out.append(
            {
                "lens_output_id": v.lens_output_id,
                "work_id": v.work_id,
                "status": v.status,
                "claim_id": v.claim_id,
                "claim_text": v.claim_text,
                "normalized_label": v.normalized_label,
                "section": v.section,
                "condition_type": v.fields.get("condition_type"),
                "confidence": v.confidence,
                "span_ids": v.span_ids,
                "access_class": v.access_class,
            }
        )
    return out


# --- public entry -----------------------------------------------------------

def run_lens(
    session: "Session",
    cache_db: Path,
    lens: "LensDefinition",
    work_ids: list[str],
    router: "LensRouter",
    *,
    sample: bool = False,
    force: bool = False,
    on_step: Optional[Callable[[str], None]] = None,
) -> LensRunResult:
    """Run ``lens`` over ``work_ids`` (the LLM path), returning a `LensRunResult`.

    For each work: resolve markdown (skip+record if ``None``); apply the
    idempotency guard unless ``force``; resolve `access_class` fail-closed; route
    `project_lens_extraction`; validate records + enforce evidence_policy; anchor
    verbatim quotes to `evidence_spans`; write `extracted_claims` + `claim_spans`
    + `claim_fts` for found records; write the `lens_outputs` row
    (found/not_found/ambiguous/extraction_failed with full `fields_json`); write
    the `extraction_run` stamped with `lens_definition_hash` + `markdown_hash`.
    ``sample=True`` marks the lens ``calibrating`` (does not promote to active);
    ``force`` reprocesses every targeted work append-only (prior runs preserved).
    Implements plan §6.1 idempotency semantics + §7 provenance/staleness.

    ``cache_db`` is the seedgraph HOME root used to open ``cache.db`` read-only
    (``None`` => ``$SEEDGRAPH_HOME``), mirroring every other cross-scope accessor.

    ``on_step`` (optional) is invoked with each ``work_id`` at the START of its
    iteration — the per-item heartbeat hook the background-job ``Progress`` handle
    binds to (:mod:`seedgraph.progress`); ``None`` (the CLI default) makes it a no-op.
    """
    result = LensRunResult(lens_id=lens.lens_id)
    cache_root = cache_db
    definition_hash = lens.definition_hash()
    conn = raw_conn(session)
    cache_conn = cache_access.open_cache_ro(cache_root)
    try:
        for work_id in work_ids:
            # Per-item heartbeat (shared Progress abstraction). Stepped at the START of
            # each item — a live "processing work_id now" signal for the run bar; the
            # aggregate outcome counts land in the terminal summary. No-op when unset.
            if on_step is not None:
                on_step(work_id)
            resolved = resolve_work_markdown(session, work_id=work_id)
            if resolved is None:
                result.skipped_no_markdown.append(work_id)
                continue
            markdown_id, markdown_hash = resolved

            if not force and _is_current(
                conn, lens.lens_id, work_id, definition_hash, markdown_hash
            ):
                result.skipped_current.append(work_id)
                continue

            access_class = resolve_source_access_class(cache_conn, markdown_id)
            md = cache_access.read_markdown(cache_conn, cache_root, markdown_id)
            if md is None:
                result.skipped_no_markdown.append(work_id)
                continue

            # Route the lens task; fall back to the deterministic no-LLM pass when
            # nothing is permitted (NoLlmRoute / content-gate refusal / no backend).
            try:
                route = resolve_route(TASK_TYPE, access_class, router.config)
            except ConfigError:
                route = None

            if router.backend is None or route is None or isinstance(route, NoLlmRoute):
                deterministic_work(
                    session,
                    conn,
                    cache_conn,
                    cache_root,
                    lens,
                    work_id=work_id,
                    markdown_id=markdown_id,
                    markdown_hash=markdown_hash,
                    definition_hash=definition_hash,
                    access_class=access_class,
                    result=result,
                )
                continue

            system_prompt, user_prompt = build_lens_prompt(lens, markdown_text=md.text)
            result.router_calls += 1
            # Dispatch through the executor (Track 1): a provider/transport error
            # returns a typed non-success result instead of raising, so a real key
            # + a failing transport degrades to the deterministic no-LLM path
            # (no traceback, and no "note extraction" message — that StubRealBackend
            # text was generalized to "this task").
            disp = _llm_dispatch(
                router.backend,
                system_prompt,
                user_prompt,
                model=getattr(route, "profile_id", None),
                provider=getattr(route, "provider", None),
                access_mode=getattr(route, "access_mode", None),
            )
            if disp.status != "success" or disp.completion is None:
                deterministic_work(
                    session,
                    conn,
                    cache_conn,
                    cache_root,
                    lens,
                    work_id=work_id,
                    markdown_id=markdown_id,
                    markdown_hash=markdown_hash,
                    definition_hash=definition_hash,
                    access_class=access_class,
                    result=result,
                )
                continue
            _process_llm_work(
                session,
                conn,
                cache_conn,
                cache_root,
                lens,
                work_id=work_id,
                markdown_id=markdown_id,
                markdown_hash=markdown_hash,
                definition_hash=definition_hash,
                access_class=access_class,
                route=route,
                completion_text=disp.completion.text,
                input_tokens=disp.input_tokens,
                output_tokens=disp.output_tokens,
                result=result,
            )
    finally:
        cache_conn.close()

    # Lifecycle: a sample run nudges a draft lens to calibrating (mutable; prior
    # runs go stale, never frozen). Promotion to ``active`` (and the one-time
    # definition_yaml snapshot) is the FULL-run boundary owned by the CLI
    # ``lens run`` verb / ``registry.promote_to_active`` — NOT this function — so
    # the revise loop (edit YAML while draft/calibrating -> sync_lens refreshes the
    # hash -> re-run) stays open until the user commits a full run.
    if sample:
        _mark_calibrating(session, lens)
    return result


def _mark_calibrating(session: "Session", lens: "LensDefinition") -> None:
    from ..vocab import LensStatus

    row = registry.get_lens_row(session, lens.lens_id)
    if row is not None and row.status == LensStatus.draft.value:
        registry.set_status(session, lens.lens_id, LensStatus.calibrating.value)
