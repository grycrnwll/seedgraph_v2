"""audit_records CRUD + seeded sampling + verdict writes (phase_9 §4/§6).

This module owns the APPEND-ONLY measurement surface (``audit_records``). It
records graded verdicts and the accept/reject/edit validation decision; it NEVER
mutates graph state — a concept-merge graph fix is enacted through ``review_queue``
(decisions 56/71/80; §4 boundary rule), so a single merge has at most one audit
row (the grade) and at most one review_queue resolution (the fix).

Reproducibility is recorded ONCE per draw via ``sample_batch_id`` + ``sample_seed``
(no frozen JSON manifest, no ``subject_fingerprint``/staleness — phase_9 §2).
``access_class`` is stamped FAIL-CLOSED (most-restrictive over every contributing
work; no-source subject → project-private; unknown/unresolved → user_supplied_private
— phase_9 §7), enforced here rather than by a column CHECK.

Also home to the re-enabled (r2 gap-12) subject samplers: ``sample_metadata_resolution``
(phase_5b ``identifier`` subjects) and ``sample_reference_extraction`` (phase_3b
``reference_entry`` subjects).
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from ..vocab import AccessClass

# doc 09 §11 target draw sizes per subject kind (10 papers / 20 claims / 20 spans /
# 10 merges / 10 edges / 5 answers / 5 boundary checks). draw_sample under-draws
# gracefully when the corpus has fewer items than the target.
SAMPLE_COUNTS: dict[str, int] = {
    "work": 10,
    "claim": 20,
    "span": 20,
    "concept_merge": 10,
    "citation_edge": 10,
    "answer": 5,
    "content_access_boundary": 5,
}

# audit_type soft vocab (phase_9 §4) — every value has a named producer.
AUDIT_TYPES: frozenset[str] = frozenset(
    {
        "duplicate_detection",
        "markdown_conversion",
        "span_indexing",
        "span_relevance",
        "claim_faithfulness",
        "concept_merge",
        "concept_identity_noise",
        "citation_resolution",
        "metadata_resolution",
        "reference_extraction",
        "unresolved_target",
        "retrieval",
        "answer_faithfulness",
        "corpus_leakage",
        "content_access_boundary",
        # Build C chunk 9 (D10/decision 56): the per-build canon decision log,
        # produced by semantic/audit.record_canon_decision_log (one row per
        # concepts build; status='resolved', JSON in the 0016 payload column).
        "canon_decision_log",
    }
)

# subject_type soft polymorphic vocab (phase_9 §4) — heterogeneous, NO cross-table FK.
SUBJECT_TYPES: frozenset[str] = frozenset(
    {
        "claim",
        "span",
        "concept_merge",
        "answer",
        "citation_edge",
        "work",
        "source_file",
        "markdown",
        "identifier",
        "reference_entry",
        # Build C chunk 9: the whole type-scoped concept set of one canon build;
        # subject_id is its concept_set_hash (semantic/audit.concept_set_hash).
        "concept_set",
    }
)

# Validation decision + severity vocab (decision 80; doc 09 §4).
DECISIONS: frozenset[str] = frozenset({"accept", "reject", "edit"})
SEVERITIES: frozenset[str] = frozenset({"low", "medium", "high", "critical"})

# Per §11 sample-kind → (audit_type, subject_type) routing. ``content_access_boundary``
# has no in-corpus subject row, so it under-draws to 0 (handled by boundary.py).
_KIND_AUDIT_TYPE: dict[str, str] = {
    "work": "markdown_conversion",
    "claim": "claim_faithfulness",
    "span": "span_relevance",
    "concept_merge": "concept_merge",
    "citation_edge": "citation_resolution",
    "answer": "answer_faithfulness",
    "content_access_boundary": "content_access_boundary",
}
_KIND_SUBJECT_TYPE: dict[str, str] = {
    "work": "work",
    "claim": "claim",
    "span": "span",
    "concept_merge": "concept_merge",
    "citation_edge": "citation_edge",
    "answer": "answer",
    "content_access_boundary": "content_access_boundary",
}

# Fixed kind order — the deterministic draw sequence (same seed ⇒ same draw).
_SAMPLE_ORDER: tuple[str, ...] = (
    "work",
    "claim",
    "span",
    "concept_merge",
    "citation_edge",
    "answer",
    "content_access_boundary",
)

_PRIVATE = AccessClass.user_supplied_private.value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SampleBatch:
    """One seeded §11 draw. ``sample_batch_id`` + ``sample_seed`` are the SINGLE
    reproducibility record (phase_9 §2/§4) — re-running with the same seed
    reproduces the draw. ``audit_ids`` are the opened (status='open') rows;
    ``counts`` is the realized per-subject-kind size (may be < SAMPLE_COUNTS when
    the corpus under-draws).
    """

    sample_batch_id: str
    sample_seed: str
    audit_ids: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def _query_ids(conn: sqlite3.Connection, sql: str) -> list[str]:
    """Return a sorted list of stringified ids; [] if the table is absent."""
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.OperationalError:
        return []
    return sorted(str(row[0]) for row in rows)


def _candidates(conn: sqlite3.Connection, kind: str) -> list[str]:
    """Deterministically-ordered candidate subject ids for a §11 sample kind."""
    if kind == "work":
        return _query_ids(conn, "SELECT work_id FROM works")
    if kind == "claim":
        return _query_ids(conn, "SELECT claim_id FROM extracted_claims")
    if kind == "span":
        return _query_ids(conn, "SELECT span_id FROM evidence_spans")
    if kind == "concept_merge":
        return _query_ids(conn, "SELECT alias_id FROM concept_aliases")
    if kind == "citation_edge":
        return _query_ids(conn, "SELECT edge_id FROM citation_edges")
    if kind == "answer":
        return _query_ids(conn, "SELECT answer_id FROM answers")
    return []  # content_access_boundary — no subject rows


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple) -> str | None:
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    return row[0]


def _subject_access_class(conn: sqlite3.Connection, kind: str, subject_id: str) -> str:
    """Fail-closed access_class for a sampled subject (phase_9 §7).

    Single-source subjects read their own row's ``access_class``; a work folds the
    most-restrictive over its spans; everything unknown / no-source defaults to the
    most-restrictive ``user_supplied_private``.
    """
    raw: str | None = None
    if kind == "claim":
        raw = _scalar(conn, "SELECT access_class FROM extracted_claims WHERE claim_id=?", (subject_id,))
    elif kind == "span":
        raw = _scalar(conn, "SELECT access_class FROM evidence_spans WHERE span_id=?", (subject_id,))
    elif kind == "concept_merge":
        raw = _scalar(
            conn,
            "SELECT c.access_class FROM concept_aliases a "
            "JOIN concepts c ON c.concept_id = a.concept_id WHERE a.alias_id=?",
            (subject_id,),
        )
    elif kind == "work":
        # Most-restrictive over this work's evidence spans (fail-closed if none).
        try:
            classes = [
                r[0]
                for r in conn.execute(
                    "SELECT access_class FROM evidence_spans WHERE work_id=?", (subject_id,)
                ).fetchall()
            ]
        except sqlite3.OperationalError:
            classes = []
        if classes:
            return AccessClass.most_restrictive(*classes).value
        raw = None
    # citation_edge / answer / boundary → no per-row access column → fail closed.
    return AccessClass.most_restrictive(raw).value


def _insert_audit(
    conn: sqlite3.Connection,
    audit_type: str,
    subject_type: str,
    subject_id: str,
    run_id: str | None,
    access_class: str,
    *,
    sample_batch_id: str | None = None,
    sample_seed: str | None = None,
) -> str:
    if audit_type not in AUDIT_TYPES:
        raise ValueError(f"unknown audit_type {audit_type!r}")
    if subject_type not in SUBJECT_TYPES:
        raise ValueError(f"unknown subject_type {subject_type!r}")
    audit_id = "audit_" + uuid4().hex
    conn.execute(
        "INSERT INTO audit_records (audit_id, audit_type, subject_type, subject_id, "
        "run_id, sample_batch_id, sample_seed, status, access_class, created_at) "
        "VALUES (?,?,?,?,?,?,?,'open',?,?)",
        (
            audit_id,
            audit_type,
            subject_type,
            subject_id,
            run_id,
            sample_batch_id,
            sample_seed,
            access_class,
            _now(),
        ),
    )
    conn.commit()
    return audit_id


def draw_sample(conn: sqlite3.Connection, run_id: str | None, seed: int) -> SampleBatch:
    """Draw the doc 09 §11 sample with a SEEDED RNG and open one ``audit_records``
    row per drawn item (status='open'), stamping a fresh ``sample_batch_id`` and
    the ``sample_seed`` on every row.

    Deterministic: the same ``(run_id, seed)`` over the same corpus reproduces the
    exact draw. Honors :data:`SAMPLE_COUNTS` per subject kind and UNDER-DRAWS
    gracefully when a kind has fewer items than its target. Each opened row's
    ``access_class`` is stamped fail-closed per the §7 propagation rule.
    """
    rng = random.Random(seed)
    sample_batch_id = "batch_" + uuid4().hex
    sample_seed = str(seed)
    audit_ids: list[str] = []
    counts: dict[str, int] = {}
    for kind in _SAMPLE_ORDER:
        population = _candidates(conn, kind)
        target = SAMPLE_COUNTS[kind]
        if len(population) <= target:
            drawn = list(population)  # graceful under-draw (take all)
        else:
            drawn = rng.sample(population, target)
        counts[kind] = len(drawn)
        audit_type = _KIND_AUDIT_TYPE[kind]
        subject_type = _KIND_SUBJECT_TYPE[kind]
        for subject_id in sorted(drawn):
            access_class = _subject_access_class(conn, kind, subject_id)
            audit_ids.append(
                _insert_audit(
                    conn,
                    audit_type,
                    subject_type,
                    subject_id,
                    run_id,
                    access_class,
                    sample_batch_id=sample_batch_id,
                    sample_seed=sample_seed,
                )
            )
    return SampleBatch(sample_batch_id, sample_seed, audit_ids, counts)


def open_audit(
    conn: sqlite3.Connection,
    audit_type: str,
    subject_type: str,
    subject_id: str,
    run_id: str | None,
    access_class: str,
) -> str:
    """Insert one open ``audit_records`` row and return its ``audit_id``
    (``'audit_' + uuid4().hex``, decision 65).

    ``audit_type`` ∈ :data:`AUDIT_TYPES`, ``subject_type`` ∈ :data:`SUBJECT_TYPES`
    (soft vocab, validated here — not by a SQL CHECK). ``access_class`` is the
    already-propagated, fail-closed value (§7); ``status`` defaults to 'open' and
    ``created_at`` is stamped now. Append-only — never mutates graph state (§4).
    """
    return _insert_audit(conn, audit_type, subject_type, subject_id, run_id, access_class)


def record_verdict(
    conn: sqlite3.Connection,
    audit_id: str,
    decision: str,
    verdict: str | None = None,
    severity: str | None = None,
    problem: str | None = None,
    fix: str | None = None,
    edit_payload: str | None = None,
    reviewer: str = "human",
) -> None:
    """Record the grade for an open audit row and mark it resolved.

    Writes ``decision`` ∈ :data:`DECISIONS`, optional ``verdict``/``severity``
    (∈ :data:`SEVERITIES`)/``problem``/``recommended_fix``, and — when
    ``decision='edit'`` — persists ``edit_payload`` (JSON). Sets ``reviewer``,
    ``resolved_at`` and ``status='resolved'``. IDEMPOTENT once
    ``status='resolved'`` (a re-call on a resolved row is a no-op, not a
    re-grade). For ``audit_type='concept_merge'`` the graph-state fix is enacted
    separately via ``review_queue`` — this call only grades (§4 boundary).
    """
    row = conn.execute(
        "SELECT status FROM audit_records WHERE audit_id=?", (audit_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown audit_id {audit_id!r}")
    if row[0] == "resolved":
        return  # idempotent — already graded, never re-grade (§11)
    if decision not in DECISIONS:
        raise ValueError(f"unknown decision {decision!r}; expected one of {sorted(DECISIONS)}")
    if severity is not None and severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}; expected one of {sorted(SEVERITIES)}")
    conn.execute(
        "UPDATE audit_records SET decision=?, verdict=?, severity=?, problem=?, "
        "recommended_fix=?, edit_payload=?, reviewer=?, status='resolved', resolved_at=? "
        "WHERE audit_id=?",
        (decision, verdict, severity, problem, fix, edit_payload, reviewer, _now(), audit_id),
    )
    conn.commit()


def sample_metadata_resolution(
    conn: sqlite3.Connection, run_id: str | None, seed: int
) -> list[str]:
    """Open ``metadata_resolution`` audits over phase_5b ``identifier`` subjects
    (r2 gap-12): ``review_queue`` ``duplicate_candidate`` items (phase_5b's
    metadata-resolution review surface). The plan also names ``identifiers`` rows
    with ``resolution_status`` ∈ {ambiguous, unresolved}; the on-disk
    ``identifiers`` table carries no ``resolution_status`` column, so the concrete
    schema-backed signal is the ``review_queue`` ``duplicate_candidate`` route.
    Deterministic draw; returns the opened ``audit_id`` list.
    ``subject_type='identifier'``.
    """
    item_ids = _query_ids(
        conn, "SELECT item_id FROM review_queue WHERE item_type='duplicate_candidate'"
    )
    opened: list[str] = []
    for item_id in item_ids:
        opened.append(
            open_audit(conn, "metadata_resolution", "identifier", item_id, run_id, _PRIVATE)
        )
    return opened


def sample_reference_extraction(
    conn: sqlite3.Connection, run_id: str | None, seed: int
) -> list[str]:
    """Open ``reference_extraction`` audits over phase_3b ``reference_entry``
    subjects (r2 gap-12): ``reference_entries`` rows with ``resolution_status`` ∈
    {ambiguous, suspect}. Deterministic draw; returns the opened ``audit_id`` list.
    ``subject_type='reference_entry'``.
    """
    ref_ids = _query_ids(
        conn,
        "SELECT reference_id FROM reference_entries "
        "WHERE resolution_status IN ('ambiguous','suspect')",
    )
    opened: list[str] = []
    for ref_id in ref_ids:
        opened.append(
            open_audit(conn, "reference_extraction", "reference_entry", ref_id, run_id, _PRIVATE)
        )
    return opened
