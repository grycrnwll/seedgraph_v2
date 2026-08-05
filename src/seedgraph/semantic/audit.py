"""Eval hooks for the concept overlay (plan §11 — measure, never mutate).

Emits sampled concept-merge audit candidates into ``audit_records`` for manual
over/under-merge grading, and surfaces cross-run constraint-flip detection. This
module only MEASURES (writes ``audit_records`` rows / reports flips); it never
mutates graph state — that is ``review_queue``'s job (decisions 56/71/80).
``audit_records`` is owned by phase_9's ``0012_evaluation.sql`` (+ the additive
``payload`` column from ``0016_audit_payload.sql``).

Build C chunk 9 (D10; decision 56's flip-detection mandate) adds the canon
decision log: :func:`concept_set_hash` keys one build's type-scoped concept set,
:func:`record_canon_decision_log` writes exactly ONE ``audit_records`` row per
concepts build (``audit_type='canon_decision_log'``), and
:func:`compare_canon_decision_logs` diffs the two most recent logs — same hash
plus any decision delta is a reported machine-side flip; a different hash means
the corpus itself changed (borderline clusters legitimately re-roll).
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def concept_set_hash(pairs: Iterable[tuple[str, str]]) -> str:
    """Stable sha256 hex digest of one build's ``(concept_type, normalized_label)``
    set — the canon decision-log key (D10; near-verbatim port of prototype
    concept_canon.py:324-351).

    Sorting + de-duping make the hash invariant to input ordering and repeats, so
    the same corpus snapshot always keys the same decision log. **Documented
    limitation (as in the prototype):** the hash is NOT stable under corpus
    growth — adding a paper that introduces a new label changes the set and
    re-rolls borderline clusters; :func:`compare_canon_decision_logs` reports
    that case honestly instead of calling it a flip.
    """
    dedup = {(str(concept_type), str(label)) for concept_type, label in pairs}
    payload = json.dumps(sorted(dedup), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_canon_decision_log(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    set_hash: str,
    decisions: dict[str, Any],
) -> str:
    """Write the ONE per-build canon decision-log row into ``audit_records``.

    ``audit_type='canon_decision_log'``, ``subject_type='concept_set'``,
    ``subject_id=set_hash``, ``status='resolved'`` (it is a machine record, not
    an open grading item), ``decisions`` JSON in the 0016 ``payload`` column.
    ``access_class`` is stamped fail-closed (the set may derive from private
    works). ``decisions`` must be JSON-serializable and must NOT embed run ids
    or timestamps — the comparator diffs payloads byte-semantically, and
    run-varying noise would masquerade as flips (run_id lives in its column).
    Returns the ``audit_id``. Caller commits (build_semantic_overlay's commit).
    """
    audit_id = "audit_" + uuid4().hex
    now = _now()
    conn.execute(
        "INSERT INTO audit_records "
        "(audit_id, audit_type, subject_type, subject_id, run_id, status, "
        "access_class, payload, created_at, resolved_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            audit_id,
            "canon_decision_log",
            "concept_set",
            set_hash,
            run_id,
            "resolved",
            "user_supplied_private",
            json.dumps(decisions, ensure_ascii=False, sort_keys=True),
            now,
            now,
        ),
    )
    return audit_id


@dataclass(frozen=True)
class CanonLogComparison:
    """Outcome of diffing the two most recent canon decision logs.

    ``status`` ∈ {``'insufficient_logs'``, ``'same_set'``, ``'set_changed'``}.
    ``flips`` is non-empty only for ``'same_set'`` with a decision delta — the
    decision-56 machine-side flip signal (identical inputs, different outcome).
    A ``'set_changed'`` report is honest, not alarming: the corpus grew and
    borderline clusters legitimately re-rolled.
    """

    status: str
    earlier_hash: str | None = None
    later_hash: str | None = None
    flips: tuple[dict[str, Any], ...] = ()
    summary: str = ""


# Per-cluster decision fields the comparator diffs (must match the shape
# concepts.build_concepts accumulates).
_CLUSTER_FIELDS = ("deterministic_folds", "cannot_link_skips", "proposals")
_TOP_FIELDS = ("must_link_applied", "must_link_skipped")


def _cluster_key(cluster: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        str(cluster.get("concept_type", "")),
        tuple(cluster.get("members", ())),
    )


def _diff_decisions(
    earlier: dict[str, Any], later: dict[str, Any]
) -> list[dict[str, Any]]:
    """Field-by-field diff of two same-hash decision payloads → flip entries."""
    flips: list[dict[str, Any]] = []
    e_clusters = {_cluster_key(c): c for c in earlier.get("clusters", [])}
    l_clusters = {_cluster_key(c): c for c in later.get("clusters", [])}
    for key in sorted(set(e_clusters) - set(l_clusters)):
        flips.append(
            {"kind": "cluster_removed", "concept_type": key[0], "members": list(key[1])}
        )
    for key in sorted(set(l_clusters) - set(e_clusters)):
        flips.append(
            {"kind": "cluster_added", "concept_type": key[0], "members": list(key[1])}
        )
    for key in sorted(set(e_clusters) & set(l_clusters)):
        for field_name in _CLUSTER_FIELDS:
            before = e_clusters[key].get(field_name)
            after = l_clusters[key].get(field_name)
            if before != after:
                flips.append(
                    {
                        "kind": "decision_changed",
                        "concept_type": key[0],
                        "members": list(key[1]),
                        "field": field_name,
                        "before": before,
                        "after": after,
                    }
                )
    for field_name in _TOP_FIELDS:
        if earlier.get(field_name) != later.get(field_name):
            flips.append(
                {
                    "kind": "decision_changed",
                    "field": field_name,
                    "before": earlier.get(field_name),
                    "after": later.get(field_name),
                }
            )
    return flips


def compare_canon_decision_logs(conn: sqlite3.Connection) -> CanonLogComparison:
    """Diff the two most recent ``canon_decision_log`` rows (decision 56).

    Same ``concept_set_hash`` on both rows ⇒ identical inputs, so ANY decision
    delta is a reported machine-side flip (``status='same_set'``, one ``flips``
    entry per changed cluster field). Different hashes ⇒ the corpus changed;
    report that honestly (``status='set_changed'``, zero flips) instead of
    diffing decisions made over different label sets. Fewer than two rows ⇒
    ``status='insufficient_logs'``. Callable from tests now; a future eval CLI
    verb is the natural consumer.
    """
    rows = conn.execute(
        "SELECT subject_id, payload FROM audit_records "
        "WHERE audit_type='canon_decision_log' "
        "ORDER BY created_at DESC, rowid DESC LIMIT 2"
    ).fetchall()
    if len(rows) < 2:
        return CanonLogComparison(
            status="insufficient_logs",
            summary="fewer than two canon decision logs recorded; nothing to compare",
        )
    (later_hash, later_payload), (earlier_hash, earlier_payload) = rows
    if earlier_hash != later_hash:
        return CanonLogComparison(
            status="set_changed",
            earlier_hash=earlier_hash,
            later_hash=later_hash,
            summary=(
                "concept set changed between builds (corpus grew, borderline "
                "clusters re-rolled); flip comparison not applicable"
            ),
        )
    earlier = json.loads(earlier_payload or "{}")
    later = json.loads(later_payload or "{}")
    flips = tuple(_diff_decisions(earlier, later))
    if flips:
        summary = (
            f"{len(flips)} machine-side flip(s) detected on an UNCHANGED "
            "concept set (same concept_set_hash)"
        )
    else:
        summary = "zero flips: identical decisions on an unchanged concept set"
    return CanonLogComparison(
        status="same_set",
        earlier_hash=earlier_hash,
        later_hash=later_hash,
        flips=flips,
        summary=summary,
    )


def emit_merge_audit_sample(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    sample_size: int = 10,
    seed: int = 0,
) -> int:
    """Sample up to ``sample_size`` folded concepts and write ``concept_merge``
    audit rows into ``audit_records`` for manual grading; return the count written.

    A folded concept is one that absorbed ≥2 surface labels (an alias-bearing
    fold). The draw is reproducible via ``seed`` (recorded as ``sample_seed`` +
    ``sample_batch_id``). ``access_class`` propagates fail-closed from the concept.
    """
    rows = conn.execute(
        "SELECT c.concept_id, c.access_class, COUNT(a.alias_id) AS n "
        "FROM concepts c JOIN concept_aliases a ON a.concept_id = c.concept_id "
        "GROUP BY c.concept_id HAVING n >= 2 ORDER BY c.concept_id"
    ).fetchall()
    if not rows:
        return 0
    rnd = random.Random(seed)
    sample = rows if len(rows) <= sample_size else rnd.sample(rows, sample_size)
    batch_id = "batch_" + uuid4().hex[:12]
    now = _now()
    written = 0
    for concept_id, access_class, _n in sample:
        conn.execute(
            "INSERT INTO audit_records "
            "(audit_id, audit_type, subject_type, subject_id, run_id, sample_batch_id, "
            "sample_seed, status, access_class, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "audit_" + uuid4().hex,
                "concept_merge",
                "concept_merge",
                concept_id,
                run_id,
                batch_id,
                str(seed),
                "open",
                access_class or "user_supplied_private",
                now,
            ),
        )
        written += 1
    return written


def detect_constraint_flips(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return label pairs that carry BOTH a ``must_link`` and a ``cannot_link``
    constraint — a cross-run human decision flip (doc 09 §6, §11).

    Sticky constraints should never contradict; a pair appearing in both is the
    flip signal surfaced for review."""
    must = {
        (a, b)
        for a, b in conn.execute(
            "SELECT label_a, label_b FROM concept_constraints WHERE kind='must_link'"
        ).fetchall()
    }
    cannot = {
        (a, b)
        for a, b in conn.execute(
            "SELECT label_a, label_b FROM concept_constraints WHERE kind='cannot_link'"
        ).fetchall()
    }
    return sorted(must & cannot)
