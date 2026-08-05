"""Constrained polymorphic edge writer + concept-edge builders (plan §6, step 8).

``project_graph_edges`` is interpretive-edges-only (deterministic citation edges
stay in ``citation_edges``). SQLite cannot FK a polymorphic column, so endpoint
integrity is enforced at this **single application insert path**: every
``(node_type, node_id)`` is validated against the home-table registry
:data:`NODE_HOME_TABLES` before the row is written; ``doctor`` re-scans for
dangling endpoints.

Builders produce three edge families:
  * ``Work --discusses--> Concept`` — aggregated from ``claim_concepts``;
    inherits the strongest contributing ``epistemic_type``.
  * ``Concept --related_to/broader_than/narrower_than/contrasts_with--> Concept``
    — interpretive, ``llm_inferred`` (borderline → ``concept_edge_candidate``).
  * ``Concept --co_occurs_with--> Concept`` — deterministic structural signal
    (shared-claim/shared-paper ≥ k), on a DISTINCT ``edge_type`` so the weak
    structural signal is never confused with asserted relatedness (doc 07 §7).

Span binding is intentionally NOT exposed this phase: ``edge_spans`` stays
unpopulated (no concept-layer edge carries single-span evidence yet).
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from uuid import uuid4

from ..errors import ValidationError
from ..vocab import AccessClass

#: Polymorphic node-type → home-table resolver registry. The insert path uses
#: this to verify an endpoint resolves to a real row before writing the edge.
NODE_HOME_TABLES: dict[str, str] = {
    "Work": "works",
    "Concept": "concepts",
    "Claim": "extracted_claims",
    "EvidenceSpan": "evidence_spans",
}

#: Primary-key column for each home table (used by the endpoint validator).
NODE_PK_COLUMNS: dict[str, str] = {
    "works": "work_id",
    "concepts": "concept_id",
    "extracted_claims": "claim_id",
    "evidence_spans": "span_id",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_endpoint(conn: sqlite3.Connection, node_type: str, node_id: str) -> None:
    """Polymorphic-FK substitute: reject an endpoint that does not resolve to a row."""
    table = NODE_HOME_TABLES.get(node_type)
    if table is None:
        raise ValidationError(
            f"unknown node_type {node_type!r}; must be one of {sorted(NODE_HOME_TABLES)}"
        )
    pk = NODE_PK_COLUMNS[table]
    row = conn.execute(
        f"SELECT 1 FROM {table} WHERE {pk} = ? LIMIT 1", (node_id,)
    ).fetchone()
    if row is None:
        raise ValidationError(
            f"dangling edge endpoint: ({node_type}, {node_id}) has no row in {table}"
        )


def _pair_access(conn: sqlite3.Connection, concept_id_a: str, concept_id_b: str) -> str:
    """MAX (most-restrictive) ``access_class`` over two concepts' denormalized values."""
    rows = conn.execute(
        "SELECT access_class FROM concepts WHERE concept_id IN (?, ?)",
        (concept_id_a, concept_id_b),
    ).fetchall()
    return AccessClass.most_restrictive(*[r[0] for r in rows]).value


def insert_edge(
    conn: sqlite3.Connection,
    *,
    src: tuple[str, str],
    tgt: tuple[str, str],
    edge_type: str,
    epistemic_type: str,
    access_class: str,
    run_id: str,
    confidence: float | None = None,
    shared_count: int | None = None,
) -> str:
    """Validate endpoints and insert one ``project_graph_edges`` row; return its
    ``edge_id`` (``'pge_' || uuid4hex``).

    ``src``/``tgt`` are ``(node_type, node_id)`` pairs. Both node types must be in
    :data:`NODE_HOME_TABLES` and each ``node_id`` must resolve to a real row in
    its home table (the polymorphic-FK substitute). ``edge_type`` is normalized
    against the open ``GraphEdgeType`` vocab; ``epistemic_type`` must be a closed
    ``EpistemicType`` value (the DB CHECK is the backstop). ``confidence`` /
    ``shared_count`` carry co-occurrence strength provenance (Build B chunk 7);
    the ``None`` defaults preserve every pre-existing caller. Idempotent under
    the ``(src, tgt, edge_type, run_id)`` UNIQUE constraint. Raises on a dangling
    endpoint or bad node/epistemic type. Pinned by ``test_project_graph_edges``.
    """
    src_type, src_id = src
    tgt_type, tgt_id = tgt
    _validate_endpoint(conn, src_type, src_id)
    _validate_endpoint(conn, tgt_type, tgt_id)

    existing = conn.execute(
        "SELECT edge_id FROM project_graph_edges "
        "WHERE source_node_type=? AND source_node_id=? AND target_node_type=? "
        "AND target_node_id=? AND edge_type=? AND IFNULL(run_id,'')=IFNULL(?,'')",
        (src_type, src_id, tgt_type, tgt_id, edge_type, run_id),
    ).fetchone()
    if existing is not None:
        return existing[0]

    edge_id = "pge_" + uuid4().hex
    # epistemic_type / node_type are backstopped by the DB CHECKs (a bad value
    # raises sqlite3.IntegrityError here — the closed-vocab guard).
    conn.execute(
        "INSERT INTO project_graph_edges "
        "(edge_id, source_node_type, source_node_id, target_node_type, target_node_id, "
        "edge_type, epistemic_type, confidence, shared_count, access_class, run_id, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            edge_id,
            src_type,
            src_id,
            tgt_type,
            tgt_id,
            edge_type,
            epistemic_type,
            confidence,
            shared_count,
            access_class,
            run_id,
            _now(),
        ),
    )
    return edge_id


def build_discusses_edges(conn: sqlite3.Connection, *, run_id: str) -> int:
    """Build ``Work --discusses--> Concept`` edges aggregated from
    ``claim_concepts``; return the count written.

    One edge per ``(work, concept)`` pair that shares ≥1 ``claim_concepts`` row;
    ``epistemic_type`` inherits the strongest contributing link (``deterministic``
    if any deterministic, else ``llm_extracted``). ``access_class`` is resolved
    from the contributing claims (:func:`access.resolve_access_class`).
    """
    from .access import resolve_access_class

    pairs = conn.execute(
        "SELECT work_id, concept_id FROM claim_concepts GROUP BY work_id, concept_id"
    ).fetchall()
    count = 0
    for work_id, concept_id in pairs:
        rows = conn.execute(
            "SELECT epistemic_type, claim_id FROM claim_concepts "
            "WHERE work_id=? AND concept_id=?",
            (work_id, concept_id),
        ).fetchall()
        epistemics = {r[0] for r in rows}
        claim_ids = [r[1] for r in rows]
        strongest = "deterministic" if "deterministic" in epistemics else "llm_extracted"
        access = resolve_access_class(conn, claim_ids)
        insert_edge(
            conn,
            src=("Work", work_id),
            tgt=("Concept", concept_id),
            edge_type="discusses",
            epistemic_type=strongest,
            access_class=access,
            run_id=run_id,
        )
        count += 1
    return count


def build_interpretive_edges(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    proposals: list[tuple[str, str, str, float]],
) -> int:
    """Insert LLM-inferred ``Concept--Concept`` interpretive edges; return count.

    Each proposal is ``(source_concept_id, target_concept_id, edge_type,
    confidence)`` with ``edge_type`` ∈ {``related_to``, ``broader_than``,
    ``narrower_than``, ``contrasts_with``}. Edges are written ``llm_inferred`` and
    borderline ones are enqueued as ``concept_edge_candidate`` review items (the
    ``UserValidation --validates/rejects--> Edge`` workflow). ``llm_inferred`` is
    NEVER written to ``citation_edges``.
    """
    from ..project.review import enqueue_raw

    count = 0
    for source_concept_id, target_concept_id, edge_type, confidence in proposals:
        access = _pair_access(conn, source_concept_id, target_concept_id)
        edge_id = insert_edge(
            conn,
            src=("Concept", source_concept_id),
            tgt=("Concept", target_concept_id),
            edge_type=edge_type,
            epistemic_type="llm_inferred",
            access_class=access,
            run_id=run_id,
        )
        enqueue_raw(
            conn,
            "concept_edge_candidate",
            target_type="Edge",
            target_id=edge_id,
            payload={
                "kind": "concept_edge_candidate",
                "edge_id": edge_id,
                "edge_type": edge_type,
                "source_concept_id": source_concept_id,
                "target_concept_id": target_concept_id,
                "confidence": confidence,
                "run_id": run_id,
            },
        )
        count += 1
    return count


def build_co_occurs_edges(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    min_shared: int = 2,
) -> int:
    """Build deterministic ``Concept --co_occurs_with--> Concept`` edges; return
    count.

    Two concepts co-occur when they are mentioned together in ≥ ``min_shared``
    shared claims/papers. Tagged ``epistemic_type='deterministic'`` on the
    distinct ``co_occurs_with`` edge_type. This is the no-LLM / structural signal
    (decision 38); it is NOT an asserted ``related_to`` relation.

    Strength provenance (Build B chunk 7): each kept edge carries
    ``confidence = jaccard = |P(A) ∩ P(B)| / |P(A) ∪ P(B)|`` over the two
    concepts' work-sets, and ``shared_count = |P(A) ∩ P(B)|`` — so an export can
    distinguish a 2-shared pair from a 20-shared pair.
    """
    rows = conn.execute(
        "SELECT DISTINCT work_id, concept_id FROM claim_concepts"
    ).fetchall()
    by_work: dict[str, set[str]] = defaultdict(set)
    works_by_concept: dict[str, set[str]] = defaultdict(set)
    for work_id, concept_id in rows:
        by_work[work_id].add(concept_id)
        works_by_concept[concept_id].add(work_id)

    pair_counts: Counter = Counter()
    for concept_ids in by_work.values():
        ordered = sorted(concept_ids)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pair_counts[(ordered[i], ordered[j])] += 1

    count = 0
    for (a, b), shared in sorted(pair_counts.items()):
        if shared < min_shared:
            continue
        union = len(works_by_concept[a] | works_by_concept[b])
        jaccard = shared / union if union else 0.0
        access = _pair_access(conn, a, b)
        insert_edge(
            conn,
            src=("Concept", a),
            tgt=("Concept", b),
            edge_type="co_occurs_with",
            epistemic_type="deterministic",
            access_class=access,
            run_id=run_id,
            confidence=jaccard,
            shared_count=shared,
        )
        count += 1
    return count
