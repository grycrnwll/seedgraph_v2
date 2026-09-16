"""Concept read projections — the milestone query path (doc 10 §11).

The shared core behind both ``seedgraph concepts show`` and the read-only API
``GET /projects/{slug}/concepts/{concept_id}`` (decision 81 — one core, two thin
surfaces). :func:`concept_detail` resolves a concept to its linked **papers**,
**claims**, and **evidence spans** via the relational path
``concepts → claim_concepts → extracted_claims → claim_spans → evidence_spans``.
All project.db-local; no cache.db read.
"""

from __future__ import annotations

import sqlite3

from .current import CURRENT_CLAIMS_CTE


def list_concepts(
    conn: sqlite3.Connection,
    *,
    concept_type: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Tabular concept list (concept_id, canonical_label, type, paper_frequency,
    weight, status, access_class), optionally filtered by ``concept_type`` /
    ``status``.

    Default order is SHARP-FIRST (``weight DESC`` — the anti-stopword IDF
    lesson; the old ``paper_frequency DESC`` was hub-first, the exact inverse).
    Weight is *discriminativeness, not importance*: a concept in every staged
    work weighs ~0 without being unimportant, so never present weight as an
    importance ranking (v1 SKILL.md rule).
    """
    clauses: list[str] = []
    params: list[str] = []
    if concept_type:
        clauses.append("concept_type = ?")
        params.append(concept_type)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        "SELECT concept_id, canonical_label, concept_type, paper_frequency, weight, "
        "status, access_class FROM concepts" + where + " ORDER BY weight DESC, concept_id",
        params,
    ).fetchall()
    return [
        {
            "concept_id": r[0],
            "canonical_label": r[1],
            "concept_type": r[2],
            "paper_frequency": r[3],
            "weight": r[4],
            "status": r[5],
            "access_class": r[6],
        }
        for r in rows
    ]


# Research-salience order for the `concept_overview` by-type blocks. This is a
# DIFFERENT axis from ``list_concepts``' ``weight DESC`` (IDF, sharp-first — right
# for query-time retrieval): the overview orients a reader by *what kind of thing,
# how often it recurs*, so it ranks by paper_frequency inside a curated type order.
# Types not in this tuple are appended alphabetically by the function.
_OVERVIEW_TYPE_ORDER = (
    "identification_assumption",
    "general_assumption",
    "regularity_condition",
    "estimand",
    "limitation",
    "robustness_check",
    "open_question",
    "research_question",
    "main_contribution",
    "setting",
    "method",
    "data",
    "result",
)


def concept_overview(
    conn: sqlite3.Connection,
    *,
    per_type_cap: int = 12,
    min_pf: int = 2,
    background_n: int = 8,
) -> dict:
    """Compact, recurrence-ranked orientation sheet over the concept overlay.

    Pure on-the-fly aggregation over the ``concepts`` table (top-K over a small
    indexed table — microseconds); NO new table/column/materialization. Solves the
    dogfood friction where the overlay on a real corpus is ~12.8k concepts (87.5%
    singletons) and dumping the whole table drowns orientation. Returns four parts:

    * ``scale`` — ``{total_concepts, total_papers, singleton_count, type_counts}``.
      ``total_papers`` is distinct works in ``claim_concepts`` (the extraction set
      that ``paper_frequency`` is relative to — NOT the ``works`` table, which also
      holds the metadata-only citation frontier).
    * ``background_frame`` — the top ``background_n`` concepts by ``paper_frequency``
      across ALL types: the ubiquitous "assumed context" generics. These are
      EXCLUDED from ``by_type`` so a type block features its *distinctive* recurring
      concepts rather than re-listing corpus-wide hubs.
    * ``by_type`` — per concept_type (in ``_OVERVIEW_TYPE_ORDER``, then unlisted
      types alphabetically) a block ``{concept_type, type_count, featured,
      more_count}``. ``featured`` = top by paper_frequency, excluding background
      ids, filtered to ``paper_frequency >= min_pf``, capped at ``per_type_cap``.
      ``more_count = type_count - len(featured)`` so ``featured`` + ``more_count``
      equals the row count of ``concepts list --type <T>`` (the queryable tail).
      A type with zero featured concepts is skipped (still counted in
      ``scale.type_counts``).
    * ``tail_hint`` — ``{not_shown_count, commands}`` acknowledging everything not
      on the sheet and how to reach it.

    Recurrence ranks by *distinct papers*, and the labels are an LLM-extracted
    interpretive overlay (CORE_CONCEPT §4) — specifics should be grounded via
    ``concepts show`` / ``search`` / evidence spans, not read off these labels.
    """
    total_concepts = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    total_papers = conn.execute(
        "SELECT COUNT(DISTINCT work_id) FROM claim_concepts"
    ).fetchone()[0]
    singleton_count = conn.execute(
        "SELECT COUNT(*) FROM concepts WHERE paper_frequency <= 1"
    ).fetchone()[0]
    type_counts = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT concept_type, COUNT(*) FROM concepts GROUP BY concept_type"
        ).fetchall()
    }

    # Background frame: the corpus-wide hubs. Deterministic secondary key
    # (concept_id) breaks paper_frequency ties so the sheet is stable across runs.
    bg_rows = conn.execute(
        "SELECT concept_id, canonical_label, paper_frequency, concept_type "
        "FROM concepts ORDER BY paper_frequency DESC, concept_id LIMIT ?",
        (background_n,),
    ).fetchall()
    background_frame = [
        {"canonical_label": r[1], "paper_frequency": r[2], "concept_type": r[3]}
        for r in bg_rows
    ]
    background_ids = {r[0] for r in bg_rows}

    present_types = set(type_counts)
    ordered_types = [t for t in _OVERVIEW_TYPE_ORDER if t in present_types]
    ordered_types += sorted(present_types - set(_OVERVIEW_TYPE_ORDER))

    by_type: list[dict] = []
    for ctype in ordered_types:
        rows = conn.execute(
            "SELECT concept_id, canonical_label, paper_frequency FROM concepts "
            "WHERE concept_type = ? AND paper_frequency >= ? "
            "ORDER BY paper_frequency DESC, concept_id",
            (ctype, min_pf),
        ).fetchall()
        featured = [
            {"concept_id": r[0], "canonical_label": r[1], "paper_frequency": r[2]}
            for r in rows
            if r[0] not in background_ids
        ][:per_type_cap]
        if not featured:
            continue
        type_count = type_counts[ctype]
        by_type.append(
            {
                "concept_type": ctype,
                "type_count": type_count,
                "featured": featured,
                "more_count": type_count - len(featured),
            }
        )

    shown = len(background_frame) + sum(len(b["featured"]) for b in by_type)
    tail_hint = {
        "not_shown_count": total_concepts - shown,
        "commands": ["concepts list --type <T>", "search <q>"],
    }

    return {
        "scale": {
            "total_concepts": total_concepts,
            "total_papers": total_papers,
            "singleton_count": singleton_count,
            "type_counts": type_counts,
        },
        "background_frame": background_frame,
        "by_type": by_type,
        "tail_hint": tail_hint,
    }


def concept_detail(conn: sqlite3.Connection, concept_id: str) -> dict | None:
    """Return ``{concept, papers[], claims[], spans[]}`` for one concept, or ``None``.

    The doc 10 §11 success criterion: select a concept → see its linked papers,
    claims, and evidence spans. Joins are project.db-local only.
    """
    row = conn.execute(
        "SELECT concept_id, normalized_label, canonical_label, concept_type, definition, "
        "paper_frequency, weight, status, epistemic_type, access_class FROM concepts "
        "WHERE concept_id = ?",
        (concept_id,),
    ).fetchone()
    if row is None:
        return None
    concept = {
        "concept_id": row[0],
        "normalized_label": row[1],
        "canonical_label": row[2],
        "concept_type": row[3],
        "definition": row[4],
        "paper_frequency": row[5],
        "weight": row[6],
        "status": row[7],
        "epistemic_type": row[8],
        "access_class": row[9],
    }

    papers = [
        {"work_id": r[0], "title": r[1], "year": r[2]}
        for r in conn.execute(
            CURRENT_CLAIMS_CTE + "SELECT DISTINCT w.work_id, w.canonical_title, w.year "
            "FROM claim_concepts cc JOIN works w ON w.work_id = cc.work_id "
            "JOIN current_claims ec ON ec.claim_id = cc.claim_id "
            "WHERE cc.concept_id = ? ORDER BY w.work_id",
            (concept_id,),
        ).fetchall()
    ]

    claims = [
        {
            "claim_id": r[0],
            "work_id": r[1],
            "claim_type": r[2],
            "claim_text": r[3],
            "epistemic_type": r[4],
        }
        for r in conn.execute(
            CURRENT_CLAIMS_CTE + "SELECT ec.claim_id, ec.work_id, ec.claim_type, ec.claim_text, cc.epistemic_type "
            "FROM claim_concepts cc JOIN current_claims ec ON ec.claim_id = cc.claim_id "
            "WHERE cc.concept_id = ? ORDER BY ec.claim_id",
            (concept_id,),
        ).fetchall()
    ]

    spans = [
        {
            "span_id": r[0],
            "work_id": r[1],
            "exact_quote": r[2],
            "access_class": r[3],
        }
        for r in conn.execute(
            CURRENT_CLAIMS_CTE + "SELECT DISTINCT es.span_id, es.work_id, es.exact_quote, es.access_class "
            "FROM claim_concepts cc "
            "JOIN current_claims ec ON ec.claim_id = cc.claim_id "
            "JOIN claim_spans cs ON cs.claim_id = cc.claim_id "
            "JOIN evidence_spans es ON es.span_id = cs.span_id "
            "WHERE cc.concept_id = ? ORDER BY es.span_id",
            (concept_id,),
        ).fetchall()
    ]

    return {"concept": concept, "papers": papers, "claims": claims, "spans": spans}


def concept_provenance(
    conn: sqlite3.Connection,
    concept_id: str,
    work_id: str | None = None,
) -> list[dict]:
    """Flat joined claim+span provenance rows for one concept (Track 3 drawer).

    Joins ``claim_concepts → extracted_claims → works`` and **LEFT JOIN**
    ``claim_spans → evidence_spans → document_sections`` so a claim with no anchored
    span is still returned (its span columns are ``None``). Each row carries the
    ``claim_id`` + ``claim_type`` / ``claim_subtype`` / ``claim_text``, the span's
    ``span_id`` + ``exact_quote`` + ``start_char`` / ``end_char`` + ``page_start`` /
    ``page_end``, the section ``heading_path`` / ``heading_text``, the claim
    ``access_class`` (+ span ``span_access_class``), and the ``extraction_run_id``.

    Grouping by work is the route handler's job — this returns flat rows ordered by
    ``work_id, claim_id, rank`` so the handler can fold them with a single pass.
    Optionally narrowed to a single ``work_id``. project.db-local; no cache.db read.
    """
    sql = (
        CURRENT_CLAIMS_CTE + "SELECT ec.work_id, w.canonical_title, w.year, "
        "cc.claim_id, ec.claim_type, ec.claim_subtype, ec.claim_text, "
        "ec.extraction_run_id, ec.access_class, "
        "es.span_id, es.exact_quote, es.start_char, es.end_char, "
        "es.page_start, es.page_end, es.access_class, "
        "ds.heading_path, ds.heading_text, cs.rank "
        "FROM claim_concepts cc "
        "JOIN current_claims ec ON ec.claim_id = cc.claim_id "
        "JOIN works w ON w.work_id = ec.work_id "
        "LEFT JOIN claim_spans cs ON cs.claim_id = cc.claim_id "
        "LEFT JOIN evidence_spans es ON es.span_id = cs.span_id "
        "LEFT JOIN document_sections ds ON ds.section_id = es.section_id "
        "WHERE cc.concept_id = ?"
    )
    params: list[str] = [concept_id]
    if work_id is not None:
        sql += " AND ec.work_id = ?"
        params.append(work_id)
    sql += " ORDER BY ec.work_id, cc.claim_id, cs.rank"

    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "work_id": r[0],
            "title": r[1],
            "year": r[2],
            "claim_id": r[3],
            "claim_type": r[4],
            "claim_subtype": r[5],
            "claim_text": r[6],
            "extraction_run_id": r[7],
            "access_class": r[8],
            "span_id": r[9],
            "exact_quote": r[10],
            "start_char": r[11],
            "end_char": r[12],
            "page_start": r[13],
            "page_end": r[14],
            "span_access_class": r[15],
            "heading_path": r[16],
            "heading_text": r[17],
            "rank": r[18],
        }
        for r in rows
    ]


def concept_provenance_counts(
    conn: sqlite3.Connection,
    concept_ids: "list[str]",
) -> dict:
    """Per-concept ``{papers, claims, spans}`` rollup feeding the 3D transform's
    ``prov`` field.

    One grouped query over ``claim_concepts`` LEFT JOIN ``claim_spans`` →
    ``evidence_spans`` (a span-less claim still counts toward ``papers`` / ``claims``
    but contributes 0 spans). ``papers`` = distinct ``claim_concepts.work_id``,
    ``claims`` = distinct ``claim_id``, ``spans`` = distinct ``evidence_spans.span_id``.
    Returns ``{cid: {"papers", "claims", "spans"}}`` for every requested id (ids with
    no rows map to all-zero); ``{}`` for an empty ``concept_ids``.
    """
    if not concept_ids:
        return {}
    placeholders = ",".join("?" for _ in concept_ids)
    rows = conn.execute(
        CURRENT_CLAIMS_CTE + "SELECT cc.concept_id, "
        "COUNT(DISTINCT cc.work_id) AS papers, "
        "COUNT(DISTINCT cc.claim_id) AS claims, "
        "COUNT(DISTINCT es.span_id) AS spans "
        "FROM claim_concepts cc "
        "JOIN current_claims ec ON ec.claim_id = cc.claim_id "
        "LEFT JOIN claim_spans cs ON cs.claim_id = cc.claim_id "
        "LEFT JOIN evidence_spans es ON es.span_id = cs.span_id "
        f"WHERE cc.concept_id IN ({placeholders}) "
        "GROUP BY cc.concept_id",
        list(concept_ids),
    ).fetchall()
    counts = {cid: {"papers": 0, "claims": 0, "spans": 0} for cid in concept_ids}
    for cid, papers, claims, spans in rows:
        counts[cid] = {"papers": papers, "claims": claims, "spans": spans}
    return counts
