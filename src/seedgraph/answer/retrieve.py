"""FTS5 + relational retrieval over the project corpus (plan §5 / §6.5, 08 §6).

Read-only over the three FTS5 vtables (``span_fts`` / ``claim_fts`` / ``note_fts``,
built phases 3-6) plus relational filters (project / work / section / claim_type /
lens / access_class) and a concept/alias lookup. **FTS5-only** — no vector backend
is built here (decisions 51/67; the ``rank_fusion()`` seam in ``rank.py`` is the
sole hook for the deferred vector signal).

Empty-/missing-table safety (plan §4 coverage gap): a project may be ingested but
not yet extracted, so an FTS table may be empty or (defensively) absent. Every
function here treats "table empty" and "table missing"
(``sqlite3.OperationalError: no such table``) identically — it returns ``[]`` and
never raises for a missing FTS table. Other execution errors propagate. The harness maps a fully-empty result to a clean
``insufficient_evidence`` short-circuit (no LLM call).

Provenance: each :class:`RetrievedItem` carries the raw ``bm25()`` score and both
D2 provenance fields (``epistemic_type`` origin + ``assertion_status``
stated/inferred) copied from the source ``extracted_claims`` row, so rank/guard
never re-query.
"""

from __future__ import annotations

import re
import sqlite3

from ..semantic.current import CURRENT_CLAIMS_CTE
from .types import QuerySpec, RetrievedItem

# Tiny stopword set so the question-token fallback issues meaningful FTS terms
# (deterministic; no NLP dependency).
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
        "was", "were", "be", "been", "by", "with", "as", "at", "it", "its", "this",
        "that", "these", "those", "what", "which", "who", "whom", "how", "why",
        "does", "do", "did", "can", "could", "should", "would", "will", "about",
        "between", "from", "into", "their", "they", "them", "there", "here",
        "explain", "describe", "tell", "me", "show", "compare", "versus", "vs",
    }
)

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")

# Inclusion-status retrieval scope (plan §7/§11 — the documented "retrieval filter
# every later retrieval path scopes through"). A work the project has marked
# ``excluded`` is NEVER surfaced: its spans / claims / notes are dropped at the SQL
# boundary, so a work excluded *after* extraction (which still retains citable spans)
# cannot leak into the citation set. ``included`` / ``metadata_only`` (and, defensively,
# a work with no ``project_documents`` row) pass. ``project_documents`` is a core
# project.db table (always present), so this clause does not affect the
# empty-/missing-FTS → ``[]`` contract — that is handled by :func:`_safe_fetchall`.
_NOT_EXCLUDED = (
    " NOT IN (SELECT pd.work_id FROM project_documents pd "
    "WHERE pd.inclusion_status = 'excluded')"
)


def _quote_phrase(text: str) -> str:
    """Wrap ``text`` as a single phrase-safe FTS5 token (internal quotes doubled)."""
    return '"' + text.strip().replace('"', '""') + '"'


def _match_query(spec: QuerySpec) -> str:
    """Build the FTS5 ``MATCH`` string from the spec.

    Exact phrases (``"Assumption 2"``) are issued verbatim as FTS5 phrases; absent
    phrases fall back to an OR of the question's content tokens for recall. Returns
    ``""`` when there is nothing to search (caller short-circuits to ``[]``).
    """
    parts: list[str] = [_quote_phrase(p) for p in spec.phrases if p.strip()]
    if not parts:
        seen: set[str] = set()
        for token in _WORD_RE.findall(spec.question.lower()):
            if len(token) < 3 or token in _STOPWORDS or token in seen:
                continue
            seen.add(token)
            parts.append(_quote_phrase(token))
    return " OR ".join(parts)


def _safe_fetchall(conn: sqlite3.Connection, sql: str, params: tuple) -> list:
    """Map missing optional FTS tables to ``[]``; propagate execution failures.

    Both "table empty" and "table missing" (``no such table``) yield ``[]`` so the
    harness short-circuit is identical for an unextracted vs partially-built project
    (plan §4). Corrupt or incompatible core tables are genuine failures.
    """
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table:" in str(exc) and any(
            name in str(exc) for name in ("span_fts", "claim_fts", "note_fts")
        ):
            return []
        raise


def concept_lookup(conn: sqlite3.Connection, spec: QuerySpec) -> list[RetrievedItem]:
    """08 §5 step 3 — resolve candidate concept tokens to project concept claims.

    Relational ``LIKE`` over ``concepts.normalized_label`` + ``concept_aliases``
    (no ``concept_fts`` — concepts are few; plan §4 over-engineering-skipped), then
    expands each matched concept through ``claim_concepts`` → ``extracted_claims``
    to citable claim-kind :class:`RetrievedItem`s (with any linked span). Empty /
    no matching concept yields ``[]``; incompatible schemas raise.
    """
    tokens = [t for t in (spec.concept_tokens or []) if t.strip()]
    tokens += [p for p in spec.phrases if p.strip()]
    if not tokens:
        return []
    items: list[RetrievedItem] = []
    seen: set[str] = set()
    for token in tokens:
        like = f"%{token.strip().lower()}%"
        rows = _safe_fetchall(
            conn,
            CURRENT_CLAIMS_CTE + "SELECT c.claim_id, c.work_id, c.normalized_label, c.claim_text, "
            "c.claim_type, c.epistemic_type, c.assertion_status, c.access_class, "
            "co.concept_id, co.weight, co.paper_frequency, "
            "(SELECT cs.span_id FROM claim_spans cs WHERE cs.claim_id = c.claim_id "
            " ORDER BY cs.rank LIMIT 1) AS span_id "
            "FROM concepts co "
            "LEFT JOIN concept_aliases a ON a.concept_id = co.concept_id "
            "JOIN claim_concepts cc ON cc.concept_id = co.concept_id "
            "JOIN current_claims c ON c.claim_id = cc.claim_id "
            "WHERE (lower(co.normalized_label) LIKE ? OR lower(a.alias_label) LIKE ?) "
            "AND c.work_id" + _NOT_EXCLUDED,
            (like, like),
        )
        for (
            claim_id, work_id, label, text, ctype, epi, assertion, access,
            concept_id, weight, paper_frequency, span_id,
        ) in rows:
            if claim_id in seen:
                continue
            seen.add(claim_id)
            body = text or label or ""
            if not body:
                continue
            items.append(
                RetrievedItem(
                    item_id=claim_id,
                    kind="claim",
                    work_id=work_id,
                    span_id=span_id,
                    claim_id=claim_id,
                    concept_id=concept_id,
                    text=body,
                    bm25_score=0.0,  # concept hit is exact (no FTS rank)
                    access_class=access or "user_supplied_private",
                    epistemic_type=epi,
                    assertion_status=assertion,
                    # decision 71: the matched concept's discriminativeness rides
                    # onto the item so rank.py can down-rank ubiquitous concepts.
                    concept_weight=weight,
                    concept_paper_frequency=paper_frequency,
                )
            )
    return items


def spans(conn: sqlite3.Connection, spec: QuerySpec, limit: int) -> list[RetrievedItem]:
    """08 §5 step 5 — retrieve evidence spans via ``span_fts`` (phrase-safe).

    ``text`` is the verbatim ``evidence_spans.exact_quote`` (decision 53). Carries
    the linked claim's D2 provenance + a ``claim_id`` when a span anchors a claim
    (drives the rank span-has-claim-link boost). Relational filters: work / section
    kind / access_class. Empty / missing table → ``[]``.
    """
    match = _match_query(spec)
    if not match:
        return []
    sql = [
        CURRENT_CLAIMS_CTE + "SELECT e.span_id, e.work_id, e.section_id, e.exact_quote, e.access_class, "
        "       bm25(span_fts) AS rank, "
        "       (SELECT cs.claim_id FROM claim_spans cs JOIN current_claims c ON c.claim_id=cs.claim_id "
        "        WHERE cs.span_id = e.span_id "
        "        ORDER BY cs.rank LIMIT 1) AS claim_id, "
        "       d.section_kind AS section_kind "
        "FROM span_fts JOIN evidence_spans e ON e.span_id = span_fts.span_id "
        "LEFT JOIN document_sections d ON d.section_id = e.section_id "
        "WHERE span_fts MATCH ? AND e.work_id" + _NOT_EXCLUDED,
    ]
    params: list[object] = [match]
    if spec.work_id is not None:
        sql.append("AND e.work_id = ?")
        params.append(spec.work_id)
    if spec.section is not None:
        sql.append("AND d.section_kind = ?")
        params.append(spec.section)
    if spec.access_class is not None:
        sql.append("AND e.access_class = ?")
        params.append(spec.access_class)
    sql.append("ORDER BY rank LIMIT ?")
    params.append(int(limit))

    out: list[RetrievedItem] = []
    for (span_id, work_id, section_id, quote, access, rank, claim_id, section_kind) in _safe_fetchall(
        conn, " ".join(sql), tuple(params)
    ):
        epi = None
        assertion = None
        if claim_id is not None:
            crow = conn.execute(
                "SELECT epistemic_type, assertion_status FROM extracted_claims WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()
            if crow is not None:
                epi, assertion = crow[0], crow[1]
        out.append(
            RetrievedItem(
                item_id=span_id,
                kind="span",
                work_id=work_id,
                span_id=span_id,
                claim_id=claim_id,
                section=section_kind,
                text=quote,
                bm25_score=float(rank),
                access_class=access or "user_supplied_private",
                epistemic_type=epi,
                assertion_status=assertion,
            )
        )
    return out


def claims(conn: sqlite3.Connection, spec: QuerySpec, limit: int) -> list[RetrievedItem]:
    """08 §5 step 4 — retrieve claims via ``claim_fts`` (+ relational filters).

    Copies ``epistemic_type`` + ``assertion_status`` (D2) onto each item and resolves
    the highest-rank linked span (so a cited claim still resolves to a verbatim span).
    Filters: claim_type / access_class. Empty / missing table → ``[]``.
    """
    match = _match_query(spec)
    if not match:
        return []
    sql = [
        CURRENT_CLAIMS_CTE + "SELECT c.claim_id, c.work_id, c.normalized_label, c.claim_text, c.claim_type, "
        "       c.epistemic_type, c.assertion_status, c.access_class, "
        "       bm25(claim_fts) AS rank, "
        "       (SELECT cs.span_id FROM claim_spans cs WHERE cs.claim_id = c.claim_id "
        "        ORDER BY cs.rank LIMIT 1) AS span_id "
        "FROM claim_fts JOIN current_claims c ON c.claim_id = claim_fts.claim_id "
        "WHERE claim_fts MATCH ? AND c.work_id" + _NOT_EXCLUDED,
    ]
    params: list[object] = [match]
    if spec.work_id is not None:
        sql.append("AND c.work_id = ?")
        params.append(spec.work_id)
    if spec.claim_type is not None:
        sql.append("AND c.claim_type = ?")
        params.append(spec.claim_type)
    if spec.access_class is not None:
        sql.append("AND c.access_class = ?")
        params.append(spec.access_class)
    sql.append("ORDER BY rank LIMIT ?")
    params.append(int(limit))

    out: list[RetrievedItem] = []
    for (claim_id, work_id, label, text, ctype, epi, assertion, access, rank, span_id) in _safe_fetchall(
        conn, " ".join(sql), tuple(params)
    ):
        body = text or label or ""
        if not body:
            continue
        out.append(
            RetrievedItem(
                item_id=claim_id,
                kind="claim",
                work_id=work_id,
                span_id=span_id,
                claim_id=claim_id,
                text=body,
                bm25_score=float(rank),
                access_class=access or "user_supplied_private",
                epistemic_type=epi,
                assertion_status=assertion,
            )
        )
    return out


def notes(conn: sqlite3.Connection, spec: QuerySpec, limit: int) -> list[RetrievedItem]:
    """08 §5 — retrieve structured notes via ``note_fts``. Empty / missing → ``[]``."""
    match = _match_query(spec)
    if not match:
        return []
    sql = [
        CURRENT_CLAIMS_CTE + "SELECT n.note_id, n.work_id, n.note_text, n.access_class, bm25(note_fts) AS rank "
        "FROM note_fts JOIN structured_notes n ON n.note_id = note_fts.note_id "
        "WHERE note_fts MATCH ? AND n.note_id IN (SELECT note_id FROM current_notes) "
        "AND n.work_id" + _NOT_EXCLUDED,
    ]
    params: list[object] = [match]
    if spec.work_id is not None:
        sql.append("AND n.work_id = ?")
        params.append(spec.work_id)
    if spec.access_class is not None:
        sql.append("AND n.access_class = ?")
        params.append(spec.access_class)
    sql.append("ORDER BY rank LIMIT ?")
    params.append(int(limit))

    out: list[RetrievedItem] = []
    for (note_id, work_id, text, access, rank) in _safe_fetchall(
        conn, " ".join(sql), tuple(params)
    ):
        if not text:
            continue
        out.append(
            RetrievedItem(
                item_id=note_id,
                kind="note",
                work_id=work_id,
                note_id=note_id,
                text=text,
                bm25_score=float(rank),
                access_class=access or "user_supplied_private",
            )
        )
    return out


def retrieve(
    conn: sqlite3.Connection,
    project_slug: str,
    spec: QuerySpec,
    limit: int,
) -> list[RetrievedItem]:
    """Top-level retrieval (plan §6.5): merge concept/claim/span/note retrieval into
    one deduplicated, project-scoped :class:`RetrievedItem` list capped at ``limit``.

    Every query runs against *this* project's ``project.db`` (private-by-default,
    decisions 30/60/76 — structural isolation, no ``project_id`` filter needed) and
    applies the ``spec`` filters at the SQL boundary. Returns ``[]``
    when the corpus is empty or an FTS table is missing — the harness owns the
    resulting clean short-circuit.
    """
    merged: list[RetrievedItem] = []
    seen: set[str] = set()
    # Spans first (the citable verbatim unit), then claims, concept-expansion, notes.
    for batch in (
        spans(conn, spec, limit),
        claims(conn, spec, limit),
        concept_lookup(conn, spec),
        notes(conn, spec, limit),
    ):
        for item in batch:
            if item.item_id in seen:
                continue
            seen.add(item.item_id)
            merged.append(item)
    return merged[:limit]
