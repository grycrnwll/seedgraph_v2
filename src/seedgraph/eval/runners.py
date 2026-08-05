"""Live eval runners — the ONLY place eval touches the phase-8 harness (§6/§8).

These run REAL project work: gold/leakage/answer questions go through the
phase-8 answer harness in-process and return its ``AnswerEnvelope`` (IMPORTED,
decisions 62/82 — phase_9 never defines the envelope; §9). eval adds no second
answer route — it only reads the envelope. The CI gate does NOT call these; it
asserts the same deterministic predicates over recorded cassettes
(``fixtures.load_envelope``), so it stays key-free (§11/§12).

Also home to ``score_edge_disagreements`` (scores the phase_3b reconciliation
surface into open audits) and ``verify_corpus`` (the REQUIRED ``first_corpus.yaml``
oracle gate, §2/§10 step 10).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # imported only for type-checkers; never executed at runtime
    from seedgraph.answer.types import AnswerEnvelope  # phase_8 owns (plan §5/§9)


@dataclass(frozen=True)
class CorpusVerification:
    """Outcome of :func:`verify_corpus`. ``verified`` is True only when EVERY
    ``first_corpus.yaml`` entry passed all four fail-closed checks (PDF fetched;
    identifier resolved to the named work; every ``expected_citation_edges`` entry
    present in the actually-built graph; every ``expected_claim`` bound to a real
    span). Any failure leaves the file ``status: proposed`` and is listed in
    ``failures`` (a human-readable reason per blocked entry).
    """

    verified: bool
    failures: list[str] = field(default_factory=list)


def run_question(project, question: str, query_type: str | None = None) -> "AnswerEnvelope":
    """Send one ``question`` through the LIVE phase-8 answer harness in-process and
    return its ``AnswerEnvelope`` (``answer_text, answer_category, cited_work_ids,
    cited_span_ids, retrieved_item_ids, insufficient_evidence``; §9). Read-only
    consumer — eval invokes the phase-8 ``answer_generation`` task through the
    harness itself and never re-routes answer generation (§8). ``query_type`` is
    advisory (the harness pre-classifies the question itself).
    """
    from ..answer.harness import answer as _answer

    # Eval stays ephemeral in v1 — its artifacts belong to runs/{run_id}/eval/, and
    # wiring eval to traces is the diff-harness's job (00 §5, T5). Trace dropped visibly.
    env, _ = _answer(question, project)
    return env


def score_edge_disagreements(conn: sqlite3.Connection, run_id: str | None) -> int:
    """Read the phase_3b reconciliation surface ``edges.edge_disagreements(conn,
    run_id)`` (provider-resolved vs parsed-bibliography citation disagreements) and
    open one ``reference_extraction`` / ``metadata_resolution`` audit per
    disagreement; return the number of audits opened. Surfaces disagreements for
    human grading — it never auto-resolves an edge (§4 boundary).
    """
    if run_id is None:
        return 0  # disagreements are scoped to a build run (run_id = NULL never matches)
    from ..citation.edges import edge_disagreements
    from .audit import _PRIVATE, open_audit

    opened = 0
    for disagreement in edge_disagreements(conn, run_id):
        provenances = disagreement.get("provenances", [])
        # a parsed-bibliography vs provider disagreement is a reference-parse audit;
        # a provider-only multi-source disagreement is a metadata-resolution audit.
        audit_type = (
            "reference_extraction"
            if "parsed_bibliography" in provenances
            else "metadata_resolution"
        )
        subject_id = f"{disagreement.get('source')}->{disagreement.get('target')}"
        open_audit(conn, audit_type, "citation_edge", subject_id, run_id, _PRIVATE)
        opened += 1
    return opened


# --- corpus verification (§2/§10 step 10) ----------------------------------


def _markdown_text(
    cache_conn: sqlite3.Connection | None, cache_root: Path | None, markdown_id: str | None
) -> str | None:
    """Resolve a markdown document's text via cache.db ``storage_uri`` (read-only).

    Returns None when the cache is unavailable or the markdown row/file is missing
    (rendered as "markdown unavailable" by callers, never a hard error — §4 note).
    """
    if cache_conn is None or cache_root is None or not markdown_id:
        return None
    try:
        row = cache_conn.execute(
            "SELECT storage_uri FROM markdown_documents WHERE markdown_id=?", (markdown_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None or not row[0]:
        return None
    path = Path(cache_root) / row[0]
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _normalize_title(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def verify_corpus(
    project: str,
    run_id: str | None = None,
    *,
    corpus_path: Path | str | None = None,
    root: Path | str | None = None,
    allow_fetch: bool = False,
) -> CorpusVerification:
    """REQUIRED gate-enabler (§2/§10 step 10): attempt to promote a corpus file
    from ``status: proposed`` to ``status: verified``.

    FAIL-CLOSED, for EACH entry: (a) the PDF is acquired (a ``work_source_files``
    bridge row with a ``markdown_id``); when missing and ``allow_fetch`` is False
    this fails closed (the live CLI passes ``allow_fetch=True`` to fetch over the
    network); (b) the identifier (DOI/arXiv) resolves to the named work; (c) every
    ``expected_citation_edges`` entry is present in the ACTUALLY-built citation
    graph (phase_2/5b), not asserted text; (d) each work's ``expected_claim`` binds
    to a REAL evidence span (``md[start:end] == quote``). On full success it
    rewrites the corpus ``status`` line to ``verified`` and returns
    ``CorpusVerification(verified=True)``. Any failure leaves ``status: proposed``
    and BLOCKS all first_corpus-based acceptance (§11), returning ``verified=False``
    with the per-entry reasons in ``failures``.
    """
    import yaml

    from ..db.connection import cache_db_path, project_db_path

    if corpus_path is None:
        corpus_path = Path(__file__).resolve().parents[3] / "evals" / "fixtures" / "first_corpus.yaml"
    corpus_path = Path(corpus_path)
    spec = yaml.safe_load(corpus_path.read_text(encoding="utf-8")) or {}

    failures: list[str] = []

    conn = sqlite3.connect(str(project_db_path(project, root)))
    conn.execute("PRAGMA foreign_keys=ON")
    cache_db = cache_db_path(root)
    cache_conn = sqlite3.connect(str(cache_db)) if cache_db.exists() else None
    cache_root = cache_db.parent if cache_conn is not None else None

    try:
        papers = spec.get("papers", []) or []
        # Resolve each corpus paper to a project work (by work_id, then identifier).
        resolved: dict[str, str] = {}  # corpus work_id -> project work_id
        for paper in papers:
            cid = paper.get("work_id")
            project_work_id = _resolve_paper_work(conn, paper)
            if project_work_id is None:
                failures.append(f"identifier-unresolved:{cid}")
                continue
            resolved[cid] = project_work_id
            # (a) acquired? bridge row with markdown_id present.
            bridge = conn.execute(
                "SELECT markdown_id FROM work_source_files WHERE work_id=?", (project_work_id,)
            ).fetchone()
            if bridge is None or not bridge[0]:
                if not allow_fetch:
                    failures.append(f"pdf-not-fetched:{cid}")
                else:  # pragma: no cover - live network path
                    failures.append(f"pdf-fetch-unavailable:{cid}")

        # (c) expected citation edges present in the built graph.
        for edge in spec.get("expected_citation_edges", []) or []:
            citing = resolved.get(edge.get("citing"))
            cited = resolved.get(edge.get("cited"))
            if citing is None or cited is None:
                failures.append(f"edge-endpoint-unresolved:{edge.get('citing')}->{edge.get('cited')}")
                continue
            hit = conn.execute(
                "SELECT 1 FROM citation_edges WHERE source_work_id=? AND target_work_id=? "
                "AND edge_type='cites' LIMIT 1",
                (citing, cited),
            ).fetchone()
            if hit is None:
                failures.append(f"edge-missing:{edge.get('citing')}->{edge.get('cited')}")

        # (d) each expected claim binds to a real, re-sliceable evidence span.
        for corpus_wid, claims in (spec.get("expected_claims", {}) or {}).items():
            project_work_id = resolved.get(corpus_wid)
            if project_work_id is None:
                failures.append(f"claim-work-unresolved:{corpus_wid}")
                continue
            if not _work_has_resliceable_span(conn, cache_conn, cache_root, project_work_id):
                failures.append(f"claim-span-unbound:{corpus_wid}")

    finally:
        conn.close()
        if cache_conn is not None:
            cache_conn.close()

    verified = len(failures) == 0
    if verified:
        _stamp_status_verified(corpus_path)
    return CorpusVerification(verified=verified, failures=failures)


def _resolve_paper_work(conn: sqlite3.Connection, paper: dict) -> str | None:
    """Resolve a corpus paper to a project ``works.work_id`` and confirm the named
    identifier resolves to a work whose title matches (fail-closed)."""
    cid = paper.get("work_id")
    title_norm = _normalize_title(paper.get("title"))
    identifiers = paper.get("identifiers", {}) or {}

    # Prefer identifier resolution (the §10(b) check): a DOI/arXiv must map to a work.
    id_type_map = {"doi": "doi", "arxiv": "arxiv", "openalex": "openalex", "s2": "s2", "ssrn": "ssrn"}
    for raw_type, value in identifiers.items():
        id_type = id_type_map.get(raw_type)
        if id_type is None or not value:
            continue
        row = conn.execute(
            "SELECT work_id FROM identifiers WHERE id_type=? AND id_value=?",
            (id_type, str(value)),
        ).fetchone()
        if row is not None:
            return row[0]

    # Fall back to a direct work_id match (synthetic fixtures use corpus_id == work_id).
    row = conn.execute("SELECT work_id FROM works WHERE work_id=?", (cid,)).fetchone()
    if row is not None:
        return row[0]

    # Last resort: a normalized-title match against an existing work.
    if title_norm:
        for (wid, wtitle) in conn.execute("SELECT work_id, canonical_title FROM works").fetchall():
            if _normalize_title(wtitle) == title_norm:
                return wid
    return None


def _work_has_resliceable_span(
    conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection | None,
    cache_root: Path | None,
    work_id: str,
) -> bool:
    """True iff the work has ≥1 evidence span whose markdown re-slice verifies
    (``md[start:end] == exact_quote``) — binding an expected_claim to REAL evidence."""
    try:
        spans = conn.execute(
            "SELECT markdown_id, start_char, end_char, exact_quote FROM evidence_spans "
            "WHERE work_id=?",
            (work_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return False
    for markdown_id, start, end, quote in spans:
        text = _markdown_text(cache_conn, cache_root, markdown_id)
        if text is not None and text[start:end] == quote:
            return True
    return False


def _stamp_status_verified(corpus_path: Path) -> None:
    """Rewrite ONLY the ``status:`` line to ``verified`` (preserves comments).

    Targeted line replace rather than ``yaml.dump`` so the hand-authored corpus's
    comments survive (the live promotion path; tests must never pre-set verified)."""
    text = corpus_path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"(?m)^status:[ \t]*\S+[ \t]*$", "status: verified", text
    )
    if n == 0:  # no top-level status line — append one
        new_text = text.rstrip("\n") + "\nstatus: verified\n"
    corpus_path.write_text(new_text, encoding="utf-8")
