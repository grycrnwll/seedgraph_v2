""""This isn't the paper": move a suspect PDF to the paper it belongs to (researcher request
after the stage 3 walk-through; docs/V3_LIBRARY_DESIGN.md §7.3 handling of a wrong PDF).

    suggest_targets(lib, paper_id, source_file_id=None)        -> tuple[Suggestion, ...]
    find_papers(lib, query, *, exclude=())                       -> tuple[Suggestion, ...]
    plan_reassign_pdf(lib, paper_id, source_file_id, *, target_paper_id=None | new_paper=True)
                                                                  -> ReassignPlan
    reassign_pdf(lib, plan, *, confirm=True)                     -> ReassignResult

Two calls, as :mod:`seedgraph.library.merge` and :mod:`seedgraph.library.lifecycle`:
:func:`plan_reassign_pdf` is pure (what moves, every blocker, a fingerprint of the rows it
covers); :func:`reassign_pdf` re-plans, refuses a stale plan (:class:`StalePlanError`) or a
blocked one (:class:`ReassignError`) and applies it in one library transaction.

**Suggestions**: the cover region of the PDF's own text layer (as ingest reads it,
``metadata_quality.cover_region``; the converted markdown when the PDF has no text layer)
is checked against every other library paper's title with the standing title check
(:func:`titlecheck.check_prepared`, the DOI too). Papers that reach at least a weak match
(``titlecheck.MISS_THRESHOLD``) are listed best first; generic titles are skipped.
:func:`find_papers` is a plain title / id search for anything the check does not find.

Rules as implemented:

- **The PDF** (the source's ``paper_files`` row) moves to the receiving paper, which becomes
  held if it was cited-only. A target that already holds a *different* PDF is refused (a
  paper holds one PDF): merge the two papers if they are one paper (:func:`offer_merge` raises
  the ``identity_check`` the merge preview works from), or detach its PDF first.
  A target that already holds this same PDF (a shared PDF) keeps its link; the source's link
  goes. The receiving paper's title check is re-run after the move
  (:func:`files.recheck_pdf`): a miss raises a fresh ``pdf_title_miss`` item there.
- **New paper**: ``identity.mint_paper_id`` of the PDF-hash key (deterministic). The PDF's
  own DOI / arXiv id from its cover (ingest's ``deterministic_identifiers``) is stored with
  ``source='pdf_scan'`` when no paper holds it; an id another paper holds is named in a
  warning instead. A ``metadata_needed`` item is raised.
- **The source** becomes cited-only ("needs a PDF") unless it holds another file. In every
  graph where it is a ``member`` it drops to ``frontier`` (graphs refuse cited-only
  members), keeping ``is_seed`` and ``reason``; this is written before the library
  transaction, as lifecycle's delete does. The receiving paper is never added to a graph.
- **Notes follow the PDF**: extraction runs of the source on this PDF's markdown, and the
  note versions made on it (or by those runs), move with their claims (quote anchors are
  keyed by claim and follow), and lens outputs of those runs / claims. Versions are
  renumbered per schema on the receiving paper by ``(created_at, note_id)``, as merge does;
  the source's remaining versions keep their numbers (gaps are harmless). FTS rows are keyed
  by note / claim id and stay consistent. ``concept_refresh`` is set on both papers.
- **Researcher edits**: a source overlay whose base note or run moves, or whose ops name
  this PDF's markdown (a quote-backed added claim), blocks: resolve or revert the edits
  first. Work in flight (a queued / in-progress batch item) on either paper blocks.
- **Attention**: the source's open items about this PDF (``pdf_title_miss``,
  ``text_garbled``, ``text_missing``, ``renote_offered`` naming its file or markdown) are
  resolved ``wrong_pdf_moved`` / ``wrong_pdf_new_paper``; a ``shared_pdf`` item for the file
  too, once fewer than two papers hold it.
- **Not moved**: reference entries parsed from this markdown stay with the source (a warning
  gives the count); citation links are untouched.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import attention, files, graphs
from .identity import derive_key, mint_paper_id, normalize_ids
from .merge import ACTIVE_BATCH_STATES, PDF_KINDS
from .metadata_quality import clean_title, cover_region, generic_title_reason
from .store import Library, LibraryError, paper_exists
from .titlecheck import (MISS_THRESHOLD, check_prepared, first_pages_from_markdown, first_pages_from_pdf,
                         prepare_pages)

REFRESH_REASON = "pdf_reassign"
RESOLUTION_MOVED = "wrong_pdf_moved"
RESOLUTION_NEW = "wrong_pdf_new_paper"
METADATA_NEEDED = "metadata_needed"
SUGGEST_LIMIT = 8
SEARCH_LIMIT = 20
_TEMP_VERSION = 1_000_000  # renumbering goes through this offset (as merge does)
_RANK = {"pass": 3, "weak": 2, "miss": 1, "unknown": 0}


class ReassignError(LibraryError):
    """A PDF reassignment was refused (bad request, blocked plan, no confirm)."""


class StalePlanError(ReassignError):
    """The papers changed since the reassignment plan was made."""


@dataclass(frozen=True)
class Suggestion:
    paper_id: str
    title: str | None
    year: int | None
    held: bool
    score: float | None  # the title check's score against the PDF cover (None: a search hit)
    status: str | None  # the title check's verdict (pass / weak), None for a search hit
    evidence: str = ""


@dataclass(frozen=True)
class GraphChange:
    slug: str
    status: str
    is_seed: bool
    new_status: str
    reason: str | None = None


@dataclass(frozen=True)
class ReassignPlan:
    """What :func:`reassign_pdf` would do. Built only by :func:`plan_reassign_pdf`."""

    paper_id: str  # the source paper (loses the PDF)
    source_file_id: str
    target_paper_id: str | None  # None on the new-paper path
    new_paper: bool
    receiving_paper_id: str
    source_title: str | None
    target_title: str | None
    target_was_cited_only: bool
    target_has_same_pdf: bool
    target_holds_other_pdf: bool  # blocked: offer a merge instead (:func:`offer_merge`)
    source_becomes_cited_only: bool
    cover_identifiers: tuple[tuple[str, str], ...]  # new paper: free ids found on the cover
    counts: dict[str, int]
    graphs: tuple[GraphChange, ...]  # the source's member rows that drop to frontier
    source_graphs: tuple[str, ...]  # every graph the source is in (the "Add to graph" hint)
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    fingerprint: str
    note_order: tuple[tuple[str, int], ...] = field(repr=False, default=())
    run_ids: tuple[str, ...] = field(repr=False, default=())
    note_ids: tuple[str, ...] = field(repr=False, default=())
    claim_ids: tuple[str, ...] = field(repr=False, default=())
    lens_output_ids: tuple[str, ...] = field(repr=False, default=())
    item_resolutions: tuple[str, ...] = field(repr=False, default=())

    @property
    def ok(self) -> bool:
        return not self.blockers

    @property
    def resolution(self) -> str:
        return RESOLUTION_NEW if self.new_paper else RESOLUTION_MOVED


@dataclass(frozen=True)
class ReassignResult:
    paper_id: str
    source_file_id: str
    receiving_paper_id: str
    new_paper: bool
    items_resolved: tuple[str, ...]
    graphs_changed: tuple[str, ...]
    check: str | None  # the receiving paper's title check after the move
    new_item_id: str | None  # a fresh pdf_title_miss item on the receiving paper


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(conn: sqlite3.Connection, sql: str, args: tuple | list = ()) -> list[list]:
    return [list(r) for r in conn.execute(sql, args)]


def _ids(conn: sqlite3.Connection, sql: str, args: tuple | list = ()) -> tuple:
    return tuple(r[0] for r in conn.execute(sql, args))


def _json(values) -> str:
    return json.dumps(list(values))


# --- the PDF's cover -----------------------------------------------------------------------------


def _source_file_of(conn: sqlite3.Connection, paper_id: str, source_file_id: str | None) -> str:
    held = _ids(conn, "SELECT source_file_id FROM paper_files WHERE paper_id = ? ORDER BY source_file_id",
                (paper_id,))
    if source_file_id is None:
        if not held:
            raise ReassignError(f"paper {paper_id} has no PDF to move")
        return held[0]
    if source_file_id not in held:
        raise ReassignError(f"{source_file_id} is not a PDF of {paper_id}")
    return source_file_id


def cover_pages(lib: Library, source_file_id: str) -> list[str]:
    """The cover region of the PDF's text layer, else of its converted markdown (``[]``: no text)."""
    row = lib.conn.execute("SELECT storage_uri FROM source_files WHERE source_file_id = ?",
                           (source_file_id,)).fetchone()
    pages = first_pages_from_pdf(lib.root / row[0]) if row is not None and row[0] else None
    cover = cover_region(pages or [])
    if cover:
        return cover
    md = files.current_markdown_id(lib.conn, source_file_id)
    md_row = lib.conn.execute("SELECT storage_uri FROM markdown_documents WHERE markdown_id = ?",
                              (md,)).fetchone() if md else None
    if md_row is None or not (lib.root / md_row[0]).is_file():
        return []
    return cover_region(first_pages_from_markdown((lib.root / md_row[0]).read_text(encoding="utf-8")))


def suggest_targets(lib: Library, paper_id: str, source_file_id: str | None = None, *,
                    limit: int = SUGGEST_LIMIT) -> tuple[Suggestion, ...]:
    """Library papers whose title (or DOI) the PDF's cover shows, best first (module doc)."""
    conn = lib.conn
    if not paper_exists(conn, paper_id):
        raise ReassignError(f"paper {paper_id!r} is not in the library")
    sf = _source_file_of(conn, paper_id, source_file_id)
    prepared = prepare_pages(cover_pages(lib, sf))
    if prepared is None:
        return ()
    dois = dict(conn.execute("SELECT paper_id, MIN(id_value) FROM identifiers WHERE id_type = 'doi' "
                             "GROUP BY paper_id").fetchall())
    held = {r[0] for r in conn.execute("SELECT DISTINCT paper_id FROM paper_files")}
    rows = conn.execute("SELECT paper_id, title, year FROM papers WHERE paper_id <> ? "
                        "AND COALESCE(TRIM(title), '') <> '' ORDER BY paper_id", (paper_id,)).fetchall()
    found: list[Suggestion] = []
    for pid, title, year in rows:
        doi = dois.get(pid)
        if not prepared.may_pass(clean_title(title) or title, doi):
            continue  # cheap necessary condition (no outcome changes)
        if generic_title_reason(title) is not None:
            continue
        status = "held" if pid in held else "cited_only"
        check = check_prepared(clean_title(title) or title, doi, prepared)
        if check.status in ("pass", "weak") and check.score >= MISS_THRESHOLD:
            found.append(Suggestion(pid, title, year, status == "held", check.score, check.status, check.evidence))
    found.sort(key=lambda s: (-(s.score or 0.0), -_RANK[s.status or "unknown"], (s.title or "").casefold(),
                              s.paper_id))
    return tuple(found[:limit])


def find_papers(lib: Library, query: str, *, exclude: tuple[str, ...] | list[str] = (),
                limit: int = SEARCH_LIMIT) -> tuple[Suggestion, ...]:
    """Library papers whose title holds every word of ``query`` (or whose id is ``query``)."""
    words = [w for w in (query or "").split() if w]
    if not words:
        return ()
    where = " AND ".join("p.title LIKE ? ESCAPE '\\'" for _ in words)
    like = ["%" + w.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%" for w in words]
    rows = lib.conn.execute(
        f"SELECT p.paper_id, p.title, p.year, s.status FROM papers p JOIN paper_status s ON s.paper_id = p.paper_id "
        f"WHERE ({where}) OR p.paper_id = ? ORDER BY p.title COLLATE NOCASE, p.paper_id",
        [*like, query.strip()]).fetchall()
    skip = set(exclude)
    out = [Suggestion(pid, title, year, status == "held", None, None) for pid, title, year, status in rows
           if pid not in skip]
    return tuple(out[:limit])


# --- plan ----------------------------------------------------------------------------------------


def _cover_ids(lib: Library, sf: str) -> list[tuple[str, str]]:
    from ..work.tasks_metadata import deterministic_identifiers  # local: work imports the library

    found = deterministic_identifiers("\n".join(cover_pages(lib, sf)))
    return [(t, v) for t, v in normalize_ids([(k, v) for k, v in found.items() if v])]


def _graph_rows(lib: Library, paper_id: str) -> list[tuple[str, str, bool, str | None]]:
    out = []
    for info in graphs.list_graphs(lib):
        with graphs.open_graph(lib, info.slug) as g:
            row = g.conn.execute("SELECT status, is_seed, reason FROM graph_papers WHERE paper_id = ?",
                                 (paper_id,)).fetchone()
        if row is not None:
            out.append((info.slug, row[0], bool(row[1]), row[2]))
    return out


def _subject(text: str) -> dict:
    try:
        obj = json.loads(text) if text else None
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def plan_reassign_pdf(lib: Library, paper_id: str, source_file_id: str | None, *,
                      target_paper_id: str | None = None, new_paper: bool = False) -> ReassignPlan:
    """Describe moving ``source_file_id`` from ``paper_id`` without writing anything."""
    conn = lib.conn
    if (target_paper_id is None) == (not new_paper):
        raise ReassignError("name the paper the PDF belongs to, or ask for a new paper (one of the two)")
    if not paper_exists(conn, paper_id):
        raise ReassignError(f"paper {paper_id!r} is not in the library")
    if target_paper_id is not None:
        if target_paper_id == paper_id:
            raise ReassignError("the PDF is already on that paper")
        if not paper_exists(conn, target_paper_id):
            raise ReassignError(f"paper {target_paper_id!r} is not in the library")
    sf = _source_file_of(conn, paper_id, source_file_id)
    blockers: list[str] = []
    warnings: list[str] = []

    papers = {r[0]: r for r in _rows(conn, "SELECT paper_id, title, title_source, year FROM papers "
                                           "WHERE paper_id IN (?, ?)", (paper_id, target_paper_id or paper_id))}
    src_files = _rows(conn, "SELECT source_file_id, markdown_id, pdf_check_status FROM paper_files "
                            "WHERE paper_id = ? ORDER BY source_file_id", (paper_id,))
    tgt_files = _rows(conn, "SELECT source_file_id, markdown_id, pdf_check_status FROM paper_files "
                            "WHERE paper_id = ? ORDER BY source_file_id", (target_paper_id,)) if target_paper_id else []
    same = any(f[0] == sf for f in tgt_files)
    if tgt_files and not same:
        blockers.append(
            f"{target_paper_id} already has a different PDF ({tgt_files[0][0][:15]}…); a paper holds one PDF. "
            f"If the two are the same paper, merge them instead (Merge these papers, or seedgraph library merge "
            f"{target_paper_id} {paper_id}); otherwise detach its PDF first")

    # receiving paper id
    cover_ids: list[tuple[str, str]] = []
    if new_paper:
        taken = set(_ids(conn, "SELECT paper_id FROM papers"))
        receiving = mint_paper_id(derive_key(pdf_hashes=[sf.removeprefix("sf_")]), taken).paper_id
        for id_type, value in _cover_ids(lib, sf):
            owner = conn.execute("SELECT paper_id FROM identifiers WHERE id_type = ? AND id_value = ?",
                                 (id_type, value)).fetchone()
            if owner is None:
                cover_ids.append((id_type, value))
            else:
                warnings.append(f"the PDF's {id_type} {value} belongs to {owner[0]}: it may be that paper; "
                                f"move the PDF there instead if so")
    else:
        assert target_paper_id is not None
        receiving = target_paper_id

    # what moves: runs on this PDF's markdown, notes on it (or by those runs), claims, lens outputs
    mds = _ids(conn, "SELECT markdown_id FROM markdown_documents WHERE source_file_id = ? ORDER BY 1", (sf,))
    run_ids = _ids(conn, "SELECT extraction_run_id FROM extraction_runs WHERE paper_id = ? AND markdown_id IN "
                         "(SELECT value FROM json_each(?)) ORDER BY 1", (paper_id, _json(mds)))
    notes = _rows(conn, "SELECT note_id, schema_id, version_no, version_label, created_at, markdown_id, "
                        "extraction_run_id FROM structured_notes WHERE paper_id = ? AND (markdown_id IN "
                        "(SELECT value FROM json_each(?)) OR extraction_run_id IN (SELECT value FROM json_each(?))) "
                        "ORDER BY note_id", (paper_id, _json(mds), _json(run_ids)))
    note_ids = tuple(n[0] for n in notes)
    run_ids = tuple(sorted(set(run_ids) | {n[6] for n in notes}))
    claim_ids = _ids(conn, "SELECT claim_id FROM extracted_claims WHERE paper_id = ? AND (extraction_run_id IN "
                           "(SELECT value FROM json_each(?)) OR structured_note_id IN (SELECT value FROM json_each(?))) "
                           "ORDER BY 1", (paper_id, _json(run_ids), _json(note_ids)))
    lens_ids = _ids(conn, "SELECT lens_output_id FROM lens_outputs WHERE paper_id = ? AND (extraction_run_id IN "
                          "(SELECT value FROM json_each(?)) OR claim_id IN (SELECT value FROM json_each(?))) "
                          "ORDER BY 1", (paper_id, _json(run_ids), _json(claim_ids)))
    anchors = conn.execute("SELECT COUNT(*) FROM claim_spans WHERE claim_id IN (SELECT value FROM json_each(?))",
                           (_json(claim_ids),)).fetchone()[0]

    # receiving notes: its own versions and the moving ones, renumbered per schema by date
    own = _rows(conn, "SELECT note_id, schema_id, version_no, version_label, created_at, markdown_id, "
                      "extraction_run_id FROM structured_notes WHERE paper_id = ? ORDER BY note_id",
                (receiving,)) if not new_paper else []
    by_schema: dict[str | None, list[list]] = {}
    for n in [*own, *notes]:
        by_schema.setdefault(n[1], []).append(n)
    note_order: list[tuple[str, int]] = []
    for schema in sorted(by_schema, key=lambda s: (s is None, s or "")):
        for i, n in enumerate(sorted(by_schema[schema], key=lambda n: (n[4], n[0])), start=1):
            note_order.append((n[0], i))

    # researcher edits on the source pinned to what moves
    overlays = _rows(conn, "SELECT o.overlay_id, o.paper_id, o.schema_id, o.base_note_id, o.base_extraction_run_id, "
                           "o.head_version_no, v.ops_json FROM note_overlays o LEFT JOIN overlay_versions v ON "
                           "v.overlay_id = o.overlay_id AND v.version_no = o.head_version_no "
                           "WHERE o.paper_id IN (?, ?) ORDER BY o.overlay_id", (paper_id, receiving))
    for ov in overlays:
        if ov[1] != paper_id:
            continue
        if ov[3] in note_ids or ov[4] in run_ids or any(md in (ov[6] or "") for md in mds):
            blockers.append(f"researcher edits on {paper_id} are pinned to a note made on this PDF; resolve them "
                            f"(accept the extraction) or revert them first, then move the PDF")

    # work in flight
    marks = ",".join("?" * len(ACTIVE_BATCH_STATES))
    batch_rows = _rows(conn, f"SELECT batch_id, subject_id, state FROM batch_items WHERE subject_kind = 'paper' AND "
                             f"subject_id IN (?, ?) AND state IN ({marks}) ORDER BY batch_id, subject_id",
                       [paper_id, receiving, *ACTIVE_BATCH_STATES])
    for batch_id, pid, state in batch_rows:
        blockers.append(f"batch {batch_id} is still working on {pid} ({state}); let it finish or stop it first")

    # the source after the move, and its graphs
    becomes_cited_only = all(f[0] == sf for f in src_files)
    graph_rows = _graph_rows(lib, paper_id)
    changes = tuple(GraphChange(slug, status, seed, "frontier", reason) for slug, status, seed, reason in graph_rows
                    if status == "member" and becomes_cited_only)
    source_graphs = tuple(slug for slug, status, _, _ in graph_rows if status != "excluded")

    # attention: the source's items about this PDF, and a shared-PDF item that stops being shared
    holders = set(_ids(conn, "SELECT paper_id FROM paper_files WHERE source_file_id = ?", (sf,)))
    holders_after = (holders - {paper_id}) | {receiving}
    items = _rows(conn, "SELECT item_id, kind, paper_id, subject_json, status FROM attention_items "
                        "WHERE status = 'open' AND (paper_id = ? OR (paper_id IS NULL AND kind = 'shared_pdf')) "
                        "ORDER BY item_id", (paper_id,))
    resolutions: list[str] = []
    for item_id, kind, pid, subject_json, _status in items:
        subject = _subject(subject_json)
        names_file = subject.get("source_file_id") == sf or subject.get("markdown_id") in mds
        if kind == "shared_pdf":
            if subject.get("source_file_id") == sf and len(holders_after) < 2:
                resolutions.append(item_id)
        elif pid == paper_id and kind in PDF_KINDS and (
                names_file or (kind == "pdf_title_miss" and "source_file_id" not in subject and becomes_cited_only)):
            resolutions.append(item_id)

    refs_left = conn.execute("SELECT COUNT(*) FROM reference_entries WHERE citing_paper_id = ? AND markdown_id IN "
                             "(SELECT value FROM json_each(?))", (paper_id, _json(mds))).fetchone()[0]
    if refs_left:
        warnings.append(f"{refs_left} reference entries parsed from this PDF stay with {paper_id}")
    if becomes_cited_only:
        warnings.append(f"{paper_id} becomes cited-only: it needs its own PDF")
    if source_graphs:
        warnings.append(f"{receiving} is not added to any graph; use Add to graph on its page "
                        f"({paper_id} is in {', '.join(source_graphs)})")

    counts = {"notes": len(note_ids), "runs": len(run_ids), "claims": len(claim_ids), "anchors": anchors,
              "lens_outputs": len(lens_ids), "attention_resolved": len(resolutions), "graphs": len(changes),
              "references_left": refs_left}
    fingerprint = hashlib.sha256(json.dumps({
        "source": paper_id, "sf": sf, "target": target_paper_id, "new": new_paper, "receiving": receiving,
        "papers": sorted(papers.values()), "src_files": src_files, "tgt_files": tgt_files, "mds": mds,
        "runs": run_ids, "notes": notes, "own": own, "claims": claim_ids, "lens": lens_ids, "anchors": anchors,
        "overlays": overlays, "batches": batch_rows, "graphs": graph_rows, "holders": sorted(holders),
        "items": items, "cover_ids": cover_ids, "blockers": blockers,
    }, sort_keys=True, default=str).encode()).hexdigest()

    return ReassignPlan(
        paper_id=paper_id, source_file_id=sf, target_paper_id=target_paper_id, new_paper=new_paper,
        receiving_paper_id=receiving, source_title=papers[paper_id][1],
        target_title=papers[target_paper_id][1] if target_paper_id else None,
        target_was_cited_only=bool(target_paper_id) and not tgt_files, target_has_same_pdf=same,
        target_holds_other_pdf=bool(tgt_files) and not same,
        source_becomes_cited_only=becomes_cited_only, cover_identifiers=tuple(cover_ids), counts=counts,
        graphs=changes, source_graphs=source_graphs, blockers=tuple(blockers), warnings=tuple(warnings),
        fingerprint=fingerprint, note_order=tuple(note_order), run_ids=run_ids, note_ids=note_ids,
        claim_ids=claim_ids, lens_output_ids=lens_ids, item_resolutions=tuple(resolutions))


# --- apply ---------------------------------------------------------------------------------------


def reassign_pdf(lib: Library, plan: ReassignPlan, *, confirm: bool = True) -> ReassignResult:
    """Execute ``plan`` exactly; refuse a blocked plan or one whose rows changed since."""
    if not confirm:
        raise ReassignError("reassign_pdf needs confirm=True")
    if plan.blockers:
        raise ReassignError("moving the PDF is blocked: " + "; ".join(plan.blockers))
    src, sf, recv = plan.paper_id, plan.source_file_id, plan.receiving_paper_id
    try:
        fresh = plan_reassign_pdf(lib, src, sf, target_paper_id=plan.target_paper_id, new_paper=plan.new_paper)
    except ReassignError as exc:
        raise StalePlanError(f"the papers changed since the plan: {exc}") from exc
    if fresh.fingerprint != plan.fingerprint:
        raise StalePlanError("the papers changed since the plan was made; plan again")

    from ..notes.overlay import mark_concepts_out_of_date

    # graphs first: a source that becomes cited-only may not stay a member
    for change in plan.graphs:
        with graphs.open_graph(lib, change.slug) as g:
            graphs.set_membership(g, src, "frontier", is_seed=change.is_seed, reason=change.reason)

    now = _now()
    with lib.write() as conn:
        if plan.new_paper:
            conn.execute("INSERT INTO papers (paper_id, created_at) VALUES (?, ?)", (recv, now))
            for id_type, value in plan.cover_identifiers:
                conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value, source) "
                             "VALUES (?, ?, ?, 'pdf_scan')", (recv, id_type, value))
            attention.raise_item(lib, METADATA_NEEDED, {"paper_id": recv}, paper_id=recv,
                                 detail=f"made from a PDF that was on {src} but belongs to a paper not in the "
                                        f"library; its title is not known yet")
        if plan.target_has_same_pdf:
            conn.execute("DELETE FROM paper_files WHERE paper_id = ? AND source_file_id = ?", (src, sf))
        else:
            conn.execute("UPDATE paper_files SET paper_id = ?, pdf_check_status = 'unknown', pdf_check_json = NULL, "
                         "attached_at = ? WHERE paper_id = ? AND source_file_id = ?", (recv, now, src, sf))
        conn.execute("UPDATE extraction_runs SET paper_id = ? WHERE extraction_run_id IN "
                     "(SELECT value FROM json_each(?))", (recv, _json(plan.run_ids)))
        # notes (two passes: UNIQUE(paper_id, schema_id, version_no) is checked row by row)
        conn.executemany("UPDATE structured_notes SET paper_id = ?, version_no = ? WHERE note_id = ?",
                         [(recv, _TEMP_VERSION + v, nid) for nid, v in plan.note_order])
        conn.execute("UPDATE structured_notes SET version_no = version_no - ? WHERE paper_id = ? AND version_no > ?",
                     (_TEMP_VERSION, recv, _TEMP_VERSION))
        conn.execute("UPDATE extracted_claims SET paper_id = ? WHERE claim_id IN (SELECT value FROM json_each(?))",
                     (recv, _json(plan.claim_ids)))
        conn.execute("UPDATE lens_outputs SET paper_id = ? WHERE lens_output_id IN (SELECT value FROM json_each(?))",
                     (recv, _json(plan.lens_output_ids)))
        conn.executemany("UPDATE attention_items SET status = 'resolved', resolution = ?, resolved_at = ? "
                         "WHERE item_id = ? AND status = 'open'",
                         [(plan.resolution, now, i) for i in plan.item_resolutions])
        mark_concepts_out_of_date(conn, src, REFRESH_REASON)
        mark_concepts_out_of_date(conn, recv, REFRESH_REASON)

    # the receiving paper's title check (a miss raises a fresh pdf_title_miss item there)
    checks = files.recheck_pdf(lib, recv)
    status = checks[0].status if checks else None
    new_item = attention.item_id_for(files.MISS_KIND, {"paper_id": recv, "source_file_id": sf}) \
        if status == "miss" else None
    return ReassignResult(paper_id=src, source_file_id=sf, receiving_paper_id=recv, new_paper=plan.new_paper,
                          items_resolved=plan.item_resolutions, graphs_changed=tuple(c.slug for c in plan.graphs),
                          check=status, new_item_id=new_item)


def offer_merge(lib: Library, paper_id: str, target_paper_id: str) -> str | None:
    """The researcher says the PDF's paper and ``target_paper_id`` (which holds another PDF) are
    the same paper: raise an ``identity_check`` naming the pair, so the merge preview
    (Review, ``seedgraph.library.merge``) can take over. ``None`` when the pair was already
    recorded as different papers."""
    from .merge import decided_different

    for pid in (paper_id, target_paper_id):
        if not paper_exists(lib.conn, pid):
            raise ReassignError(f"paper {pid!r} is not in the library")
    if paper_id == target_paper_id:
        raise ReassignError("a paper cannot be merged with itself")
    if decided_different(lib.conn, paper_id, target_paper_id):
        return None
    return attention.raise_item(
        lib, "identity_check", {"paper_id": paper_id, "candidates": [target_paper_id]}, paper_id=paper_id,
        detail=f"the researcher says the PDF on {paper_id} is {target_paper_id}'s paper; merge them or mark "
               f"them as different papers").item_id


def plan_as_dict(plan: ReassignPlan) -> dict[str, Any]:
    """The researcher-facing parts of a plan (CLI ``--json``)."""
    return {
        "paper_id": plan.paper_id, "source_file_id": plan.source_file_id, "target_paper_id": plan.target_paper_id,
        "new_paper": plan.new_paper, "receiving_paper_id": plan.receiving_paper_id,
        "source_title": plan.source_title, "target_title": plan.target_title,
        "target_was_cited_only": plan.target_was_cited_only, "target_has_same_pdf": plan.target_has_same_pdf,
        "target_holds_other_pdf": plan.target_holds_other_pdf,
        "source_becomes_cited_only": plan.source_becomes_cited_only,
        "cover_identifiers": [list(i) for i in plan.cover_identifiers], "counts": plan.counts,
        "graphs": [c.__dict__ for c in plan.graphs], "source_graphs": list(plan.source_graphs),
        "blockers": list(plan.blockers), "warnings": list(plan.warnings), "fingerprint": plan.fingerprint,
    }
