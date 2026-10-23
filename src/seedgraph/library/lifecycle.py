"""Paper lifecycle: remove from a graph, plan a delete, delete exactly the plan (v3 design §5).

- :func:`remove_from_graph` drops one ``graph_papers`` row. Nothing in the library
  changes.
- :func:`plan_delete` is pure: it names every graph still using the paper, counts
  what would go, and carries a fingerprint of the exact rows it would delete.
- :func:`delete_paper` re-plans, refuses if the fingerprint changed
  (:class:`StalePlanError`), then removes the paper from every graph and deletes its
  held parts in one library transaction. Blobs are unlinked only after the commit.

What a delete removes: the paper's ``paper_files`` links, extraction runs, notes
(every version, including researcher-edited ones; stage 3 overlays will join this
list), claims, quote anchors, lens outputs, its reference list (with the citation
links and review decisions that hang off it) and the matching FTS rows. A source
file, with its markdown, sections and spans, goes only when no other paper uses it
(another ``paper_files`` row on it or on its markdown, or another paper's claim
anchored in its spans).

The ``papers`` row and its identifiers stay, as a cited-only paper, when another
paper cites it (an incoming ``citation_links`` row or another paper's reference
resolved to it). Otherwise they go too, with the paper's attention items.

Graph rows are removed before the library transaction: if the library write then
fails, the paper is left intact in no graph, rather than graphs pointing at a
deleted paper. Researcher-edited notes are deleted only here, never by
:func:`remove_from_graph`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import graphs
from .store import Library, LibraryError, paper_exists


class LifecycleError(LibraryError):
    """A lifecycle operation was refused."""


class StalePlanError(LifecycleError):
    """The library or graph usage changed since the delete plan was made."""


@dataclass(frozen=True)
class RemoveResult:
    slug: str
    paper_id: str
    removed: bool
    was_seed: bool
    status: str | None


@dataclass(frozen=True)
class GraphUsage:
    slug: str
    status: str
    is_seed: bool


@dataclass(frozen=True)
class SourceFileUse:
    source_file_id: str
    shared: bool


@dataclass(frozen=True)
class DeletePlan:
    """What :func:`delete_paper` would do. Built only by :func:`plan_delete`."""

    paper_id: str
    graphs: tuple[GraphUsage, ...]
    counts: dict[str, int]
    keeps_cited_only: bool
    has_researcher_edits: bool
    pdf_shared: bool
    source_files: tuple[SourceFileUse, ...]
    warnings: tuple[str, ...]
    fingerprint: str
    # the exact ids to delete (part of the fingerprint)
    note_ids: tuple[str, ...] = field(repr=False, default=())
    claim_ids: tuple[str, ...] = field(repr=False, default=())
    run_ids: tuple[str, ...] = field(repr=False, default=())
    anchor_ids: tuple[int, ...] = field(repr=False, default=())
    lens_output_ids: tuple[str, ...] = field(repr=False, default=())
    reference_ids: tuple[str, ...] = field(repr=False, default=())


@dataclass(frozen=True)
class DeleteResult:
    paper_id: str
    paper_row_deleted: bool
    graphs_removed: tuple[str, ...]
    source_files_deleted: tuple[str, ...]
    blobs_removed: tuple[str, ...]


# --- remove_from_graph --------------------------------------------------------------------------

def remove_from_graph(lib: Library, slug: str, paper_id: str) -> RemoveResult:
    """Delete the paper's row from graph ``slug``. Nothing in the library changes."""
    with graphs.open_graph(lib, slug) as g:
        row = graphs.remove_membership(g, paper_id)
    if row is None:
        return RemoveResult(slug, paper_id, removed=False, was_seed=False, status=None)
    return RemoveResult(slug, paper_id, removed=True, was_seed=row.is_seed, status=row.status)


# --- plan_delete --------------------------------------------------------------------------------

def _ids(conn: sqlite3.Connection, sql: str, args: tuple | list) -> tuple:
    return tuple(r[0] for r in conn.execute(sql, args))


def _json(values: tuple | list) -> str:
    return json.dumps(list(values))


def _graph_usage(lib: Library, paper_id: str) -> tuple[GraphUsage, ...]:
    out: list[GraphUsage] = []
    for info in graphs.list_graphs(lib):  # sorted by slug
        with graphs.open_graph(lib, info.slug) as g:
            row = g.conn.execute(
                "SELECT status, is_seed FROM graph_papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        if row is not None:
            out.append(GraphUsage(info.slug, row[0], bool(row[1])))
    return tuple(out)


def _source_file_uses(
    conn: sqlite3.Connection, paper_id: str, claim_ids: tuple[str, ...]
) -> tuple[SourceFileUse, ...]:
    uses: list[SourceFileUse] = []
    for sf in _ids(conn, "SELECT source_file_id FROM paper_files WHERE paper_id = ? "
                         "ORDER BY source_file_id", (paper_id,)):
        other_link = conn.execute(
            "SELECT 1 FROM paper_files WHERE paper_id <> ? AND (source_file_id = ? OR markdown_id IN "
            "(SELECT markdown_id FROM markdown_documents WHERE source_file_id = ?)) LIMIT 1",
            (paper_id, sf, sf),
        ).fetchone()
        other_anchor = conn.execute(
            "SELECT 1 FROM claim_spans cs JOIN evidence_spans e ON e.span_id = cs.span_id "
            "WHERE e.source_file_id = ? AND cs.claim_id NOT IN (SELECT value FROM json_each(?)) "
            "LIMIT 1",
            (sf, _json(claim_ids)),
        ).fetchone()
        uses.append(SourceFileUse(sf, shared=other_link is not None or other_anchor is not None))
    return tuple(uses)


def plan_delete(lib: Library, paper_id: str) -> DeletePlan:
    """Describe the delete of ``paper_id`` without writing anything."""
    conn = lib.conn
    if not paper_exists(conn, paper_id):
        raise LifecycleError(f"paper {paper_id!r} is not in the library")

    run_ids = _ids(conn, "SELECT extraction_run_id FROM extraction_runs WHERE paper_id = ? "
                         "ORDER BY extraction_run_id", (paper_id,))
    note_ids = _ids(
        conn,
        "SELECT note_id FROM structured_notes WHERE paper_id = ? "
        "OR extraction_run_id IN (SELECT value FROM json_each(?)) ORDER BY note_id",
        (paper_id, _json(run_ids)),
    )
    claim_ids = _ids(
        conn,
        "SELECT claim_id FROM extracted_claims WHERE paper_id = ? "
        "OR extraction_run_id IN (SELECT value FROM json_each(?)) "
        "OR structured_note_id IN (SELECT value FROM json_each(?)) ORDER BY claim_id",
        (paper_id, _json(run_ids), _json(note_ids)),
    )
    anchor_ids = _ids(
        conn,
        "SELECT id FROM claim_spans WHERE claim_id IN (SELECT value FROM json_each(?)) ORDER BY id",
        (_json(claim_ids),),
    )
    lens_output_ids = _ids(
        conn,
        "SELECT lens_output_id FROM lens_outputs WHERE paper_id = ? "
        "OR extraction_run_id IN (SELECT value FROM json_each(?)) "
        "OR claim_id IN (SELECT value FROM json_each(?)) ORDER BY lens_output_id",
        (paper_id, _json(run_ids), _json(claim_ids)),
    )
    reference_ids = _ids(conn, "SELECT reference_id FROM reference_entries WHERE citing_paper_id = ? "
                               "ORDER BY reference_id", (paper_id,))
    current_notes = conn.execute(
        "SELECT COUNT(*) FROM note_current WHERE paper_id = ?", (paper_id,)).fetchone()[0]
    outgoing = conn.execute(
        "SELECT COUNT(*) FROM citation_links WHERE source_paper_id = ? "
        "OR reference_id IN (SELECT value FROM json_each(?))",
        (paper_id, _json(reference_ids)),
    ).fetchone()[0]
    incoming = conn.execute(
        "SELECT COUNT(*) FROM citation_links WHERE target_paper_id = ? AND source_paper_id <> ?",
        (paper_id, paper_id),
    ).fetchone()[0]
    cited_by_reference = conn.execute(
        "SELECT 1 FROM reference_entries WHERE resolved_paper_id = ? AND citing_paper_id <> ? LIMIT 1",
        (paper_id, paper_id),
    ).fetchone()
    keeps_cited_only = incoming > 0 or cited_by_reference is not None

    usage = _graph_usage(lib, paper_id)
    files = _source_file_uses(conn, paper_id, claim_ids)
    pdf_shared = any(f.shared for f in files)
    has_researcher_edits = False  # stage 3: note_overlays

    counts = {
        "notes": len(note_ids), "current_notes": current_notes, "claims": len(claim_ids),
        "anchors": len(anchor_ids), "lens_outputs": len(lens_output_ids),
        "references": len(reference_ids), "outgoing_links": outgoing, "incoming_links": incoming,
    }
    warnings: list[str] = []
    if usage:
        warnings.append("used by graphs " + ", ".join(u.slug for u in usage)
                        + "; the paper will be removed from them")
    if has_researcher_edits:
        warnings.append("has researcher edits, which will be deleted")
    if pdf_shared:
        warnings.append("its PDF is shared with another paper and will be kept")
    if keeps_cited_only:
        warnings.append("cited by other papers; it stays in the library as a cited-only paper")

    fingerprint = hashlib.sha256(json.dumps({
        "paper_id": paper_id,
        "graphs": [[u.slug, u.status, u.is_seed] for u in usage],
        "files": [[f.source_file_id, f.shared] for f in files],
        "notes": note_ids, "claims": claim_ids, "runs": run_ids, "anchors": anchor_ids,
        "lens_outputs": lens_output_ids, "references": reference_ids,
        "counts": counts, "keeps_cited_only": keeps_cited_only,
        "has_researcher_edits": has_researcher_edits,
    }, sort_keys=True).encode()).hexdigest()

    return DeletePlan(
        paper_id=paper_id, graphs=usage, counts=counts, keeps_cited_only=keeps_cited_only,
        has_researcher_edits=has_researcher_edits, pdf_shared=pdf_shared, source_files=files,
        warnings=tuple(warnings), fingerprint=fingerprint, note_ids=note_ids,
        claim_ids=claim_ids, run_ids=run_ids, anchor_ids=anchor_ids,
        lens_output_ids=lens_output_ids, reference_ids=reference_ids,
    )


# --- delete_paper -------------------------------------------------------------------------------

def _delete_in(conn: sqlite3.Connection, table: str, column: str, ids: tuple | list) -> None:
    if ids:
        conn.execute(
            f"DELETE FROM {table} WHERE {column} IN (SELECT value FROM json_each(?))", (_json(ids),)
        )


def delete_paper(lib: Library, plan: DeletePlan, *, confirm: bool = True) -> DeleteResult:
    """Execute ``plan`` exactly; refuse if anything it covers changed since it was made."""
    if not confirm:
        raise LifecycleError("delete_paper needs confirm=True")
    pid = plan.paper_id
    try:
        fresh = plan_delete(lib, pid)
    except LifecycleError as exc:
        raise StalePlanError(f"paper {pid!r} changed since the plan: {exc}") from exc
    if fresh.fingerprint != plan.fingerprint:
        raise StalePlanError(f"paper {pid!r} changed since the plan was made; plan again")

    for usage in plan.graphs:
        remove_from_graph(lib, usage.slug, pid)

    unshared = tuple(f.source_file_id for f in plan.source_files if not f.shared)
    blobs: list[str] = []
    with lib.write() as conn:
        if unshared:
            blobs = list(_ids(
                conn,
                "SELECT storage_uri FROM source_files WHERE source_file_id IN "
                "(SELECT value FROM json_each(?)) AND storage_uri IS NOT NULL "
                "UNION SELECT storage_uri FROM markdown_documents WHERE source_file_id IN "
                "(SELECT value FROM json_each(?)) ORDER BY 1",
                (_json(unshared), _json(unshared)),
            ))
        # notes, claims, anchors, lens outputs (FK order) and their FTS rows
        _delete_in(conn, "claim_spans", "id", plan.anchor_ids)
        _delete_in(conn, "lens_outputs", "lens_output_id", plan.lens_output_ids)
        _delete_in(conn, "claim_fts", "claim_id", plan.claim_ids)
        _delete_in(conn, "extracted_claims", "claim_id", plan.claim_ids)
        _delete_in(conn, "note_fts", "note_id", plan.note_ids)
        _delete_in(conn, "structured_notes", "note_id", plan.note_ids)
        _delete_in(conn, "extraction_runs", "extraction_run_id", plan.run_ids)
        # its reference list and what hangs off it
        _delete_in(conn, "citation_review_decisions", "reference_id", plan.reference_ids)
        _delete_in(conn, "citation_links", "reference_id", plan.reference_ids)
        conn.execute("DELETE FROM citation_links WHERE source_paper_id = ?", (pid,))
        _delete_in(conn, "reference_entries", "reference_id", plan.reference_ids)
        # the file links, then files no other paper uses
        conn.execute("DELETE FROM paper_files WHERE paper_id = ?", (pid,))
        if unshared:
            span_ids = _ids(conn, "SELECT span_id FROM evidence_spans WHERE source_file_id IN "
                                  "(SELECT value FROM json_each(?))", (_json(unshared),))
            _delete_in(conn, "span_fts", "span_id", span_ids)
            _delete_in(conn, "evidence_spans", "span_id", span_ids)
            _delete_in(conn, "document_sections", "source_file_id", unshared)
            _delete_in(conn, "conversion_queue", "source_file_id", unshared)
            _delete_in(conn, "markdown_documents", "source_file_id", unshared)
            _delete_in(conn, "conversion_runs", "source_file_id", unshared)
            _delete_in(conn, "source_files", "source_file_id", unshared)
        if not plan.keeps_cited_only:
            conn.execute("DELETE FROM attention_items WHERE paper_id = ?", (pid,))
            conn.execute("DELETE FROM identifiers WHERE paper_id = ?", (pid,))
            conn.execute("DELETE FROM papers WHERE paper_id = ?", (pid,))

    removed: list[str] = []
    for rel in blobs:
        path = Path(lib.root) / rel
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(rel)
    return DeleteResult(
        paper_id=pid, paper_row_deleted=not plan.keeps_cited_only,
        graphs_removed=tuple(u.slug for u in plan.graphs), source_files_deleted=unshared,
        blobs_removed=tuple(removed),
    )
