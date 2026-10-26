"""Merge two library papers the researcher says are the same paper (duplicate review).

Two calls, as :mod:`seedgraph.library.lifecycle`'s delete:

- :func:`plan_merge` is pure. It names what moves from the **other** paper to the
  **kept** one (the researcher picks which id survives), every blocker, and carries a
  fingerprint of all the rows it covers.
- :func:`merge_papers` re-plans, refuses a stale plan (:class:`StalePlanError`) or a
  blocked one (:class:`MergeError`), then applies it in one library transaction and
  updates the context graphs after the commit.

Rules as implemented:

- **Identifiers** move. Two different values of one id type (two DOIs, two OpenAlex
  ids, ...) block the merge: precedence never resolves a conflict (design §4.2).
- **Metadata**: the kept paper's fields win; a field it lacks (NULL) is filled from the
  other paper (``title`` together with ``title_source``).
- **PDF**: a paper has at most one file (``paper_files.role`` is ``'primary'`` only), so
  one PDF stays primary: the kept paper's by default, the other's when the kept paper
  has none, or the one the researcher picks (``primary_file``). A *different* PDF on the
  other side is **detached**, not deleted: its ``source_files`` row, markdown, spans and
  blobs stay, and the decision's reason names it. The same PDF on both is one link.
- **Notes**: every version of both papers is kept under the surviving paper, renumbered
  per schema by ``(created_at, note_id)``. Versions made on the detached PDF's markdown
  are labelled ``other_markdown`` (as the migration does), so they are never current.
  If both papers have researcher edits (``note_overlays``) the merge is refused until
  one is resolved; a single overlay moves and stays pinned to its base note. An overlay
  whose base note was made on the PDF that would be detached also blocks (pick that PDF).
- **Runs, claims, lens outputs** move (``paper_id`` only; ids never change).
- **Graphs** (after the library commit): the union of both rows per graph, member
  beats frontier beats excluded, ``is_seed`` if either was a seed; the other paper's
  row goes.
- **Citations**: every link is re-pointed, then deduped on (source, target, provenance)
  keeping the highest confidence, ties to the smallest ``reference_id`` (§4.4); a link
  that becomes a self-citation is dropped. The other paper's reference list is appended
  to the kept one's; an entry duplicating a kept entry (same text after case/space
  folding, or resolved to the same paper) is dropped, its links and review decision
  moving to the kept entry. References resolved to the other paper are re-pointed.
- **Attention**: an identity item naming only these two papers is resolved ``same``;
  open items about a detached PDF (``pdf_title_miss``, ``shared_pdf``, ``text_*``,
  ``renote_offered``) and open items of a kind the kept paper already has open are
  moved and resolved ``merged``; everything else moves.
- **Work in flight** (a ``queued`` / ``in_progress`` batch item on either paper) blocks.
- **Audit**: a ``duplicate_decisions`` row (``verdict='same'``, ``decided_by='researcher'``),
  a ``v2_id_map`` row ``('library:merge', 'papers', other, 'papers', keep)`` that
  :func:`resolve_merged` follows, v2 works mapped to the other paper re-pointed, and
  ``paper_merge_members`` re-pointed. ``concept_refresh`` is set for the kept paper.
- FTS rows are keyed by claim / note / span id (no paper id), so they stay consistent;
  a moved overlay's interpretation rows are rebuilt. The text-quality cache (keyed by
  markdown) is untouched. Blobs are never deleted.

:func:`record_different` records the "different papers" verdict; ingest consults
:func:`decided_different` before raising an identity check.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import graphs
from .store import Library, LibraryError, paper_exists

#: ``concept_refresh.reason`` written by a merge.
REFRESH_REASON = "paper_merge"
#: ``v2_id_map.v2_project`` of a merge trace row (``v2_id`` = removed id, ``v3_id`` = survivor).
MERGE_TRACE_PROJECT = "library:merge"
IDENTITY_KINDS: tuple[str, ...] = ("identity_check", "identity_collision")
#: Attention kinds about a paper's PDF: resolved when that PDF is detached by a merge.
PDF_KINDS: frozenset[str] = frozenset({"pdf_title_miss", "shared_pdf", "text_garbled", "text_missing",
                                       "renote_offered"})
ACTIVE_BATCH_STATES: tuple[str, ...] = ("queued", "in_progress")
_STATUS_RANK = {"member": 3, "frontier": 2, "excluded": 1}

#: Every (table, column) that names a library paper and is re-pointed by a merge.
PAPER_REFERENCES: tuple[tuple[str, str], ...] = (
    ("identifiers", "paper_id"), ("paper_files", "paper_id"), ("extraction_runs", "paper_id"),
    ("structured_notes", "paper_id"), ("extracted_claims", "paper_id"),
    ("reference_entries", "citing_paper_id"), ("reference_entries", "resolved_paper_id"),
    ("citation_links", "source_paper_id"), ("citation_links", "target_paper_id"),
    ("lens_outputs", "paper_id"), ("attention_items", "paper_id"), ("paper_merge_members", "paper_id"),
    ("note_overlays", "paper_id"), ("concept_refresh", "paper_id"),
    ("interpretation_claims", "paper_id"), ("interpretation_notes", "paper_id"),
)
_FILL_FIELDS = ("authors_json", "year", "venue", "abstract", "oa_status")
_TEMP_VERSION = 1_000_000  # note renumbering goes through this offset (no existing version reaches it)


class MergeError(LibraryError):
    """A merge was refused (blocked plan, unknown paper, no confirm)."""


class StalePlanError(MergeError):
    """The library or graph usage changed since the merge plan was made."""


@dataclass(frozen=True)
class GraphChange:
    slug: str
    keep_status: str | None
    other_status: str | None
    status: str  # the kept paper's status after the merge
    is_seed: bool
    reason: str | None = None


@dataclass(frozen=True)
class MergePlan:
    """What :func:`merge_papers` would do. Built only by :func:`plan_merge`."""

    keep_id: str
    other_id: str
    keep_title: str | None
    other_title: str | None
    identifiers: tuple[tuple[str, str], ...]  # (id_type, id_value) moving to the kept paper
    primary_file: str | None
    detached_file: str | None
    counts: dict[str, int]
    graphs: tuple[GraphChange, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    fingerprint: str
    note_order: tuple[tuple[str, int], ...] = field(repr=False, default=())  # (note_id, new version_no)
    relabel: tuple[str, ...] = field(repr=False, default=())  # note ids -> 'other_markdown'
    duplicate_refs: tuple[tuple[str, str], ...] = field(repr=False, default=())  # (dropped, kept)
    ref_ordinals: tuple[tuple[str, int], ...] = field(repr=False, default=())  # other's kept refs
    links: tuple[tuple, ...] = field(repr=False, default=())  # final citation_links rows involving keep
    item_moves: tuple[str, ...] = field(repr=False, default=())
    item_resolutions: tuple[tuple[str, str], ...] = field(repr=False, default=())

    @property
    def ok(self) -> bool:
        return not self.blockers


@dataclass(frozen=True)
class MergeResult:
    keep_id: str
    other_id: str
    detached_file: str | None
    graphs_changed: tuple[str, ...]
    items_resolved: tuple[str, ...]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(conn: sqlite3.Connection, sql: str, args: tuple | list = ()) -> list[list]:
    return [list(r) for r in conn.execute(sql, args)]


def _fold(text: str | None) -> str:
    return " ".join((text or "").casefold().split())


def _ids_in(value: object) -> set[str]:
    """Paper ids in a ``candidates`` / ``papers`` list: plain ids or ``{"paper_id": ...}`` objects."""
    out: set[str] = set()
    for c in value if isinstance(value, list) else ():
        if isinstance(c, str):
            out.add(c)
        elif isinstance(c, dict) and isinstance(c.get("paper_id"), str):
            out.add(c["paper_id"])
    return out


def named_papers(paper_id: str | None, subject_json: str, detail: str | None = None) -> set[str]:
    """Every paper an attention item names: its ``paper_id``, the subject's paper keys and
    ``candidates`` (ingest's shape), and the evidence's ``candidates`` / ``papers`` (the
    migration puts identity candidates and shared-PDF sides in ``detail``)."""
    named = {paper_id} if paper_id else set()
    for text in (subject_json, detail):
        try:
            obj = json.loads(text) if text else None
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("paper_id", "paper_a", "paper_b", "other_paper_id"):
            if isinstance(obj.get(key), str):
                named.add(obj[key])
        named |= _ids_in(obj.get("candidates")) | _ids_in(obj.get("papers"))
    return named


# --- decisions -----------------------------------------------------------------------------------


def _pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise MergeError("a paper cannot be a duplicate of itself")
    return (a, b) if a < b else (b, a)


def _upsert_decision(conn: sqlite3.Connection, a: str, b: str, verdict: str, reason: str | None) -> None:
    pa, pb = _pair(a, b)
    conn.execute(
        "INSERT INTO duplicate_decisions (paper_a, paper_b, verdict, decided_by, reason, decided_at) "
        "VALUES (?, ?, ?, 'researcher', ?, ?) ON CONFLICT(paper_a, paper_b) DO UPDATE SET "
        "verdict = excluded.verdict, decided_by = excluded.decided_by, reason = excluded.reason, "
        "decided_at = excluded.decided_at, origin = 'library'",
        (pa, pb, verdict, reason, _now()))


def record_different(lib: Library, paper_a: str, paper_b: str, *, reason: str | None = None) -> None:
    """Record the researcher's "different papers" verdict for the pair (never merged by rule,
    never raised again as an identity check)."""
    with lib.write() as conn:
        _upsert_decision(conn, paper_a, paper_b, "different", reason)


def decided_different(conn: sqlite3.Connection, paper_a: str, paper_b: str) -> bool:
    if paper_a == paper_b:
        return False
    pa, pb = _pair(paper_a, paper_b)
    row = conn.execute("SELECT verdict FROM duplicate_decisions WHERE paper_a = ? AND paper_b = ?",
                       (pa, pb)).fetchone()
    return row is not None and row[0] == "different"


def resolve_merged(conn: sqlite3.Connection, paper_id: str) -> str:
    """The surviving paper id of a paper merged away (following merge chains), else ``paper_id``."""
    seen = {paper_id}
    current = paper_id
    while True:
        row = conn.execute("SELECT v3_id FROM v2_id_map WHERE v2_project = ? AND v2_table = 'papers' AND v2_id = ?",
                           (MERGE_TRACE_PROJECT, current)).fetchone()
        if row is None or row[0] in seen:
            return current
        current = row[0]
        seen.add(current)


def suggest_survivor(conn: sqlite3.Connection, a: str, b: str) -> tuple[str, str]:
    """``(keep, other)``: the held paper, then more strong ids, more notes, older, smaller id."""
    def key(pid: str) -> tuple:
        held = conn.execute("SELECT 1 FROM paper_files WHERE paper_id = ? LIMIT 1", (pid,)).fetchone() is not None
        strong = conn.execute("SELECT COUNT(*) FROM identifiers WHERE paper_id = ? AND id_type IN "
                              "('doi', 'openalex', 'arxiv')", (pid,)).fetchone()[0]
        notes = conn.execute("SELECT COUNT(*) FROM structured_notes WHERE paper_id = ?", (pid,)).fetchone()[0]
        created = conn.execute("SELECT created_at FROM papers WHERE paper_id = ?", (pid,)).fetchone()
        return (not held, -strong, -notes, created[0] if created else "", pid)
    keep, other = sorted((a, b), key=key)
    return keep, other


# --- plan_merge ----------------------------------------------------------------------------------


def _graph_rows(lib: Library, keep: str, other: str) -> list[tuple[str, list | None, list | None]]:
    out = []
    for info in graphs.list_graphs(lib):
        with graphs.open_graph(lib, info.slug) as g:
            rows = {r[0]: list(r)[1:] for r in g.conn.execute(
                "SELECT paper_id, status, is_seed, reason FROM graph_papers WHERE paper_id IN (?, ?)", (keep, other))}
        if rows:
            out.append((info.slug, rows.get(keep), rows.get(other)))
    return out


def _graph_changes(rows: list[tuple[str, list | None, list | None]]) -> tuple[GraphChange, ...]:
    out = []
    for slug, k, o in rows:
        if o is None:
            continue  # the kept paper's row is untouched
        present = [r for r in (k, o) if r is not None]
        best = max(present, key=lambda r: _STATUS_RANK[r[0]])
        reason = k[2] if k is not None and k[2] is not None else o[2]
        out.append(GraphChange(slug, k[0] if k else None, o[0], best[0], any(bool(r[1]) for r in present), reason))
    return tuple(out)


def _file_row(conn: sqlite3.Connection, pid: str) -> list | None:
    rows = _rows(conn, "SELECT source_file_id, markdown_id FROM paper_files WHERE paper_id = ? "
                       "ORDER BY role, source_file_id", (pid,))
    return rows[0] if rows else None


def plan_merge(lib: Library, keep_id: str, other_id: str, *, primary_file: str | None = None) -> MergePlan:
    """Describe merging ``other_id`` into ``keep_id`` without writing anything."""
    conn = lib.conn
    if keep_id == other_id:
        raise MergeError("a paper cannot be merged with itself")
    for pid in (keep_id, other_id):
        if not paper_exists(conn, pid):
            raise MergeError(f"paper {pid!r} is not in the library")
    both = (keep_id, other_id)
    blockers: list[str] = []
    warnings: list[str] = []

    papers = {r[0]: r for r in _rows(conn, "SELECT paper_id, title, title_source, authors_json, year, venue, "
                                           "abstract, oa_status FROM papers WHERE paper_id IN (?, ?)", both)}

    # identifiers
    ids = _rows(conn, "SELECT paper_id, id_type, id_value FROM identifiers WHERE paper_id IN (?, ?) "
                      "ORDER BY paper_id, id_type, id_value", both)
    keep_ids: dict[str, list[str]] = {}
    other_ids: dict[str, list[str]] = {}
    for pid, t, v in ids:
        (keep_ids if pid == keep_id else other_ids).setdefault(t, []).append(v)
    for t in sorted(other_ids):
        if t in keep_ids:
            blockers.append(f"both papers have a {t}: {', '.join(keep_ids[t])} ({keep_id}) and "
                            f"{', '.join(other_ids[t])} ({other_id}); two ids of one type cannot be merged")
    moving_ids = tuple((t, v) for pid, t, v in ids if pid == other_id)

    # files
    kf, of = _file_row(conn, keep_id), _file_row(conn, other_id)
    held = {f[0] for f in (kf, of) if f is not None}
    if primary_file is not None and primary_file not in held:
        raise MergeError(f"{primary_file!r} is not a PDF of {keep_id} or {other_id}")
    primary = primary_file or (kf[0] if kf else (of[0] if of else None))
    detached = next(iter(held - {primary}), None) if primary else None

    # notes: renumber per schema by (created_at, note_id); label the detached PDF's notes
    notes = _rows(conn, "SELECT note_id, paper_id, schema_id, version_no, version_label, created_at, markdown_id "
                        "FROM structured_notes WHERE paper_id IN (?, ?) ORDER BY note_id", both)
    detached_mds = {r[0] for r in conn.execute("SELECT markdown_id FROM markdown_documents WHERE source_file_id = ?",
                                               (detached,))} if detached else set()
    by_schema: dict[str | None, list[list]] = {}
    for n in notes:
        by_schema.setdefault(n[2], []).append(n)
    note_order: list[tuple[str, int]] = []
    for schema in sorted(by_schema, key=lambda s: (s is None, s or "")):
        for i, n in enumerate(sorted(by_schema[schema], key=lambda n: (n[5], n[0])), start=1):
            note_order.append((n[0], i))
    relabel = tuple(n[0] for n in notes if n[6] in detached_mds and n[4] is None)

    # researcher edits
    overlays = _rows(conn, "SELECT overlay_id, paper_id, schema_id, base_note_id, head_version_no FROM note_overlays "
                           "WHERE paper_id IN (?, ?) ORDER BY overlay_id", both)
    if {o[1] for o in overlays} == set(both):
        blockers.append("both papers have researcher edits; resolve one paper's edits first "
                        "(accept the extraction there), then merge")
    elif any(o[3] in relabel for o in overlays):
        blockers.append(f"researcher edits are pinned to a note made on {detached}, which this merge would detach; "
                        f"keep that PDF as the primary one")

    # work in flight
    marks = ",".join("?" * len(ACTIVE_BATCH_STATES))
    for batch_id, pid, state in _rows(
            conn, f"SELECT batch_id, subject_id, state FROM batch_items WHERE subject_kind = 'paper' AND "
                  f"subject_id IN (?, ?) AND state IN ({marks}) ORDER BY batch_id, subject_id",
            [*both, *ACTIVE_BATCH_STATES]):
        blockers.append(f"batch {batch_id} is still working on {pid} ({state}); let it finish or stop it first")

    # runs, claims, lens outputs
    runs = _rows(conn, "SELECT extraction_run_id FROM extraction_runs WHERE paper_id = ? ORDER BY 1", (other_id,))
    claims = _rows(conn, "SELECT claim_id FROM extracted_claims WHERE paper_id = ? ORDER BY 1", (other_id,))
    lens = _rows(conn, "SELECT lens_output_id FROM lens_outputs WHERE paper_id = ? ORDER BY 1", (other_id,))

    # references: append the other list, dropping duplicates of kept entries
    def remap(pid: str | None) -> str | None:
        return keep_id if pid == other_id else pid

    refs = _rows(conn, "SELECT reference_id, citing_paper_id, raw_reference_text, resolved_paper_id, ordinal "
                       "FROM reference_entries WHERE citing_paper_id IN (?, ?) "
                       "ORDER BY citing_paper_id, ordinal IS NULL, ordinal, reference_id", both)
    keep_refs = [r for r in refs if r[1] == keep_id]
    by_text = {_fold(r[2]): r[0] for r in keep_refs}
    by_target = {}
    for r in keep_refs:
        if r[3] is not None:
            by_target.setdefault(remap(r[3]), r[0])
    next_ordinal = max([r[4] or 0 for r in keep_refs] or [0]) + 1
    duplicate_refs: list[tuple[str, str]] = []
    ref_ordinals: list[tuple[str, int]] = []
    for r in refs:
        if r[1] != other_id:
            continue
        match = by_text.get(_fold(r[2])) or (by_target.get(remap(r[3])) if r[3] is not None else None)
        if match is not None:
            duplicate_refs.append((r[0], match))
        else:
            ref_ordinals.append((r[0], next_ordinal))
            next_ordinal += 1
    ref_map = dict(duplicate_refs)

    # citation links: re-point, drop self-links, dedupe on (source, target, provenance)
    link_rows = _rows(conn, "SELECT source_paper_id, target_paper_id, provenance, confidence, reference_id, "
                            "created_at, origin FROM citation_links WHERE source_paper_id IN (?, ?) "
                            "OR target_paper_id IN (?, ?) ORDER BY source_paper_id, target_paper_id, provenance",
                      [*both, *both])
    final: dict[tuple, list] = {}
    self_links = 0
    for s, t, prov, conf, rid, created, origin in link_rows:
        s2, t2, rid2 = remap(s), remap(t), ref_map.get(rid, rid)
        if s2 == t2:
            self_links += 1
            continue
        row = [s2, t2, prov, conf, rid2, created, origin]
        cur = final.get((s2, t2, prov))
        if cur is None or _wins(conf, rid2, cur[3], cur[4]):
            final[(s2, t2, prov)] = row
    links = tuple(tuple(final[k]) for k in sorted(final))

    # attention
    items = _rows(conn, "SELECT item_id, kind, paper_id, subject_json, status, detail FROM attention_items "
                        "WHERE paper_id IN (?, ?) ORDER BY item_id", both)
    pdf_owner = {kf[0]: keep_id} if kf else {}
    if of:
        pdf_owner.setdefault(of[0], other_id)
    detached_owner = pdf_owner.get(detached) if detached else None
    resolutions: list[tuple[str, str]] = []
    moves: list[str] = []
    keep_open_kinds = {i[1] for i in items if i[2] == keep_id and i[4] == "open"}
    for item_id, kind, pid, subject_json, status, detail in items:
        if pid == other_id:
            moves.append(item_id)
        if status != "open":
            continue
        if kind in IDENTITY_KINDS and set(both) <= named_papers(pid, subject_json, detail) <= set(both):
            resolutions.append((item_id, "same"))
        elif kind in PDF_KINDS and pid == detached_owner:
            resolutions.append((item_id, "merged"))
        elif pid == other_id and kind in keep_open_kinds:
            resolutions.append((item_id, "merged"))
    if kf and of and kf[0] == of[0]:
        # one PDF shared by exactly these two papers is no longer shared once they are one paper
        shared = _rows(conn, "SELECT item_id, paper_id, subject_json, detail FROM attention_items "
                             "WHERE kind = 'shared_pdf' AND status = 'open' ORDER BY item_id")
        for item_id, pid, subject_json, detail in shared:
            try:
                subject = json.loads(subject_json)
            except ValueError:
                continue
            if isinstance(subject, dict) and subject.get("source_file_id") == kf[0]                     and named_papers(pid, subject_json, detail) <= set(both)                     and item_id not in {i for i, _ in resolutions}:
                resolutions.append((item_id, "merged"))
                items.append([item_id, "shared_pdf", pid, subject_json, "open", detail])
    # graphs, prior decisions
    graph_rows = _graph_rows(lib, keep_id, other_id)
    changes = _graph_changes(graph_rows)
    pa, pb = _pair(keep_id, other_id)
    decision = conn.execute("SELECT verdict, decided_by, reason FROM duplicate_decisions "
                            "WHERE paper_a = ? AND paper_b = ?", (pa, pb)).fetchone()
    if decision is not None and decision[0] == "different":
        warnings.append(f"these papers were recorded as different ({decision[1]}); merging replaces that verdict")
    if detached:
        warnings.append(f"PDF {detached} will be detached (kept in the library, not deleted); "
                        f"notes made on it are labelled other_markdown")

    counts = {
        "identifiers": len(moving_ids), "notes": sum(1 for n in notes if n[1] == other_id),
        "relabelled_notes": len(relabel), "runs": len(runs), "claims": len(claims), "lens_outputs": len(lens),
        "references": len(ref_ordinals), "duplicate_references": len(duplicate_refs),
        "citation_links": len(link_rows), "self_links_dropped": self_links,
        "attention_moved": len(moves), "attention_resolved": len(resolutions), "graphs": len(changes),
        "overlays": sum(1 for o in overlays if o[1] == other_id),
    }
    fingerprint = hashlib.sha256(json.dumps({
        "pair": both, "papers": [papers[p] for p in both], "ids": ids, "files": [kf, of],
        "primary": primary, "detached": detached, "notes": notes, "overlays": overlays, "runs": runs,
        "claims": claims, "lens": lens, "refs": refs, "links": link_rows, "items": items,
        "graphs": graph_rows, "decision": list(decision) if decision else None, "blockers": blockers,
        "resolved_to_other": _rows(conn, "SELECT reference_id FROM reference_entries WHERE resolved_paper_id = ? "
                                         "ORDER BY 1", (other_id,)),
    }, sort_keys=True, default=str).encode()).hexdigest()

    return MergePlan(
        keep_id=keep_id, other_id=other_id, keep_title=papers[keep_id][1], other_title=papers[other_id][1],
        identifiers=moving_ids, primary_file=primary, detached_file=detached, counts=counts, graphs=changes,
        blockers=tuple(blockers), warnings=tuple(warnings), fingerprint=fingerprint,
        note_order=tuple(note_order), relabel=relabel, duplicate_refs=tuple(duplicate_refs),
        ref_ordinals=tuple(ref_ordinals), links=links, item_moves=tuple(moves),
        item_resolutions=tuple(resolutions),
    )


def _wins(conf: float, rid: str | None, cur_conf: float, cur_rid: str | None) -> bool:
    """Highest confidence wins; a tie goes to the smallest ``reference_id`` (NULL loses)."""
    if conf != cur_conf:
        return conf > cur_conf
    if rid is None or cur_rid is None:
        return rid is not None and cur_rid is None
    return rid < cur_rid


# --- merge_papers --------------------------------------------------------------------------------


def _decision_reason(plan: MergePlan) -> str:
    parts = [f"merged {plan.other_id} into {plan.keep_id}"]
    if plan.detached_file:
        parts.append(f"detached PDF {plan.detached_file}")
    if plan.primary_file:
        parts.append(f"primary PDF {plan.primary_file}")
    return "; ".join(parts)


def merge_papers(lib: Library, plan: MergePlan, *, confirm: bool = True) -> MergeResult:
    """Execute ``plan`` exactly; refuse a blocked plan or one whose rows changed since."""
    if not confirm:
        raise MergeError("merge_papers needs confirm=True")
    if plan.blockers:
        raise MergeError("the merge is blocked: " + "; ".join(plan.blockers))
    keep, other = plan.keep_id, plan.other_id
    try:
        fresh = plan_merge(lib, keep, other, primary_file=plan.primary_file)
    except MergeError as exc:
        raise StalePlanError(f"the papers changed since the merge plan: {exc}") from exc
    if fresh.fingerprint != plan.fingerprint:
        raise StalePlanError("the papers changed since the merge plan was made; plan again")

    from ..notes.overlay import mark_concepts_out_of_date, rebuild

    now = _now()
    with lib.write() as conn:
        # identifiers and metadata
        conn.execute("UPDATE identifiers SET paper_id = ? WHERE paper_id = ?", (keep, other))
        other_title = conn.execute("SELECT title, title_source FROM papers WHERE paper_id = ?", (other,)).fetchone()
        if other_title[0] is not None:
            conn.execute("UPDATE papers SET title = ?, title_source = ? WHERE paper_id = ? AND title IS NULL",
                         (other_title[0], other_title[1], keep))
        for col in _FILL_FIELDS:
            conn.execute(f"UPDATE papers SET {col} = (SELECT {col} FROM papers WHERE paper_id = ?) "
                         f"WHERE paper_id = ? AND {col} IS NULL", (other, keep))
        # the one primary file
        primary_row = None
        if plan.primary_file:
            primary_row = conn.execute(
                "SELECT source_file_id, markdown_id, role, pdf_check_status, pdf_check_json, attached_at, origin "
                "FROM paper_files WHERE source_file_id = ? AND paper_id IN (?, ?) "
                "ORDER BY paper_id = ? DESC LIMIT 1", (plan.primary_file, keep, other, keep)).fetchone()
        conn.execute("DELETE FROM paper_files WHERE paper_id IN (?, ?)", (keep, other))
        if primary_row is not None:
            conn.execute("INSERT INTO paper_files (paper_id, source_file_id, markdown_id, role, pdf_check_status, "
                         "pdf_check_json, attached_at, origin) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (keep, *tuple(primary_row)))
        # notes (two passes: UNIQUE(paper_id, schema_id, version_no) is checked row by row)
        conn.executemany("UPDATE structured_notes SET paper_id = ?, version_no = ? WHERE note_id = ?",
                         [(keep, _TEMP_VERSION + v, nid) for nid, v in plan.note_order])
        conn.execute("UPDATE structured_notes SET version_no = version_no - ? WHERE paper_id = ? AND version_no > ?",
                     (_TEMP_VERSION, keep, _TEMP_VERSION))
        conn.executemany("UPDATE structured_notes SET version_label = 'other_markdown' WHERE note_id = ?",
                         [(nid,) for nid in plan.relabel])
        for table in ("extraction_runs", "extracted_claims", "lens_outputs"):
            conn.execute(f"UPDATE {table} SET paper_id = ? WHERE paper_id = ?", (keep, other))
        # researcher edits
        moved_schemas = [r[0] for r in conn.execute("SELECT schema_id FROM note_overlays WHERE paper_id = ?",
                                                    (other,))]
        for table in ("note_overlays", "interpretation_claims", "interpretation_notes"):
            conn.execute(f"UPDATE {table} SET paper_id = ? WHERE paper_id = ?", (keep, other))
        for schema_id in moved_schemas:
            rebuild(conn, keep, schema_id)
        # citation links: replace every row touching either paper by the deduped set
        conn.execute("DELETE FROM citation_links WHERE source_paper_id IN (?, ?) OR target_paper_id IN (?, ?)",
                     (keep, other, keep, other))
        conn.executemany("INSERT INTO citation_links (source_paper_id, target_paper_id, provenance, confidence, "
                         "reference_id, created_at, origin) VALUES (?, ?, ?, ?, ?, ?, ?)", plan.links)
        # reference lists
        for dropped, kept in plan.duplicate_refs:
            has = conn.execute("SELECT 1 FROM citation_review_decisions WHERE reference_id = ?", (kept,)).fetchone()
            if has is None:
                conn.execute("UPDATE citation_review_decisions SET reference_id = ? WHERE reference_id = ?",
                             (kept, dropped))
            else:
                conn.execute("DELETE FROM citation_review_decisions WHERE reference_id = ?", (dropped,))
            conn.execute("UPDATE citation_links SET reference_id = ? WHERE reference_id = ?", (kept, dropped))
            conn.execute("DELETE FROM reference_entries WHERE reference_id = ?", (dropped,))
        conn.executemany("UPDATE reference_entries SET citing_paper_id = ?, ordinal = ? WHERE reference_id = ?",
                         [(keep, o, rid) for rid, o in plan.ref_ordinals])
        conn.execute("UPDATE reference_entries SET resolved_paper_id = ? WHERE resolved_paper_id = ?", (keep, other))
        # attention
        conn.executemany("UPDATE attention_items SET paper_id = ? WHERE item_id = ?",
                         [(keep, i) for i in plan.item_moves])
        conn.executemany("UPDATE attention_items SET status = 'resolved', resolution = ?, resolved_at = ? "
                         "WHERE item_id = ? AND status = 'open'",
                         [(res, now, i) for i, res in plan.item_resolutions])
        # traceability, concepts, the decision, then the paper row
        conn.execute("UPDATE paper_merge_members SET paper_id = ? WHERE paper_id = ?", (keep, other))
        conn.execute("UPDATE v2_id_map SET v3_id = ? WHERE v3_table = 'papers' AND v3_id = ?", (keep, other))
        conn.execute("INSERT INTO v2_id_map (v2_project, v2_table, v2_id, v3_table, v3_id) "
                     "VALUES (?, 'papers', ?, 'papers', ?) ON CONFLICT (v2_project, v2_table, v2_id) DO UPDATE SET v3_id = excluded.v3_id",
                     (MERGE_TRACE_PROJECT, other, keep))
        conn.execute("DELETE FROM concept_refresh WHERE paper_id = ?", (other,))
        mark_concepts_out_of_date(conn, keep, REFRESH_REASON)
        _upsert_decision(conn, keep, other, "same", _decision_reason(plan))
        conn.execute("DELETE FROM papers WHERE paper_id = ?", (other,))

    # graphs after the commit: a member needs the kept paper to be held, which it now is
    for change in plan.graphs:
        with graphs.open_graph(lib, change.slug) as g:
            graphs.set_membership(g, keep, change.status, is_seed=change.is_seed,  # type: ignore[arg-type]
                                  reason=change.reason)
            graphs.remove_membership(g, other)
    return MergeResult(keep_id=keep, other_id=other, detached_file=plan.detached_file,
                       graphs_changed=tuple(c.slug for c in plan.graphs),
                       items_resolved=tuple(i for i, _ in plan.item_resolutions))


def plan_as_dict(plan: MergePlan) -> dict[str, Any]:
    """The researcher-facing parts of a plan (CLI ``--json``)."""
    return {
        "keep_id": plan.keep_id, "other_id": plan.other_id, "keep_title": plan.keep_title,
        "other_title": plan.other_title, "identifiers": [list(i) for i in plan.identifiers],
        "primary_file": plan.primary_file, "detached_file": plan.detached_file, "counts": plan.counts,
        "graphs": [c.__dict__ for c in plan.graphs], "blockers": list(plan.blockers),
        "warnings": list(plan.warnings), "fingerprint": plan.fingerprint,
    }
