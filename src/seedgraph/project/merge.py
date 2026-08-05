"""Work-merge executor (Build A ch6) — collapse a reviewed duplicate pair.

:func:`merge_works` ports v1 ``db.merge_papers`` semantics (empty-only survivor
scalar backfill, identifier remap, edge remap honoring the dedup UNIQUE with
collision/self-loop drops, victim delete) onto EVERY v2 table referencing
``works`` — in two explicit sections:

* the **FK remap manifest** (everything ``PRAGMA foreign_key_list`` can see) —
  covered by an invariant test that walks the PRAGMAs on a migrated DB, so a
  future FK into ``works`` fails the suite instead of silently dangling;
* the hand-maintained **soft-ref manifest** for references no PRAGMA walk can see
  by construction: ``span_fts.work_id`` (UNINDEXED FTS column), open
  ``review_queue`` work targets, and ``project_graph_edges`` polymorphic Work
  endpoints (FK-less on purpose — 0010). Schema reviews adding a new FK-less work
  reference MUST extend this manifest; the runtime net is doctor
  (``work_refs_reconcile`` + the existing ``semantic_graph_dangling_edges``).

Remap runs strictly BEFORE the victim delete: three tables (``identifiers``,
``project_documents``, ``work_source_files``) are ``ON DELETE CASCADE`` and the
cascade must never be the remap mechanism. No tombstone table — the resolved
``review_queue`` row is the merge audit trail (Design 6).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Session

from ..db.adapter import raw_conn
from ..db.project_models import Work
from ..errors import ValidationError
from . import identity

# --- FK remap manifest (PRAGMA-visible references into works) ----------------

#: Plain ``UPDATE t SET col=survivor WHERE col=victim`` remaps (no per-table policy).
PLAIN_FK_REMAPS: tuple[tuple[str, str], ...] = (
    ("identifiers", "work_id"),          # global UNIQUE(id_type,id_value): no collision possible
    ("reference_entries", "citing_work_id"),
    ("reference_entries", "resolved_work_id"),
    ("document_sections", "work_id"),
    ("evidence_spans", "work_id"),
    ("extraction_runs", "work_id"),
    ("structured_notes", "work_id"),
    ("extracted_claims", "work_id"),
    ("lens_outputs", "work_id"),
    ("claim_concepts", "work_id"),
)

#: FK references handled by a dedicated per-table step below (edge dedup,
#: bridge UNIQUE(work_id), membership PK). Still part of the coverage manifest.
POLICY_FK_REMAPS: tuple[tuple[str, str], ...] = (
    ("citation_edges", "source_work_id"),
    ("citation_edges", "target_work_id"),
    ("work_source_files", "work_id"),
    ("project_documents", "work_id"),
)

#: The complete FK coverage set the invariant test checks against the PRAGMA walk.
FK_REMAP_MANIFEST: tuple[tuple[str, str], ...] = PLAIN_FK_REMAPS + POLICY_FK_REMAPS

# --- soft-ref manifest (FK-less by construction; hand-maintained) -------------
# (table, description) — documentation + the doctor check's scan list.
SOFT_REF_MANIFEST: tuple[tuple[str, str], ...] = (
    ("span_fts", "work_id (UNINDEXED FTS column)"),
    ("review_queue", "target_id where target_type='work' (open rows retargeted)"),
    ("project_graph_edges", "source/target_node_id where node_type='Work'"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _victim_worklike(victim: Work) -> dict:
    """A WorkLike over the victim's scalars for ``identity.backfill`` (empty-only
    survivor fold — never overwrites a survivor value)."""
    return {
        "title": victim.canonical_title,
        "authors": victim.authors,
        "venue": victim.venue,
        "year": victim.year,
        "doi": victim.doi,
        "arxiv": victim.arxiv_id,
        "openalex": victim.openalex_id,
        "s2": victim.semantic_scholar_id,
        "ssrn": victim.ssrn_id,
    }


def _remap_citation_edges(conn, survivor_id: str, victim_id: str) -> None:
    """Remap both endpoints per row; drop rows that would collide on
    ``UNIQUE(source,target,edge_type,provenance,run_id)`` or become self-loops."""
    survivor_keys = {
        tuple(row)
        for row in conn.execute(
            "SELECT source_work_id, target_work_id, edge_type, provenance, run_id "
            "FROM citation_edges WHERE source_work_id = ? OR target_work_id = ?",
            (survivor_id, survivor_id),
        ).fetchall()
    }
    victim_rows = conn.execute(
        "SELECT edge_id, source_work_id, target_work_id, edge_type, provenance, run_id "
        "FROM citation_edges WHERE source_work_id = ? OR target_work_id = ?",
        (victim_id, victim_id),
    ).fetchall()
    for edge_id, src, tgt, edge_type, provenance, run_id in victim_rows:
        new_src = survivor_id if src == victim_id else src
        new_tgt = survivor_id if tgt == victim_id else tgt
        key = (new_src, new_tgt, edge_type, provenance, run_id)
        if new_src == new_tgt or key in survivor_keys:
            conn.execute("DELETE FROM citation_edges WHERE edge_id = ?", (edge_id,))
            continue
        conn.execute(
            "UPDATE citation_edges SET source_work_id = ?, target_work_id = ? "
            "WHERE edge_id = ?",
            (new_src, new_tgt, edge_id),
        )
        survivor_keys.add(key)


def _remap_graph_edges(conn, survivor_id: str, victim_id: str) -> None:
    """Soft-ref remap of ``project_graph_edges`` polymorphic Work endpoints.

    Load-bearing beyond the next ``concepts build``: the rebuild PRESERVES
    ``user_validated``/``user_supplied`` rows, so an unremapped human-validated
    edge would dangle permanently (skipped by graph_build, red in doctor's
    ``semantic_graph_dangling_edges``). Machine rows get the same remap — cheaper
    than special-casing ``epistemic_type`` and keeps doctor green between the
    merge and the next build. Collision/self-loop handling mirrors citation_edges
    for its ``UNIQUE(src_type, src_id, tgt_type, tgt_id, edge_type, run_id)``.
    """
    survivor_keys = {
        tuple(row)
        for row in conn.execute(
            "SELECT source_node_type, source_node_id, target_node_type, "
            "target_node_id, edge_type, run_id FROM project_graph_edges "
            "WHERE (source_node_type = 'Work' AND source_node_id = ?) "
            "   OR (target_node_type = 'Work' AND target_node_id = ?)",
            (survivor_id, survivor_id),
        ).fetchall()
    }
    victim_rows = conn.execute(
        "SELECT edge_id, source_node_type, source_node_id, target_node_type, "
        "target_node_id, edge_type, run_id FROM project_graph_edges "
        "WHERE (source_node_type = 'Work' AND source_node_id = ?) "
        "   OR (target_node_type = 'Work' AND target_node_id = ?)",
        (victim_id, victim_id),
    ).fetchall()
    for edge_id, st, sid, tt, tid, edge_type, run_id in victim_rows:
        new_sid = survivor_id if (st == "Work" and sid == victim_id) else sid
        new_tid = survivor_id if (tt == "Work" and tid == victim_id) else tid
        key = (st, new_sid, tt, new_tid, edge_type, run_id)
        if (st == tt and new_sid == new_tid) or key in survivor_keys:
            conn.execute("DELETE FROM project_graph_edges WHERE edge_id = ?", (edge_id,))
            continue
        conn.execute(
            "UPDATE project_graph_edges SET source_node_id = ?, target_node_id = ? "
            "WHERE edge_id = ?",
            (new_sid, new_tid, edge_id),
        )
        survivor_keys.add(key)


def _merge_bridge_and_membership(conn, survivor_id: str, victim_id: str) -> None:
    """``work_source_files`` (UNIQUE(work_id)) and ``project_documents`` (PK) policy."""
    # Bridge: survivor already has a row -> delete the victim's; else retarget.
    has_survivor_bridge = conn.execute(
        "SELECT 1 FROM work_source_files WHERE work_id = ?", (survivor_id,)
    ).fetchone()
    if has_survivor_bridge:
        conn.execute("DELETE FROM work_source_files WHERE work_id = ?", (victim_id,))
    else:
        conn.execute(
            "UPDATE work_source_files SET work_id = ?, updated_at = ? WHERE work_id = ?",
            (survivor_id, _now(), victim_id),
        )

    # Membership: keep the survivor's row, but adopt included + the victim's
    # access_status when the survivor was metadata_only and the victim included
    # (the merged work HAS the full text via the bridge step above).
    surv_doc = conn.execute(
        "SELECT inclusion_status FROM project_documents WHERE work_id = ?",
        (survivor_id,),
    ).fetchone()
    vict_doc = conn.execute(
        "SELECT inclusion_status, access_status FROM project_documents WHERE work_id = ?",
        (victim_id,),
    ).fetchone()
    if surv_doc is None and vict_doc is not None:
        conn.execute(
            "UPDATE project_documents SET work_id = ?, updated_at = ? WHERE work_id = ?",
            (survivor_id, _now(), victim_id),
        )
        return
    if surv_doc is not None and vict_doc is not None:
        if surv_doc[0] == "metadata_only" and vict_doc[0] == "included":
            conn.execute(
                "UPDATE project_documents SET inclusion_status = 'included', "
                "access_status = ?, updated_at = ? WHERE work_id = ?",
                (vict_doc[1], _now(), survivor_id),
            )
        conn.execute("DELETE FROM project_documents WHERE work_id = ?", (victim_id,))


def merge_works(session: Session, survivor_id: str, victim_id: str) -> Work:
    """Collapse ``victim_id`` into ``survivor_id`` on the caller's ``session``
    (same-transaction; the caller commits — decision 80). Returns the survivor.

    Guards: ``survivor_id == victim_id`` is a no-op; a missing survivor or victim
    raises :class:`ValidationError` — the executor never guesses (the real-world
    missing case is a STALE sibling review item naming an already-merged work;
    chunk 7 surfaces the refusal to the human instead of recording a merge that
    collapsed nothing).
    """
    survivor = session.get(Work, survivor_id)
    if survivor is None:
        raise ValidationError(f"merge_works: survivor work {survivor_id!r} not found")
    if survivor_id == victim_id:
        return survivor  # no-op
    victim = session.get(Work, victim_id)
    if victim is None:
        raise ValidationError(f"merge_works: victim work {victim_id!r} not found")

    # 1. Survivor scalar backfill (empty-only fold — never overwrites).
    identity.backfill(survivor, _victim_worklike(victim), victim.title_hash)
    session.add(survivor)
    session.flush()

    conn = raw_conn(session)

    # 2. Edge remap with collision/self-loop drops (per-table policy).
    _remap_citation_edges(conn, survivor_id, victim_id)

    # 3. Bridge + membership policy remaps (CASCADE tables — remapped explicitly,
    #    never left to the cascade).
    _merge_bridge_and_membership(conn, survivor_id, victim_id)

    # 4. Plain FK remaps (includes identifiers, the third CASCADE table).
    for table, column in PLAIN_FK_REMAPS:
        conn.execute(
            f"UPDATE {table} SET {column} = ? WHERE {column} = ?",  # noqa: S608 - manifest-driven identifiers, not user input
            (survivor_id, victim_id),
        )

    # 5. Soft refs (the FK-less manifest above).
    conn.execute(
        "UPDATE span_fts SET work_id = ? WHERE work_id = ?", (survivor_id, victim_id)
    )
    conn.execute(
        "UPDATE review_queue SET target_id = ? "
        "WHERE target_type = 'work' AND target_id = ? AND status = 'open'",
        (survivor_id, victim_id),
    )
    _remap_graph_edges(conn, survivor_id, victim_id)

    # 6. Victim delete LAST (all references already remapped; nothing cascades).
    session.delete(victim)
    session.flush()
    return survivor
