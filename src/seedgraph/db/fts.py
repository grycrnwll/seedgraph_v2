"""FTS5 maintenance for the default note (plan §4.5/§5/§10 step 3).

The ``claim_fts`` / ``note_fts`` virtual tables are AUTHORED by the numbered
migration ``schema/project/0007_notes.sql`` (D6). The helpers here are idempotent
guards + application-managed reindex (delete-by-id + insert) run inside the SAME
write transaction as the note write — there are NO FTS triggers (plan §4.5/§7).
``span_fts`` is owned by phase_3 and is not touched here.

All three helpers reach the DBAPI connection bound to the caller's live ORM
transaction via :func:`seedgraph.db.adapter.raw_conn`, so the reindex commits (or
rolls back) atomically with the note/claim writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .adapter import raw_conn

if TYPE_CHECKING:
    from sqlmodel import Session


def ensure_fts_tables(session: "Session") -> None:
    """Idempotently ensure ``claim_fts`` and ``note_fts`` exist (guard/reindex helper).

    A belt-and-suspenders ``CREATE VIRTUAL TABLE IF NOT EXISTS`` guard matching the
    migration DDL — NOT the schema author (D6: the ``.sql`` migration owns the
    schema). Safe to call before a reindex on a freshly migrated db.
    """
    conn = raw_conn(session)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS claim_fts USING fts5("
        "claim_id UNINDEXED, normalized_label, claim_text, "
        'tokenize = "unicode61 remove_diacritics 2")'
    )
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5("
        "note_id UNINDEXED, note_text, "
        'tokenize = "unicode61 remove_diacritics 2")'
    )


def reindex_claim_fts(session: "Session", note_id: str) -> None:
    """Re-sync ``claim_fts`` for every claim of ``note_id`` (delete-by-id + insert).

    Runs in the caller's open transaction (no autocommit, no triggers). Inserts
    ``(claim_id, normalized_label, claim_text)`` for each claim of the note after
    deleting any stale rows for those claim ids.
    """
    conn = raw_conn(session)
    rows = conn.execute(
        "SELECT claim_id, normalized_label, claim_text "
        "FROM extracted_claims WHERE structured_note_id = ?",
        (note_id,),
    ).fetchall()
    for claim_id, normalized_label, claim_text in rows:
        conn.execute("DELETE FROM claim_fts WHERE claim_id = ?", (claim_id,))
        conn.execute(
            "INSERT INTO claim_fts(claim_id, normalized_label, claim_text) "
            "VALUES (?, ?, ?)",
            (claim_id, normalized_label, claim_text),
        )


def reindex_note_fts(session: "Session", note_id: str) -> None:
    """Re-sync ``note_fts`` for ``note_id`` (delete-by-id + insert ``note_text``).

    On producing a new current note the prior note's ``note_fts`` row is deleted
    and the new one inserted within the same transaction (plan §7).
    """
    conn = raw_conn(session)
    row = conn.execute(
        "SELECT note_text FROM structured_notes WHERE note_id = ?", (note_id,)
    ).fetchone()
    conn.execute("DELETE FROM note_fts WHERE note_id = ?", (note_id,))
    if row is not None:
        conn.execute(
            "INSERT INTO note_fts(note_id, note_text) VALUES (?, ?)",
            (note_id, row[0]),
        )
