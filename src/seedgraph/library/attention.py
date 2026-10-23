"""Attention items: things the researcher should look at (v3 design §2.1, §4.3, §7.3).

An item is identified by a hash of its ``kind`` and its ``subject`` (a JSON
object), so raising the same thing twice is one row. An **open** item with
``hides_notes = 1`` hides its paper's notes and claims from agent-facing search
(§2.1 "Hidden from agents"); :func:`hidden_paper_ids` / :data:`HIDDEN_PAPERS_SQL`
are the single definition of that rule, shared by search and parity.

Rules chosen where the design is silent:
- Raising an item that already exists keeps its ``created_at`` and ``status``. It
  refreshes ``detail``, ``paper_id`` and ``hides_notes``. A **resolved** item is
  never reopened by a re-raise: the researcher's decision stands.
- Resolving an already-resolved item is a no-op that returns the stored row.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from .store import Library, LibraryError

ITEM_PREFIX = "att_"
STATUSES: tuple[str, ...] = ("open", "resolved")

Status = Literal["open", "resolved"]

#: Paper ids whose notes/claims are hidden from agents by an open attention item.
HIDDEN_PAPERS_SQL = (
    "SELECT paper_id FROM attention_items "
    "WHERE status = 'open' AND hides_notes = 1 AND paper_id IS NOT NULL"
)

_COLUMNS = (
    "item_id, kind, paper_id, subject_json, detail, hides_notes, status, resolution, "
    "created_at, resolved_at, origin"
)


class AttentionError(LibraryError):
    """An attention operation was refused."""


@dataclass(frozen=True)
class AttentionItem:
    item_id: str
    kind: str
    paper_id: str | None
    subject_json: str
    detail: str | None
    hides_notes: bool
    status: str
    resolution: str | None
    created_at: str
    resolved_at: str | None
    origin: str

    @property
    def subject(self) -> Any:
        return json.loads(self.subject_json)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_subject(subject: Any) -> str:
    """The canonical JSON text of a subject (sorted keys, fixed separators)."""
    return json.dumps(subject, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def item_id_for(kind: str, subject: Any) -> str:
    """``'att_' + sha256(kind + '\\n' + canonical subject)[:20]``."""
    if not kind:
        raise AttentionError("attention kind must not be empty")
    digest = hashlib.sha256(f"{kind}\n{canonical_subject(subject)}".encode()).hexdigest()
    return ITEM_PREFIX + digest[:20]


def _row_to_item(row: sqlite3.Row | tuple) -> AttentionItem:
    return AttentionItem(
        item_id=row[0], kind=row[1], paper_id=row[2], subject_json=row[3], detail=row[4],
        hides_notes=bool(row[5]), status=row[6], resolution=row[7], created_at=row[8],
        resolved_at=row[9], origin=row[10],
    )


def _fetch(conn: sqlite3.Connection, item_id: str) -> AttentionItem | None:
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM attention_items WHERE item_id = ?", (item_id,)
    ).fetchone()
    return None if row is None else _row_to_item(row)


def get_item(lib: Library, item_id: str) -> AttentionItem | None:
    return _fetch(lib.conn, item_id)


def raise_item(
    lib: Library,
    kind: str,
    subject: Any,
    *,
    paper_id: str | None = None,
    detail: str | None = None,
    hides_notes: bool = False,
    created_at: str | None = None,
    origin: str = "library",
) -> AttentionItem:
    """Create the item, or refresh it if it exists (idempotent upsert; see module doc)."""
    item_id = item_id_for(kind, subject)
    with lib.write() as conn:
        conn.execute(
            "INSERT INTO attention_items (item_id, kind, paper_id, subject_json, detail, "
            "hides_notes, status, created_at, origin) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?) "
            "ON CONFLICT(item_id) DO UPDATE SET detail = excluded.detail, "
            "paper_id = excluded.paper_id, hides_notes = excluded.hides_notes",
            (item_id, kind, paper_id, canonical_subject(subject), detail, int(hides_notes),
             created_at or _now(), origin),
        )
        item = _fetch(conn, item_id)
    assert item is not None
    return item


def resolve_item(
    lib: Library, item_id: str, resolution: str, *, resolved_at: str | None = None
) -> AttentionItem:
    """Mark an item resolved (un-hides its paper's notes). Unknown ids are refused."""
    with lib.write() as conn:
        item = _fetch(conn, item_id)
        if item is None:
            raise AttentionError(f"no attention item {item_id!r}")
        if item.status == "resolved":
            return item
        conn.execute(
            "UPDATE attention_items SET status = 'resolved', resolution = ?, resolved_at = ? "
            "WHERE item_id = ?",
            (resolution, resolved_at or _now(), item_id),
        )
        item = _fetch(conn, item_id)
    assert item is not None
    return item


def list_items(
    lib: Library,
    *,
    status: Status | None = None,
    kind: str | None = None,
    paper_id: str | None = None,
) -> list[AttentionItem]:
    """Items ordered by ``kind``, then ``paper_id`` (items without one first), then ``item_id``."""
    if status is not None and status not in STATUSES:
        raise AttentionError(f"invalid status {status!r}; expected one of {STATUSES}")
    where: list[str] = []
    args: list[str] = []
    for col, val in (("status", status), ("kind", kind), ("paper_id", paper_id)):
        if val is not None:
            where.append(f"{col} = ?")
            args.append(val)
    sql = f"SELECT {_COLUMNS} FROM attention_items"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY kind, COALESCE(paper_id, ''), item_id"
    return [_row_to_item(r) for r in lib.conn.execute(sql, args)]


def hidden_paper_ids(conn: sqlite3.Connection) -> frozenset[str]:
    """Papers whose notes and claims agents must not see (open ``hides_notes`` items)."""
    return frozenset(r[0] for r in conn.execute(HIDDEN_PAPERS_SQL))
