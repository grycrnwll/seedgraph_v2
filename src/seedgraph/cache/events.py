"""``cache_events`` append-only audit writer.

Every cache mutation emits exactly one event row (plan §4 / §6 algorithm), so the
cache is fully auditable. The ``event_type`` vocab is: ``source_file_ingested``,
``source_file_deduped``, ``access_class_reconciled``, ``conversion_started``,
``conversion_succeeded``, ``conversion_failed``, ``conversion_reused``,
``markdown_written``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .models import CacheEvent


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_event(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    event_type: str,
    detail: dict | None = None,
) -> CacheEvent:
    """Append one ``cache_events`` row on ``conn``'s active transaction.

    ``object_type`` is one of ``source_file`` / ``conversion_run`` /
    ``markdown_document``; ``detail`` (if given) is serialized to canonical
    (sorted-key) JSON in ``detail_json``. ``created_at`` is ISO-8601 UTC. Returns
    the inserted :class:`CacheEvent`. The write participates in the caller's
    transaction (it is committed/rolled back with it).

    NOTE (deviation from the scaffold hint): the cache core runs on the
    foundation's RAW ``sqlite3`` connection (``open_cache_db`` — ``foreign_keys=ON``
    already applied, the same raw path the ``migrate`` runner uses), so this takes a
    ``sqlite3.Connection`` rather than an ORM ``Session``. The typed ORM door
    (``cache_session``) remains available for later phases.
    """
    detail_json = (
        json.dumps(detail, sort_keys=True, separators=(",", ":")) if detail is not None else None
    )
    created_at = _now_iso()
    cursor = conn.execute(
        "INSERT INTO cache_events (object_type, object_id, event_type, detail_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (object_type, object_id, event_type, detail_json, created_at),
    )
    return CacheEvent(
        cache_event_id=cursor.lastrowid,
        object_type=object_type,
        object_id=object_id,
        event_type=event_type,
        detail_json=detail_json,
        created_at=created_at,
    )
