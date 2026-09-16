"""Read-only access to cache.db + markdown bytes (cross-scope, no copy).

Phase 3 — Evidence Spans. cache.db is opened strictly **read-only** (ATTACH with
``mode=ro``); spans/FTS are written only to project.db (CONTENT_ACCESS_POLICY §2;
doc 04 §13). This module resolves, by ``markdown_id``:

* the markdown **text/bytes** (from the file at ``cache.markdown_documents.storage_uri``,
  relative to the cache root, decision 17),
* the content anchor ``markdown_hash`` and lineage ``source_file_id`` /
  ``source_file_hash`` / ``access_class`` (via
  ``markdown_id -> source_file_id -> cache.source_files.access_class``),
* and the *current* markdown for a source lineage, for staleness detection.

Under decision D1 the cache ids are content-addressed (``markdown_id == "md_" +
sha256(bytes)``), so re-hashing the bytes on read and asserting equality with the
stored ``markdown_hash`` (== id minus prefix) is a cheap integrity guard.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from seedgraph.acquisition import doctor_reconcile as _dr
from seedgraph.cache import store as _store
from seedgraph.ids import sha256_hex

# Reuse phase_5b's ONE hash-anchored cache primitive (decision r2-10): the
# read-only opener is identical, so phase_3 never re-implements the ATTACH-ro /
# staleness logic.
open_cache_ro = _dr.open_cache_ro


@dataclass(frozen=True)
class MarkdownRow:
    """Resolved cache row for one markdown document (read-only view).

    ``text`` is the decoded markdown string (UTF-8); offsets elsewhere index it as
    Python ``str`` code points. The remaining fields are denormalized onto spans /
    sections for rebuild-safe joins and fail-closed access stamping.
    """

    text: str
    markdown_hash: str
    storage_uri: str
    source_file_id: str
    source_file_hash: str
    access_class: str


def read_markdown(
    cache_conn: sqlite3.Connection, cache_root: Path | str | None, markdown_id: str
) -> MarkdownRow | None:
    """Resolve ``markdown_id`` to a :class:`MarkdownRow`, or ``None`` if missing.

    Reads ``cache.markdown_documents`` for ``storage_uri`` / ``markdown_hash`` /
    ``source_file_id``, joins ``cache.source_files`` for ``source_file_hash`` /
    ``access_class``, loads the markdown bytes from ``cache_root / storage_uri``,
    decodes UTF-8, and (cheap integrity guard) re-hashes the bytes and asserts they
    equal the stored ``markdown_hash`` (and ``markdown_id`` minus the ``md_`` prefix).
    Returns ``None`` when the ``markdown_id`` row or its bytes are absent (cache
    pruned) — callers degrade rather than crash.

    ``cache_root`` is the seedgraph HOME root (the same ``--root`` override every
    other accessor takes; ``None`` => ``$SEEDGRAPH_HOME``); the ``storage_uri`` is
    resolved relative to ``home/cache`` via :func:`seedgraph.cache.store.read_uri`.
    """
    row = cache_conn.execute(
        "SELECT m.markdown_hash AS markdown_hash, m.storage_uri AS storage_uri, "
        "       m.source_file_id AS source_file_id, "
        "       s.file_hash AS source_file_hash, s.access_class AS access_class "
        "FROM markdown_documents m "
        "JOIN source_files s ON s.source_file_id = m.source_file_id "
        "WHERE m.markdown_id = ?",
        (markdown_id,),
    ).fetchone()
    if row is None:
        return None
    markdown_hash = row["markdown_hash"]
    storage_uri = row["storage_uri"]
    try:
        raw = _store.read_uri(storage_uri, cache_root)
    except OSError:
        # Cache blob pruned/moved — degrade rather than crash (callers treat None
        # as "markdown unresolvable").
        return None
    text = raw.decode("utf-8")

    # Cheap integrity guard (D1): the markdown_id IS the content hash, so the bytes
    # must re-hash to the stored markdown_hash (== markdown_id minus the 'md_' prefix).
    actual = sha256_hex(raw)
    if actual != markdown_hash or f"md_{actual}" != markdown_id:
        raise ValueError(
            f"cache integrity violation for {markdown_id!r}: re-hashed bytes "
            f"{actual!r} != stored markdown_hash {markdown_hash!r}"
        )

    return MarkdownRow(
        text=text,
        markdown_hash=markdown_hash,
        storage_uri=storage_uri,
        source_file_id=row["source_file_id"],
        source_file_hash=row["source_file_hash"],
        access_class=row["access_class"],
    )


def proven_markdown_source(
    cache_conn: sqlite3.Connection, source_file_id: str,
    source_file_hash: str, markdown_hash: str,
) -> bool:
    """Prove producer lineage independently of the shared blob's access pointer.

    Distinct PDFs can produce identical Markdown. The single markdown row keeps
    a most-restrictive access representative; successful conversion_runs retain
    every actual source-to-output association.
    """
    return cache_conn.execute(
        "SELECT 1 FROM conversion_runs c JOIN source_files s "
        "ON s.source_file_id=c.source_file_id "
        "WHERE c.source_file_id=? AND c.source_file_hash=? AND s.file_hash=? "
        "AND c.markdown_hash=? AND c.run_status='success' LIMIT 1",
        (source_file_id, source_file_hash, source_file_hash, markdown_hash),
    ).fetchone() is not None


def current_markdown_for_source(
    cache_conn: sqlite3.Connection, source_file_id: str, source_file_hash: str
) -> tuple[str, str] | None:
    """Return the newest ``(markdown_id, markdown_hash)`` for a source lineage.

    Keyed on the denormalized lineage (``source_file_id`` + ``source_file_hash``) so
    "a newer markdown exists for this source" stays detectable even after the old
    ``markdown_id`` row is GC'd (decisions 18/32/42; doc 03 §12). Returns ``None``
    when no successful conversion remains for this source. The returned identity
    may outlive its Markdown row or blob: callers must use ``read_markdown`` to
    check availability, and must not silently fall back to older text. Backs lazy staleness in
    ``spans verify`` / ``reanchor`` / ``doctor`` / ``index``.

    Thin wrapper over phase_5b's :func:`doctor_reconcile.current_markdown_for_source`
    (decision r2-10 — one staleness primitive, many call sites). Under D1
    ``source_file_id == "sf_" + source_file_hash``, so the lineage key is the
    ``source_file_hash``.
    """
    markdown_id, markdown_hash, status = _dr.current_markdown_for_source(
        cache_conn, file_hash=source_file_hash
    )
    if status != "ok" or markdown_id is None or markdown_hash is None:
        return None
    return (markdown_id, markdown_hash)
