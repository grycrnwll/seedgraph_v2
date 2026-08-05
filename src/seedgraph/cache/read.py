"""Read APIs later phases consume (markdown resolution + access-class walk).

These are how ``project.db`` phases pull markdown bytes (read-only accessor,
decisions 2/24) and how every private-by-default check resolves the authoritative
``access_class`` (stored once on ``source_files``, decision 11).
"""

from __future__ import annotations

from pathlib import Path

from ..db.connection import open_cache_db
from ..vocab import AccessClass
from . import store
from .db import init_cache_db
from .models import MarkdownDocument


def resolve_markdown(
    *,
    markdown_hash: str | None = None,
    markdown_id: str | None = None,
    root: Path | str | None = None,
) -> "MarkdownDocument | None":
    """Resolve a ``markdown_documents`` row by id or hash (exactly one kwarg).

    ``markdown_id``: exact PK lookup, at most one row. ``markdown_hash``: returns
    the LATEST by ``created_at`` — and because the markdown PK IS the content hash
    (``markdown_id == 'md_' + markdown_hash``, D1), the hash is effectively 1:1
    with the id, so "latest" is the single byte-identical row. Returns ``None``
    when nothing matches.
    """
    if (markdown_id is None) == (markdown_hash is None):
        raise ValueError("resolve_markdown: provide exactly one of markdown_id or markdown_hash")
    init_cache_db(root)
    conn = open_cache_db(root)
    try:
        if markdown_id is not None:
            row = conn.execute(
                "SELECT * FROM markdown_documents WHERE markdown_id = ?", (markdown_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM markdown_documents WHERE markdown_hash = ? "
                "ORDER BY created_at DESC, markdown_id DESC LIMIT 1",
                (markdown_hash,),
            ).fetchone()
        return MarkdownDocument(**dict(row)) if row is not None else None
    finally:
        conn.close()


def markdown_text(md: "MarkdownDocument", root: Path | str | None = None) -> str:
    """Return the decoded markdown text for ``md`` (read its content-addressed
    ``.md`` blob via the store)."""
    return store.read_uri(md.storage_uri, root).decode("utf-8")


def markdown_path(md: "MarkdownDocument", root: Path | str | None = None) -> Path:
    """Return the absolute filesystem path of ``md``'s stored ``.md`` blob
    (resolve its relative ``storage_uri`` against the cache root)."""
    return store.resolve_uri(md.storage_uri, root)


def source_access_class(markdown_id: str, root: Path | str | None = None) -> AccessClass:
    """Walk ``markdown_documents.source_file_id -> source_files.access_class`` and
    return the authoritative :class:`AccessClass` for a markdown artifact
    (decision 11 — access is never denormalized onto markdown). Fail-closed to the
    most restrictive value (``user_supplied_private``) if the walk cannot resolve."""
    init_cache_db(root)
    conn = open_cache_db(root)
    try:
        row = conn.execute(
            "SELECT s.access_class FROM markdown_documents m "
            "JOIN source_files s ON m.source_file_id = s.source_file_id "
            "WHERE m.markdown_id = ?",
            (markdown_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row[0] is None:
        return AccessClass.user_supplied_private  # fail closed
    return AccessClass.most_restrictive(row[0])
