"""PDF/HTML ingest into the content-addressed cache (plan §6, doc 04 §7).

``ingest_file`` streams a SHA-256, dedups on ``source_files.file_hash`` (UNIQUE),
content-addresses the bytes into the store (copy-not-link, atomic), writes the
``source_files`` row, and emits the matching ``cache_events`` row. On a dedup hit
it reconciles ``access_class`` **most-restrictive / never-downgrade** (decision
11; see :func:`reconcile_access_class`) and writes NO second blob.

Content-addressed identity (D1): ``source_file_id == 'sf_' + file_hash`` — so
re-ingesting identical bytes always yields the same id. ``access_class`` defaults
fail-closed to ``user_supplied_private`` and is set to ``open_access`` only when
``acquisition_method == 'open_access_fetch'`` (plan §7).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..errors import ValidationError
from ..vocab import AccessClass, AcquisitionMethod, FileType
from . import store
from .db import init_cache_db
from .events import record_event
from .hashing import sha256_file
from .models import SourceFile

try:  # raw connection opener (foreign_keys=ON already applied by the foundation)
    from ..db.connection import open_cache_db
except ImportError:  # pragma: no cover - foundation is always present
    raise


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_file_type(path: Path) -> str:
    """Best-effort source content type by suffix, then magic bytes (PDF-centric)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return FileType.pdf.value
    if suffix in (".html", ".htm"):
        return FileType.html.value
    head = b""
    try:
        head = path.read_bytes()[:64]
    except OSError:
        head = b""
    if head.startswith(b"%PDF"):
        return FileType.pdf.value
    if b"<htm" in head.lower() or b"<!doctype html" in head.lower():
        return FileType.html.value
    return FileType.pdf.value


def _default_access_for_acquisition(acquisition_method: AcquisitionMethod) -> AccessClass:
    """Fail-closed default access class implied by the acquisition method (§7):
    ``open_access`` ONLY for ``open_access_fetch``; everything else private."""
    if acquisition_method == AcquisitionMethod.open_access_fetch:
        return AccessClass.open_access
    return AccessClass.user_supplied_private


def reconcile_access_class(stored: AccessClass, incoming: AccessClass) -> AccessClass:
    """Return the MORE restrictive of ``stored`` / ``incoming`` (never downgrade).

    Restrictiveness order (most -> least): ``user_supplied_private`` > ``unknown`` >
    ``licensed_future`` > ``metadata_only`` > ``open_access`` (decisions 11/76). A
    file first seen as private and later as open-access STAYS private. Used on a
    dedup hit before returning the existing row; if the result differs from
    ``stored`` the row is tightened (UPDATE + bump ``updated_at`` + emit
    ``access_class_reconciled``). Delegates to the authoritative ordering on
    :meth:`seedgraph.vocab.AccessClass.most_restrictive`.
    """
    return AccessClass.most_restrictive(stored, incoming)


def ingest_file(
    path: Path,
    *,
    access_class: AccessClass,
    acquisition_method: "AcquisitionMethod",
    root: Path | str | None = None,
) -> "SourceFile":
    """Hash, dedup, store, and record a source file; return its ``SourceFile`` row.

    Algorithm: stream SHA-256 -> ``SELECT … WHERE file_hash=?`` ->
    (hit) reconcile ``access_class`` most-restrictive + emit ``source_file_deduped``
    / ``access_class_reconciled``, write no second blob; (miss) content-address
    the bytes into the store (atomic copy), INSERT the ``source_files`` row with
    ``source_file_id = 'sf_' + file_hash`` (D1), emit ``source_file_ingested``.

    ``root`` (additive over the plan signature) lets the CLI thread its ``--root``
    override; ``None`` resolves the cache root from ``$SEEDGRAPH_HOME``/default.
    """
    src_path = Path(path)
    if not src_path.is_file():
        raise ValidationError(f"ingest_file: not a file: {src_path}")

    access_class = AccessClass(access_class)
    acquisition_method = AcquisitionMethod(acquisition_method)
    # Effective access class is the MOST restrictive of the caller's request and the
    # fail-closed default implied by the acquisition method (open_access requires
    # BOTH an explicit open_access request AND acquisition_method=open_access_fetch).
    incoming = AccessClass.most_restrictive(
        access_class, _default_access_for_acquisition(acquisition_method)
    )

    init_cache_db(root)
    file_hash = sha256_file(src_path)
    source_file_id = f"sf_{file_hash}"
    file_type = _detect_file_type(src_path)

    conn = open_cache_db(root)
    try:
        existing = conn.execute(
            "SELECT * FROM source_files WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        now = _now_iso()

        if existing is not None:
            # Dedup hit: NO second blob, NO new row. Reconcile access_class.
            src = SourceFile(**dict(existing))
            stored = AccessClass(src.access_class)
            reconciled = reconcile_access_class(stored, incoming)
            if reconciled != stored:
                conn.execute(
                    "UPDATE source_files SET access_class = ?, updated_at = ? "
                    "WHERE source_file_id = ?",
                    (reconciled.value, now, src.source_file_id),
                )
                record_event(
                    conn,
                    "source_file",
                    src.source_file_id,
                    "access_class_reconciled",
                    {"from": stored.value, "to": reconciled.value},
                )
                src.access_class = reconciled.value
                src.updated_at = now
            record_event(
                conn,
                "source_file",
                src.source_file_id,
                "source_file_deduped",
                {"file_hash": file_hash},
            )
            conn.commit()
            return src

        # Miss: content-address the bytes into the store (atomic copy), insert row.
        if file_type == FileType.pdf.value:
            dest = store.pdf_path(file_hash, root)
        elif file_type == FileType.html.value:
            dest = store.html_path(file_hash, root)
        else:  # pragma: no cover - _detect_file_type never returns anything else
            raise ValidationError(f"ingest_file: unsupported file type {file_type!r}")
        store.copy_file_atomic(src_path, dest)
        storage_uri = store.relative_uri(dest, root)
        byte_size = dest.stat().st_size

        conn.execute(
            "INSERT INTO source_files "
            "(source_file_id, file_hash, file_type, access_class, acquisition_method, "
            " storage_uri, byte_size, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_file_id,
                file_hash,
                file_type,
                incoming.value,
                acquisition_method.value,
                storage_uri,
                byte_size,
                now,
                now,
            ),
        )
        record_event(
            conn,
            "source_file",
            source_file_id,
            "source_file_ingested",
            {
                "file_hash": file_hash,
                "file_type": file_type,
                "access_class": incoming.value,
                "acquisition_method": acquisition_method.value,
            },
        )
        conn.commit()
        return SourceFile(
            source_file_id=source_file_id,
            file_hash=file_hash,
            file_type=file_type,
            access_class=incoming.value,
            acquisition_method=acquisition_method.value,
            storage_uri=storage_uri,
            byte_size=byte_size,
            created_at=now,
            updated_at=now,
        )
    finally:
        conn.close()
