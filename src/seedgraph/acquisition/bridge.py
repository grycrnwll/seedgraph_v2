"""``work_source_files`` bridge — the load-bearing work->markdown link (§4.2/§6.3).

Implements decision 3 + must-fix #3 + NEW-A/NEW-B:

* :class:`WorkSourceFile` — re-exported from :mod:`seedgraph.db.project_models`
  (ORM MAPPING ONLY, mirrors ``db/schema/project/0003_acquisition.sql``; D6).
  **ONE row per work** (``UNIQUE(work_id)``); NO ``role`` / ``manifestation_kind``
  column (decision 1 deferral; NEW-A).
* :func:`write_bridge` — upsert ON ``work_id`` (newest acquisition wins; never a
  second competing row).
* :func:`backfill_markdown` — fill ``markdown_id`` / ``markdown_hash`` after the
  phase_1 conversion returns.
* :func:`resolve_work_markdown` — OWNED + DECLARED HERE (the bridge owner; NEW-B).
  Plain ``WHERE work_id = ?`` read, NO ``role='primary'`` filter; deterministic by
  ``UNIQUE(work_id)``. Imported by phase_3b and phase_6 FROM HERE (upstream of both
  so the DAG stays forward).
* :func:`derive_acquisition_state` — the doc 04 §6 status as a DERIVED display
  string from ``vocab.ACQUISITION_STATES`` (decision 1; NOT a stored column / enum).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlmodel import Session, select

from ..db.project_models import ProjectDocument, Work, WorkSourceFile
from ..vocab import ACQUISITION_STATES

#: OA colors that mean "an OA copy exists — a re-run or a bigger budget will get
#: it" (Build D ch10, D-10). A deliberate SUPERSET of gap scan §4.3's
#: gold/green/hybrid/bronze/closed list: ``diamond`` is a real OpenAlex
#: ``oa_status`` value (diamond OA journals) and belongs on the open side;
#: ``closed`` is the paywalled color and stays out.
_OA_OPEN_STATUSES: frozenset[str] = frozenset(
    {"gold", "green", "hybrid", "bronze", "diamond"}
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "WorkSourceFile",
    "WorkSourceResolution",
    "write_bridge",
    "backfill_markdown",
    "resolve_work_markdown",
    "resolve_work_source",
    "derive_acquisition_state",
]


@dataclass(frozen=True)
class WorkSourceResolution:
    """The full bridge row a work resolves to (Track 3 source-serving, review #4).

    The richer sibling of :func:`resolve_work_markdown`'s ``(markdown_id,
    markdown_hash)`` tuple: it additionally carries the **source-file** anchors
    (``source_file_id`` / ``file_hash``) the pdf/markdown endpoints need to look the
    file up in cache.db ``source_files`` / ``markdown_documents`` by ``storage_uri``.
    ``markdown_id`` / ``markdown_hash`` are ``None`` until conversion backfills them.
    """

    work_id: str
    source_file_id: str
    file_hash: str
    markdown_id: str | None
    markdown_hash: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_bridge(
    project_session: "Session",
    *,
    work_id: str,
    source_file_id: str,
    file_hash: str,
    markdown_id: str | None = None,
    markdown_hash: str | None = None,
    acquisition_method: str,
) -> None:
    """Upsert the single bridge row ON ``work_id`` (must-fix #3).

    Re-acquiring the same physical file (same ``source_file_id``) is a no-op
    update; a replacement acquisition (a different file for the same work) replaces
    the row in place (newest wins). Never produces a second competing row — a
    second INSERT for the same ``work_id`` raises ``IntegrityError`` via
    ``UNIQUE(work_id)``. ``acquisition_method`` is one of ``open_access_fetch`` /
    ``manual_upload`` / ``already_cached_local``. Flushes (the caller commits) so
    the write participates in the caller's transaction.
    """
    now = _now()
    existing = project_session.exec(
        select(WorkSourceFile).where(WorkSourceFile.work_id == work_id)
    ).first()
    if existing is not None:
        existing.source_file_id = source_file_id
        existing.file_hash = file_hash
        # Markdown columns are set only when supplied so a no-markdown re-acquire
        # (ingest before convert) never wipes an already-backfilled markdown.
        if markdown_id is not None:
            existing.markdown_id = markdown_id
        if markdown_hash is not None:
            existing.markdown_hash = markdown_hash
        existing.acquisition_method = acquisition_method
        existing.updated_at = now
        project_session.add(existing)
    else:
        project_session.add(
            WorkSourceFile(
                work_id=work_id,
                source_file_id=source_file_id,
                file_hash=file_hash,
                markdown_id=markdown_id,
                markdown_hash=markdown_hash,
                acquisition_method=acquisition_method,
                created_at=now,
                updated_at=now,
            )
        )
    project_session.flush()


def backfill_markdown(
    project_session: "Session",
    *,
    work_id: str,
    markdown_id: str,
    markdown_hash: str,
) -> None:
    """Fill ``markdown_id`` / ``markdown_hash`` on the work's bridge row after the
    phase_1 ``convert_source_file`` returns; bumps ``updated_at`` (§4.2)."""
    row = project_session.exec(
        select(WorkSourceFile).where(WorkSourceFile.work_id == work_id)
    ).first()
    if row is None:
        return
    row.markdown_id = markdown_id
    row.markdown_hash = markdown_hash
    row.updated_at = _now()
    project_session.add(row)
    project_session.flush()


def resolve_work_markdown(
    project_session: "Session",
    *,
    work_id: str,
) -> "tuple[str, str] | None":
    """Resolve a work to its current markdown — the gap-3 read accessor (NEW-B).

    Plain read, NO role filter::

        SELECT markdown_id, markdown_hash FROM work_source_files WHERE work_id = ?

    Returns ``(markdown_id, markdown_hash)`` or ``None`` (no bridge row, or markdown
    not yet backfilled). Deterministic by ``UNIQUE(work_id)`` (§4.2). OWNED +
    DECLARED HERE; phase_3b and phase_6 import it FROM HERE.
    """
    row = project_session.exec(
        select(WorkSourceFile).where(WorkSourceFile.work_id == work_id)
    ).first()
    if row is None or row.markdown_id is None or row.markdown_hash is None:
        return None
    return (row.markdown_id, row.markdown_hash)


def resolve_work_source(
    project_session: "Session",
    *,
    work_id: str,
) -> "WorkSourceResolution | None":
    """Resolve a work to its full bridge row (Track 3 source-serving; review #4).

    A NEW resolver added alongside — never replacing — :func:`resolve_work_markdown`
    (whose ``(markdown_id, markdown_hash)`` return arity ~12 callers unpack and which
    is left untouched). Plain ``WHERE work_id = ?`` read over ``work_source_files``;
    deterministic by ``UNIQUE(work_id)``. Returns a :class:`WorkSourceResolution`
    (``work_id``, ``source_file_id``, ``file_hash``, ``markdown_id``,
    ``markdown_hash``) or ``None`` when there is no bridge row. Unlike
    ``resolve_work_markdown`` it does NOT require markdown to be backfilled — a
    pdf-only (pre-conversion) work still resolves, with ``markdown_id`` /
    ``markdown_hash`` ``None``.
    """
    row = project_session.exec(
        select(WorkSourceFile).where(WorkSourceFile.work_id == work_id)
    ).first()
    if row is None:
        return None
    return WorkSourceResolution(
        work_id=row.work_id,
        source_file_id=row.source_file_id,
        file_hash=row.file_hash,
        markdown_id=row.markdown_id,
        markdown_hash=row.markdown_hash,
    )


def derive_acquisition_state(
    project_session: "Session",
    cache_root: "Path | str | None",
    work_id: str,
) -> str:
    """Return the doc 04 §6 acquisition status as a DERIVED display string.

    One of ``vocab.ACQUISITION_STATES``, computed from bridge-row presence +
    reconcile state + ``project_documents`` membership — NOT a stored column, NOT
    an enum (decision 1):

    * inclusion ``excluded``                       -> ``excluded``
    * a bridge row whose markdown reconciles ``ok`` -> ``already_cached_local``
    * a bridge row that no longer reconciles        -> ``failed``
    * no bridge row, ``works.oa_status`` open       -> ``available_open_access``
    * no bridge row, inclusion ``included``         -> ``requires_user_upload``
    * no bridge row, inclusion ``metadata_only``    -> ``metadata_only``

    The ``available_open_access`` branch (Build D ch10) activates the previously
    dead vocab state: an un-bridged, non-excluded work whose OA color is in
    :data:`_OA_OPEN_STATUSES` is "OA exists — re-run/budget will get it", NOT
    "genuinely paywalled" (``requires_user_upload``) or plain ``metadata_only``.
    Costs one extra ``session.get`` for the ``Work`` row on the no-bridge path.
    """
    doc = project_session.get(ProjectDocument, work_id)
    inclusion = doc.inclusion_status if doc is not None else "metadata_only"
    if inclusion == "excluded":
        return "excluded"

    bridge = project_session.exec(
        select(WorkSourceFile).where(WorkSourceFile.work_id == work_id)
    ).first()
    if bridge is not None and bridge.markdown_hash is not None:
        from .doctor_reconcile import bridge_row_status

        status = bridge_row_status(
            cache_root,
            file_hash=bridge.file_hash,
            markdown_hash=bridge.markdown_hash,
        )
        state = "already_cached_local" if status == "ok" else "failed"
        return state if state in ACQUISITION_STATES else "failed"

    # UN-bridged only (bridge is None): a bridged-but-unconverted work already
    # HAS its file — the OA color adds nothing there and the pre-ch10 states stand.
    if bridge is None:
        work = project_session.get(Work, work_id)
        if work is not None and (work.oa_status or "") in _OA_OPEN_STATUSES:
            return "available_open_access"

    if inclusion == "included":
        return "requires_user_upload"
    return "metadata_only"
