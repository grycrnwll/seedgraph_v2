"""Consolidated cross-DB reconcile (decision r2-10) — the gap-3 + gap-13 closure (§4.5).

One hash-anchored ``ok | stale | missing`` primitive over the ``work_source_files``
bridge, plus ``PRAGMA foreign_key_check`` on both scopes. Exposed as the
``cross_db_bridge_reconcile`` doctor check (surfaced by the ``seedgraph doctor``
CLI; phase_0's ``doctor.py`` is not edited by this phase). Until phase_3 lands, this
module is the canonical owner of :func:`open_cache_ro` /
:func:`current_markdown_for_source`, which phase_3's span reconcile reuses (one
primitive, many call sites).

Classification per bridge row (by CONTENT HASH, never by cache row id):
* ``ok``      — ``file_hash`` present in ``cache.db source_files`` AND
                ``markdown_hash`` is the newest markdown for that source lineage.
* ``stale``   — ``markdown_hash`` present but a newer markdown exists (reconversion).
* ``missing`` — ``file_hash`` / ``markdown_hash`` absent from ``cache.db`` (GC'd) ->
                fail-closed; the work falls back to ``metadata_only`` on read.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from sqlmodel import Session, select

from ..cache.db import init_cache_db
from ..db.connection import cache_db_path
from ..db.project_models import WorkSourceFile

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from ..doctor import CheckResult


@dataclass
class ReconcileFinding:
    """Per-bridge-row classification anchored on CONTENT HASH (§4.5)."""

    work_id: str
    status: str  # 'ok' | 'stale' | 'missing'
    file_hash: str
    markdown_hash: Optional[str] = None
    detail: str = ""


def open_cache_ro(cache_root: Path | str | None) -> sqlite3.Connection:
    """Connect to ``cache.db`` read-only for the bridge reconcile (§4.5).

    ``cache_root`` is the seedgraph HOME root (the same override every other
    function takes). The DB is migrated first (idempotent) so the tables exist, then
    re-opened read-only. Shared primitive: phase_3's span reconcile reuses this exact
    opener (decision r2-10 — one primitive, many call sites).
    """
    init_cache_db(cache_root)
    db_path = cache_db_path(cache_root)
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def current_markdown_for_source(
    conn: sqlite3.Connection,
    *,
    file_hash: str,
) -> "tuple[str | None, str | None, str]":
    """Newest successful conversion identity for a source lineage (§4.5).

    The identity may remain after its Markdown row/blob is pruned; callers verify
    availability separately. Never select an older conversion just because its
    bytes survive. Shared blob access pointers do not define producer lineage.
    Returns ``(markdown_id, markdown_hash, status)`` where ``status`` is ``'ok'``
    (a successful conversion identity exists) or ``'none'``; newest is by
    ``created_at`` with ``conversion_run_id`` as the deterministic tie-break.
    """
    source_file_id = f"sf_{file_hash}"
    row = conn.execute(
        "SELECT c.markdown_hash FROM conversion_runs c "
        "JOIN source_files s ON s.source_file_id=c.source_file_id "
        "WHERE c.source_file_id=? AND c.source_file_hash=? AND s.file_hash=? "
        "AND c.run_status='success' AND c.markdown_hash IS NOT NULL "
        "ORDER BY c.created_at DESC, c.conversion_run_id DESC LIMIT 1",
        (source_file_id, file_hash, file_hash),
    ).fetchone()
    if row is None:
        return (None, None, "none")
    # The content-addressed row can be shared by many producing sources. Its
    # source pointer is access provenance, not the exclusive production link.
    markdown_hash = row["markdown_hash"]
    return (f"md_{markdown_hash}", markdown_hash, "ok")


def bridge_row_status(
    cache_root: Path | str | None,
    *,
    file_hash: str,
    markdown_hash: Optional[str],
) -> str:
    """Classify ONE bridge row ``ok | stale | missing`` by content hash (§4.5).

    Used by ``derive_acquisition_state`` and :func:`reconcile_bridge`. Opens its own
    read-only ``cache.db`` connection.
    """
    conn = open_cache_ro(cache_root)
    try:
        return _classify(conn, file_hash=file_hash, markdown_hash=markdown_hash)
    finally:
        conn.close()


def _classify(conn: sqlite3.Connection, *, file_hash: str, markdown_hash: Optional[str]) -> str:
    src = conn.execute(
        "SELECT 1 FROM source_files WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    if src is None:
        return "missing"
    if markdown_hash is None:
        return "ok"  # source present, markdown not yet backfilled
    md = conn.execute(
        "SELECT 1 FROM markdown_documents WHERE markdown_hash = ?", (markdown_hash,)
    ).fetchone()
    if md is None:
        return "missing"
    _id, current_hash, status = current_markdown_for_source(conn, file_hash=file_hash)
    if status == "ok" and current_hash is not None and current_hash != markdown_hash:
        return "stale"
    return "ok"


def reconcile_bridge(project_engine: "Engine", cache_root: Path | str | None) -> list[ReconcileFinding]:
    """Classify every ``work_source_files`` row ``ok | stale | missing`` by content
    hash (§4.5). The same query phase_3's span reconcile uses (keyed on its anchor)."""
    with Session(project_engine) as session:
        rows = list(session.exec(select(WorkSourceFile)).all())

    findings: list[ReconcileFinding] = []
    conn = open_cache_ro(cache_root)
    try:
        for row in rows:
            status = _classify(conn, file_hash=row.file_hash, markdown_hash=row.markdown_hash)
            findings.append(
                ReconcileFinding(
                    work_id=row.work_id,
                    status=status,
                    file_hash=row.file_hash,
                    markdown_hash=row.markdown_hash,
                    detail=f"{row.acquisition_method}",
                )
            )
    finally:
        conn.close()
    return findings


def foreign_key_check_all(project_engine: "Engine", cache_root: Path | str | None) -> list:
    """``PRAGMA foreign_key_check`` on BOTH scopes; return any violations (r2-10)."""
    violations: list = []
    raw = project_engine.raw_connection()
    try:
        dbapi = getattr(raw, "dbapi_connection", None) or getattr(raw, "driver_connection", None) or raw
        for v in dbapi.execute("PRAGMA foreign_key_check").fetchall():
            violations.append(("project",) + tuple(v))
    finally:
        raw.close()

    conn = open_cache_ro(cache_root)
    try:
        for v in conn.execute("PRAGMA foreign_key_check").fetchall():
            violations.append(("cache",) + tuple(v))
    finally:
        conn.close()
    return violations


def cross_db_bridge_reconcile(
    root: Path | str | None = None,
    slug: str | None = None,
) -> "list[CheckResult]":
    """Doctor-check wrapper (decision r2-10): run :func:`reconcile_bridge` +
    :func:`foreign_key_check_all` and map results to ``doctor.CheckResult`` rows.

    Surfaced by the ``seedgraph doctor`` CLI (cli.py) — extends the doctor surface
    without editing phase_0's ``doctor.py``.
    """
    from ..doctor import CheckResult

    if slug is None:
        return [
            CheckResult(
                "cross_db_bridge_reconcile",
                True,
                "no --project given; bridge reconcile skipped (no-op)",
                severity="warning",
            )
        ]

    try:
        from ..project.service import open_project

        h = open_project(slug, root=root)
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("cross_db_bridge_reconcile", False, f"cannot open project: {exc}")]

    findings = reconcile_bridge(h.engine, root)
    ok = sum(1 for f in findings if f.status == "ok")
    stale = [f for f in findings if f.status == "stale"]
    missing = [f for f in findings if f.status == "missing"]

    results: list[CheckResult] = []
    if missing:
        results.append(
            CheckResult(
                "cross_db_bridge_reconcile",
                False,
                f"{len(missing)} bridge row(s) MISSING cache artifact "
                f"(work_ids: {[f.work_id for f in missing][:5]}); {ok} ok, {len(stale)} stale",
            )
        )
    elif stale:
        results.append(
            CheckResult(
                "cross_db_bridge_reconcile",
                True,
                f"{len(stale)} bridge row(s) stale (reconversion); {ok} ok — re-acquire to re-link",
                severity="warning",
            )
        )
    else:
        results.append(
            CheckResult(
                "cross_db_bridge_reconcile", True,
                f"{ok} bridge row(s) ok ({len(findings)} total)",
            )
        )

    fk = foreign_key_check_all(h.engine, root)
    results.append(
        CheckResult(
            "cross_db_foreign_key_check",
            not fk,
            "no cross-scope foreign-key violations" if not fk
            else f"{len(fk)} foreign-key violation(s): {fk[:5]}",
        )
    )
    return results
