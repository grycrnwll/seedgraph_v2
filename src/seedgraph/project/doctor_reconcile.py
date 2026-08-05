"""``work_refs_reconcile`` doctor check (Build A ch6) — soft work-ref + drift scan.

Registered via the established no-edit-to-doctor.py pattern (mirrors
``acquisition/doctor_reconcile.cross_db_bridge_reconcile``): ``cli.py``'s doctor
command appends these results when ``--project`` is given; phase_0's ``doctor.py``
is not edited.

Scans two things ``PRAGMA foreign_key_check`` can never see:

* **dangling soft work refs** — ``span_fts.work_id`` rows and OPEN ``review_queue``
  rows with ``target_type='work'`` whose work no longer exists (the FK-less half of
  ``merge.SOFT_REF_MANIFEST``; ``project_graph_edges`` is deliberately NOT
  duplicated here — the existing ``semantic_graph_dangling_edges`` check already
  scans every polymorphic endpoint of that table);
* **identifier normalization drift** — stored ``identifiers`` rows whose
  ``id_value`` differs from ``normalize_id(id_type, id_value)``. Go-forward-only
  normalization (Build A ch1/2) leaves pre-existing surface forms in place; this
  count is the measured residue for the deferred renormalization verb (a WARNING,
  never a gate failure).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .identity import normalize_id

if TYPE_CHECKING:
    from ..doctor import CheckResult


def work_refs_findings(conn) -> tuple[list[tuple[str, str]], int]:
    """Scan one project connection: ``(dangling, drift_count)``.

    ``dangling`` is ``[(table, work_id)]`` for soft work refs pointing at no
    ``works`` row; ``drift_count`` counts identifiers stored in a non-normalized
    surface form.
    """
    dangling: list[tuple[str, str]] = []
    for work_id, in conn.execute(
        "SELECT DISTINCT work_id FROM span_fts "
        "WHERE work_id IS NOT NULL "
        "AND work_id NOT IN (SELECT work_id FROM works)"
    ).fetchall():
        dangling.append(("span_fts", work_id))
    for target_id, in conn.execute(
        "SELECT DISTINCT target_id FROM review_queue "
        "WHERE target_type = 'work' AND status = 'open' "
        "AND target_id NOT IN (SELECT work_id FROM works)"
    ).fetchall():
        dangling.append(("review_queue", target_id))

    drift = 0
    for id_type, id_value in conn.execute(
        "SELECT id_type, id_value FROM identifiers"
    ).fetchall():
        if normalize_id(id_type, id_value) != id_value:
            drift += 1
    return dangling, drift


def work_refs_reconcile(
    root: Path | str | None = None,
    slug: str | None = None,
) -> "list[CheckResult]":
    """Doctor-check wrapper: map :func:`work_refs_findings` to ``CheckResult`` rows."""
    from ..doctor import CheckResult

    if slug is None:
        return [
            CheckResult(
                "work_refs_reconcile",
                True,
                "no --project given; work-ref reconcile skipped (no-op)",
                severity="warning",
            )
        ]

    try:
        from ..db.connection import open_project_db

        conn = open_project_db(slug, root)
        try:
            dangling, drift = work_refs_findings(conn)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - a reconcile error is a check failure
        return [CheckResult("work_refs_reconcile", False, f"scan failed: {exc}")]

    results = [
        CheckResult(
            "work_refs_reconcile",
            not dangling,
            "no dangling soft work refs (span_fts / open review_queue targets)"
            if not dangling
            else f"{len(dangling)} dangling soft work ref(s): {dangling[:5]}",
        ),
        CheckResult(
            "identifier_normalization_drift",
            True,
            "no identifier normalization drift"
            if not drift
            else f"{drift} identifier(s) stored in a non-normalized surface form "
            f"(go-forward normalization residue; renormalization verb deferred)",
            severity="warning",
        ),
    ]
    return results
