"""Lens calibration — sample run + FP/FN review report (plan §5/§10 step 8).

`run_calibration` runs the lens over a small sample work-set and assembles a
review report (per-record found/not-found, verbatim quote, span back-ref,
confidence) for false-positive / false-negative inspection, setting the lens
``status='calibrating'``.

The *revise loop* (docs/06 §7 step 6) needs no dedicated verb: the user edits the
YAML, ``registry.sync_lens`` refreshes the hash (prior sample runs go stale), and
``lens calibrate`` / ``lens run`` reprocesses the affected works — closing the
loop. Signatures are phase-private (not pinned in plan §6.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import results as results_mod
from .runner import LensRouter, run_lens

if TYPE_CHECKING:
    from sqlmodel import Session

    from .schema import LensDefinition


@dataclass
class CalibrationReport:
    """Assembled sample-run review surface for FP/FN inspection (plan §10 step 8)."""

    lens_id: str
    sample_work_ids: list[str] = field(default_factory=list)
    found_records: list[dict[str, Any]] = field(default_factory=list)
    not_found_work_ids: list[str] = field(default_factory=list)
    skipped_no_markdown: list[str] = field(default_factory=list)
    review_items: list[dict[str, Any]] = field(default_factory=list)


def run_calibration(
    session: "Session",
    cache_db: Path,
    lens: "LensDefinition",
    router: "LensRouter",
    *,
    sample: int = 3,
    work_ids: list[str] | None = None,
    force: bool = True,
) -> CalibrationReport:
    """Sample-run ``lens`` and assemble a `CalibrationReport`.

    Selects up to ``sample`` works (or the explicit ``work_ids``), runs the lens
    in sample mode (``runner.run_lens(..., sample=True)`` — sets
    ``status='calibrating'``, does not promote to active), and collects per-record
    found/not-found rows with verbatim quotes, span back-refs, and confidence for
    FP/FN review. Implements the docs/06 §7 calibration workflow steps 4–5.

    ``work_ids`` defaults to the project's included corpus (capped at ``sample``).
    ``force`` defaults to True so re-calibrating after a YAML edit reprocesses the
    sample without needing ``--force`` plumbing.
    """
    if work_ids is None:
        from ..db.adapter import raw_conn

        conn = raw_conn(session)
        work_ids = [
            r[0]
            for r in conn.execute(
                "SELECT work_id FROM project_documents WHERE inclusion_status = 'included' "
                "ORDER BY created_at LIMIT ?",
                (sample,),
            ).fetchall()
        ]
    else:
        work_ids = list(work_ids)[:sample] if sample else list(work_ids)

    run_lens(session, cache_db, lens, work_ids, router, sample=True, force=force)

    report = CalibrationReport(lens_id=lens.lens_id, sample_work_ids=list(work_ids))
    found_views = results_mod.lens_results(session, lens.lens_id, status="found")
    for view in found_views:
        if view.work_id not in work_ids:
            continue
        report.found_records.append(
            {
                "work_id": view.work_id,
                "claim_text": view.claim_text,
                "normalized_label": view.normalized_label,
                "section": view.section,
                "confidence": view.confidence,
                "span_ids": view.span_ids,
                "fields": view.fields,
            }
        )
    for view in results_mod.lens_results(session, lens.lens_id, status="not_found"):
        if view.work_id in work_ids:
            report.not_found_work_ids.append(view.work_id)

    # Review items raised during the sample (ambiguous / extraction_failed) feed the
    # FP/FN inspection surface.
    from ..db.adapter import raw_conn

    conn = raw_conn(session)
    for item_id, target_id, payload in conn.execute(
        "SELECT item_id, target_id, payload FROM review_queue "
        "WHERE item_type = 'lens_record' AND status = 'open'"
    ).fetchall():
        report.review_items.append(
            {"item_id": item_id, "target_id": target_id, "payload": payload}
        )
    return report
