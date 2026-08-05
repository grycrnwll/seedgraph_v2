"""Lens result/query helpers (plan §5/§6.2/§9) — the success-criterion surface.

`lens_results` is the primary phase-6 success criterion: return every record for
a lens across the project with its verbatim quote, normalized label,
`condition_type`, section, and evidence-span back-reference. `coverage` reports
found/not_found/skipped-no-markdown separately for honest recall accounting
(plan §9); `stale_outputs` lists runs whose `lens_definition_hash` or
`markdown_hash` no longer match (pull-based staleness, plan §7).

`lens results` reads the *latest non-stale* run per work (shadow-don't-delete).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..acquisition.bridge import resolve_work_markdown
from ..db.adapter import raw_conn

if TYPE_CHECKING:
    from sqlmodel import Session

    from .schema import LensDefinition


@dataclass(frozen=True)
class LensOutputView:
    """One row of `lens results`: the projected claim spine + the full instance.

    The typed refinement (e.g. `condition_type[]`) lives in `fields_json` /
    `fields`, not in `claim_type` (§4.4 rule); `span_ids` are the evidence
    back-references.
    """

    lens_output_id: str
    work_id: str
    status: str
    claim_id: str | None
    claim_text: str | None
    normalized_label: str | None
    section: str | None
    confidence: float | None
    span_ids: list[str] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    access_class: str = "user_supplied_private"


@dataclass(frozen=True)
class CoverageReport:
    """Honest recall accounting (plan §9): found/not_found/skipped split out."""

    lens_id: str
    found: int = 0
    not_found: int = 0
    ambiguous: int = 0
    extraction_failed: int = 0
    not_applicable: int = 0
    skipped_no_markdown: int = 0
    works_total: int = 0
    works_covered: int = 0


def _current_def_hash(conn, lens_id: str) -> str | None:
    row = conn.execute(
        "SELECT definition_hash FROM lenses WHERE lens_id = ?", (lens_id,)
    ).fetchone()
    return row[0] if row is not None else None


def _current_run_per_work(session: "Session", lens_id: str) -> dict[str, str]:
    """Map work_id -> the latest NON-STALE extraction_run_id for ``lens_id``.

    Non-stale = the run's ``lens_definition_hash`` equals the registry's current
    ``definition_hash`` AND its ``markdown_hash`` equals the work's current
    markdown_hash (resolved through the bridge). Shadow-don't-delete: prior runs
    remain but are not surfaced.
    """
    conn = raw_conn(session)
    def_hash = _current_def_hash(conn, lens_id)
    if def_hash is None:
        return {}
    rows = conn.execute(
        "SELECT extraction_run_id, work_id, markdown_hash, created_at "
        "FROM extraction_runs WHERE lens_id = ? AND lens_definition_hash = ? "
        "ORDER BY created_at ASC, rowid ASC",
        (lens_id, def_hash),
    ).fetchall()
    current_md: dict[str, str | None] = {}
    chosen: dict[str, str] = {}
    for run_id, work_id, md_hash, _created in rows:
        if work_id not in current_md:
            resolved = resolve_work_markdown(session, work_id=work_id)
            current_md[work_id] = resolved[1] if resolved is not None else None
        if md_hash == current_md[work_id]:
            chosen[work_id] = run_id  # ASC order -> last assignment is the latest
    return chosen


def lens_results(
    session: "Session",
    lens_id: str,
    status: str = "found",
) -> list[LensOutputView]:
    """Return all `lens_outputs` for ``lens_id`` filtered by ``status``
    (``found``/``not_found``/``all``), each joined to its projected claim +
    evidence spans, reading the latest non-stale run per work. This is the
    `lens results` success criterion (plan §1/§6.1).
    """
    conn = raw_conn(session)
    run_ids = set(_current_run_per_work(session, lens_id).values())
    if not run_ids:
        return []
    placeholders = ", ".join(["?"] * len(run_ids))
    rows = conn.execute(
        f"SELECT lens_output_id, work_id, status, claim_id, confidence, fields_json, "
        f"access_class, extraction_run_id FROM lens_outputs "
        f"WHERE lens_id = ? AND extraction_run_id IN ({placeholders}) "
        f"ORDER BY work_id, created_at",
        (lens_id, *run_ids),
    ).fetchall()

    views: list[LensOutputView] = []
    for (lo_id, work_id, st, claim_id, confidence, fields_json, access_class, _run) in rows:
        if status != "all" and st != status:
            continue
        try:
            fields = json.loads(fields_json) if fields_json else {}
        except json.JSONDecodeError:
            fields = {}
        claim_text = None
        normalized_label = None
        span_ids: list[str] = []
        if claim_id is not None:
            crow = conn.execute(
                "SELECT claim_text, normalized_label FROM extracted_claims WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()
            if crow is not None:
                claim_text, normalized_label = crow[0], crow[1]
            span_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT span_id FROM claim_spans WHERE claim_id = ? ORDER BY rank",
                    (claim_id,),
                ).fetchall()
            ]
        section = fields.get("section") if isinstance(fields, dict) else None
        views.append(
            LensOutputView(
                lens_output_id=lo_id,
                work_id=work_id,
                status=st,
                claim_id=claim_id,
                claim_text=claim_text,
                normalized_label=normalized_label,
                section=section,
                confidence=confidence,
                span_ids=span_ids,
                fields=fields if isinstance(fields, dict) else {},
                access_class=access_class,
            )
        )
    return views


def stale_outputs(session: "Session", lens: "LensDefinition") -> list[str]:
    """`extraction_run_id`s whose `lens_definition_hash` != the lens's current
    `definition_hash` OR whose `markdown_hash` != the work's current
    `markdown_hash` (plan §7 pull-based staleness). Detection only — recompute is
    user-triggered via `lens run`.
    """
    conn = raw_conn(session)
    current_hash = lens.definition_hash()
    rows = conn.execute(
        "SELECT extraction_run_id, work_id, markdown_hash, lens_definition_hash "
        "FROM extraction_runs WHERE lens_id = ?",
        (lens.lens_id,),
    ).fetchall()
    current_md: dict[str, str | None] = {}
    stale: list[str] = []
    for run_id, work_id, md_hash, def_hash in rows:
        if def_hash != current_hash:
            stale.append(run_id)
            continue
        if work_id not in current_md:
            resolved = resolve_work_markdown(session, work_id=work_id)
            current_md[work_id] = resolved[1] if resolved is not None else None
        if md_hash != current_md[work_id]:
            stale.append(run_id)
    return stale


def coverage(session: "Session", lens_id: str) -> CoverageReport:
    """Per-status coverage for ``lens_id`` including the separate
    skipped-no-markdown count (plan §9 honest recall accounting).

    Iterates the project's included corpus: a work with no selected markdown is
    counted in ``skipped_no_markdown`` (not as a failure); otherwise the latest
    non-stale run's lens_outputs are tallied by status.
    """
    conn = raw_conn(session)
    current_runs = _current_run_per_work(session, lens_id)

    included = [
        r[0]
        for r in conn.execute(
            "SELECT work_id FROM project_documents WHERE inclusion_status = 'included'"
        ).fetchall()
    ]
    counts = {
        "found": 0,
        "not_found": 0,
        "ambiguous": 0,
        "extraction_failed": 0,
        "not_applicable": 0,
    }
    skipped_no_markdown = 0
    works_covered = 0
    for work_id in included:
        resolved = resolve_work_markdown(session, work_id=work_id)
        if resolved is None:
            skipped_no_markdown += 1
            continue
        run_id = current_runs.get(work_id)
        if run_id is None:
            continue
        works_covered += 1
        for (st,) in conn.execute(
            "SELECT status FROM lens_outputs WHERE lens_id = ? AND extraction_run_id = ?",
            (lens_id, run_id),
        ).fetchall():
            if st in counts:
                counts[st] += 1
    return CoverageReport(
        lens_id=lens_id,
        found=counts["found"],
        not_found=counts["not_found"],
        ambiguous=counts["ambiguous"],
        extraction_failed=counts["extraction_failed"],
        not_applicable=counts["not_applicable"],
        skipped_no_markdown=skipped_no_markdown,
        works_total=len(included),
        works_covered=works_covered,
    )
