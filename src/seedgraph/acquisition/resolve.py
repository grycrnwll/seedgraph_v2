"""Metadata resolution + confidence rubric (§6.3) — deterministic, no-LLM.

Implements decision 15/58 (no-LLM) + the decision r2-1 rubric + doc 04 §5 (ambiguous
title-only -> review, never silent merge). Fills the ``identifiers.resolution_source``
/ ``confidence`` columns phase_5 §7 reserved, and routes mid-band title matches to
``project.review.enqueue`` (decision 13/73/80).

Rubric (decision r2-1):
* strong-id hit                       -> ``confidence=1.0``, ``status='resolved'``,
                                         ``resolution_source=<idtype>``.
* title+year, sim >= ``SIM_RESOLVE`` & year ok -> ``resolved`` (``confidence=sim``).
* ``SIM_REVIEW`` <= sim < ``SIM_RESOLVE``      -> enqueue review
                                         (``title_year_ambiguous``), ``status='ambiguous'``,
                                         NO merge.
* sim < ``SIM_REVIEW``                 -> ``status='unresolved'`` (no merge, no edge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlmodel import Session, select

from ..db.project_models import Identifier, Work
from ..project import identity, review
from ..project.identity import canonical_key, normalize_title, strong_ids

if TYPE_CHECKING:
    from ..progress import Progress
    from ..project.service import ProjectHandle
    from ..providers.base import ProviderChain

# Rubric thresholds (decision r2-1): token-Jaccard over normalize_title(); the
# year must match within +/-1 for a title match to count.
SIM_RESOLVE = 0.90
SIM_REVIEW = 0.65


@dataclass
class ResolutionOutcome:
    """Result of resolving ONE work via the rubric."""

    work_id: str
    status: str
    confidence: float | None = None
    resolution_source: str | None = None
    review_kind: str | None = None
    matched_record: dict | None = None


@dataclass
class ResolveReport:
    """Aggregate counters for the resolve stage — written ONCE as the disjoint
    ``resolution`` manifest section (D5; no cross-stage accumulation)."""

    resolved: int = 0
    ambiguous: int = 0
    unresolved: int = 0
    reviewed: int = 0
    per_provider_calls: dict = field(default_factory=dict)

    def to_section(self) -> dict:
        return {
            "resolved": self.resolved,
            "ambiguous": self.ambiguous,
            "unresolved": self.unresolved,
            "reviewed": self.reviewed,
            "per_provider_calls": dict(self.per_provider_calls),
        }


def _tokens(title: str | None) -> set[str]:
    return set(normalize_title(title or "").split())


def score_title_match(query_title: str, query_year: int | None, cand: dict) -> float:
    """Token-Jaccard ``|A∩B| / |A∪B|`` over ``normalize_title()`` token sets.

    Returns 0.0 when the candidate year is present and differs from ``query_year``
    by more than 1 (the year gate). Deterministic, no-LLM (decision 15/58).
    """
    cand_year = cand.get("year") if isinstance(cand, dict) else None
    if query_year is not None and cand_year is not None:
        try:
            if abs(int(cand_year) - int(query_year)) > 1:
                return 0.0
        except (TypeError, ValueError):
            pass
    a = _tokens(query_title)
    b = _tokens(cand.get("title") if isinstance(cand, dict) else None)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _work_to_incoming(work: Work) -> dict:
    incoming: dict = {}
    for attr, key in (
        ("doi", "doi"),
        ("openalex_id", "openalex"),
        ("arxiv_id", "arxiv"),
        ("semantic_scholar_id", "s2"),
        ("ssrn_id", "ssrn"),
    ):
        value = getattr(work, attr, None)
        if value:
            incoming[key] = value
    if work.canonical_title:
        incoming["title"] = work.canonical_title
    if work.year is not None:
        incoming["year"] = work.year
    return incoming


def _stamp_identifiers(session: Session, work_id: str, *, confidence: float, source: str | None) -> None:
    rows = session.exec(select(Identifier).where(Identifier.work_id == work_id)).all()
    for row in rows:
        row.confidence = confidence
        row.resolution_source = source if source is not None else row.id_type
        session.add(row)
    session.flush()


def _attach_resolved_ids(session: Session, work: Work, cand: dict, *, confidence: float) -> None:
    """Attach a title-resolved candidate's UNCLAIMED strong ids to ``work``.

    Sets the work's scalar id columns and inserts ``identifiers`` rows
    (``resolution_source='title'``) for each candidate strong id not already owned
    by another work — never blind-inserting a colliding id (no UNIQUE crash).
    """
    for id_type, id_value in strong_ids(cand):
        existing = session.exec(
            select(Identifier).where(
                Identifier.id_type == id_type, Identifier.id_value == id_value
            )
        ).first()
        if existing is not None:
            continue  # already claimed -> do not steal / collide
        column = identity.ID_TYPE_TO_WORK_COLUMN[id_type]
        if getattr(work, column, None) is None:
            setattr(work, column, id_value)
        session.add(
            Identifier(
                work_id=work.work_id,
                id_type=id_type,
                id_value=id_value,
                resolution_source="title",
                confidence=confidence,
            )
        )
    session.add(work)
    session.flush()


async def resolve_record(chain: "ProviderChain", project_session: Session, *, work) -> ResolutionOutcome:
    """Resolve one ``work`` against the provider chain per the rubric above.

    Strong-id hits auto-resolve (``confidence=1.0``); mid-band title matches are
    enqueued to ``review_queue`` (never merged); sub-threshold matches are left
    ``unresolved``. Fills ``identifiers.resolution_source`` / ``confidence`` on a
    resolve. NO-LLM; uses phase_5 ``normalize_title`` + identity helpers (never
    re-implements the merge). Async because it consults the async provider chain.
    """
    incoming = _work_to_incoming(work)
    key = canonical_key(incoming)
    if key is not None:
        # Strong-id hit: deterministic confidence 1.0; the strongest id is the source.
        idtype = key[0]
        _stamp_identifiers(project_session, work.work_id, confidence=1.0, source=idtype)
        return ResolutionOutcome(
            work_id=work.work_id,
            status="resolved",
            confidence=1.0,
            resolution_source=idtype,
        )

    # Title-only path.
    title = incoming.get("title")
    year = incoming.get("year")
    if not title:
        return ResolutionOutcome(work_id=work.work_id, status="unresolved")

    cand = await chain.by_title(title, year)
    if not cand:
        return ResolutionOutcome(work_id=work.work_id, status="unresolved")

    sim = score_title_match(title, year, cand)
    if sim >= SIM_RESOLVE:
        _attach_resolved_ids(project_session, work, cand, confidence=sim)
        _stamp_identifiers(project_session, work.work_id, confidence=sim, source="title")
        return ResolutionOutcome(
            work_id=work.work_id, status="resolved", confidence=sim,
            resolution_source="title", matched_record=cand,
        )
    if sim >= SIM_REVIEW:
        # Mid-band: route to review_queue — NEVER auto-merge (doc 04 §5). The review
        # payload union (phase_5-owned) carries no title_year variant, so the row is
        # enqueued payload-less; the kind rides the outcome.
        review.enqueue_in_session(
            project_session,
            "duplicate_candidate",
            target_type="work",
            target_id=work.work_id,
            payload=None,
        )
        return ResolutionOutcome(
            work_id=work.work_id, status="ambiguous", confidence=sim,
            review_kind="title_year_ambiguous", matched_record=cand,
        )
    return ResolutionOutcome(work_id=work.work_id, status="unresolved", confidence=sim)


async def resolve_corpus(
    h: "ProjectHandle",
    chain: "ProviderChain",
    *,
    statuses: tuple[str, ...] = ("included", "metadata_only"),
    run_id: str | None = None,
    progress: "Progress | None" = None,
) -> ResolveReport:
    """Resolve every work in ``statuses``; fill identifiers; route mid-band matches
    to ``review_queue``; return the disjoint ``resolution`` section counters (D5).

    No network when ``provider_cache`` is fresh (offline-resilient; decision 39/73).
    When ``run_id`` is given, writes the single ``resolution`` manifest section once.

    ``progress`` (additive, Build F ch10 — the resolve-loop rider on Build D ch6's
    D-7 hook) steps once per work with ``detail=work_id`` and the running rubric
    counters; the caller builds :class:`Progress` before the select, so its
    placeholder total is reconciled here from the loaded worklist. ``emit=None``
    (the CLI verbs) keeps it stdout-only — no non-terminal ``progress`` event lands
    in ``events.jsonl``, so a CLI resolve can never strand ``_status_from_events``
    on ``running``; ``None`` (default) disables it.
    """
    from ..db.project_models import ProjectDocument

    report = ResolveReport()
    with Session(h.engine, expire_on_commit=False) as session:
        works = list(
            session.exec(
                select(Work)
                .join(ProjectDocument, ProjectDocument.work_id == Work.work_id)
                .where(ProjectDocument.inclusion_status.in_(tuple(statuses)))
                .order_by(Work.created_at)
            ).all()
        )
        if progress is not None:
            # This loop owns the true target count — the caller constructs Progress
            # BEFORE the select, so its total is a placeholder we reconcile here,
            # mirroring _acquire_async / walk_corpus (Build D ch6 / D-7).
            progress.total = len(works)
        for work in works:
            outcome = await resolve_record(chain, session, work=work)
            if outcome.status == "resolved":
                report.resolved += 1
            elif outcome.status == "ambiguous":
                report.ambiguous += 1
                report.reviewed += 1
            else:
                report.unresolved += 1
            if progress is not None:
                # One frame per work, detail=work_id, carrying the running rubric
                # counters (the resolve-loop one-liner Build D's plan names). With
                # emit=None this is stdout-only — the event-silent invariant risk 7's
                # crash-sweep rationale and chunk 9's interrupted-sweep both rest on.
                progress.step(
                    work.work_id,
                    resolved=report.resolved,
                    ambiguous=report.ambiguous,
                    unresolved=report.unresolved,
                )
        session.commit()

    report.per_provider_calls = dict(getattr(chain, "call_counts", {}))
    if run_id is not None:
        from .. import run as run_mod

        run_mod.update_manifest(h.slug, run_id, {"resolution": report.to_section()}, root=h.root)
    return report
