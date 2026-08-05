"""Post-walk title backfill (§5.10, Build D ch9) — a READ-REPAIR pass, no schema.

A walk-discovered stub can resolve a real ``W…`` ``openalex_id`` yet lose its
title to a provider hiccup: the per-ref enrichment miss is negative-cached
under ``by_doi:`` for ~30 days, so without a repair path the untitled state
would stay pinned across re-walks. This pass is that recovery path:

1. select ``works`` rows with an ``openalex_id`` and a NULL **or empty**
   ``canonical_title`` — match BOTH forms (the v1 lesson: a once-NULL title
   can be persisted as ``""`` by an intermediate upsert);
2. batch the W-ids through ``chain.by_openalex_ids`` in ``ceil(N/batch_size)``
   calls (§4.7 politeness). The verb's DISTINCT ``by_openalex_ids:`` cache
   namespace is LOAD-BEARING here (D-8): a stale 30-day ``by_doi:`` negative
   for the same W-id can never starve this pass, and an empty batch result is
   never negative-cached, so a whole-batch outage is retried next run;
3. fold each titled record back via :func:`identity.upsert_work` — the
   empty-only fill NEVER regresses a present title, and the incoming record
   carries the W-id so it folds into the EXISTING row (no duplicate).

Provably idempotent: a filled row leaves the selection, so a re-run reports
``queried == 0`` and issues zero provider calls. A parked
``duplicate_candidate`` row that only MIRRORS a W-id another work owns in
``identifiers`` is excluded up front: the fold resolves through the
``identifiers`` table, so it could never fill that row — selecting it would
re-bill the provider every run without ever converging (review resolution,
not this pass, is its repair path). A row the provider cannot title stays
exactly as it was and is counted ``still_missing`` (the v1
predict-then-prove discipline: the four counters are the pass's contract).

Exposed as ``corpus backfill-titles`` (cli.py) and hooked FAIL-SOFT between
walk and acquire inside ``service.run_corpus`` (which writes the counters once
as the additive ``title_backfill`` manifest section).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import or_
from sqlmodel import Session, select

from ..db.project_models import Identifier, Work
from ..project import identity

if TYPE_CHECKING:
    from ..project.service import ProjectHandle
    from ..providers.base import ProviderChain

__all__ = ["BackfillResult", "backfill_titles"]


@dataclass
class BackfillResult:
    """Counters from one :func:`backfill_titles` pass (v1 parity).

    * ``queried`` — works selected as untitled-with-W-id (the backfill universe).
    * ``fetched`` — titled records the chain returned across all batches.
    * ``filled`` — works whose title the fold actually populated.
    * ``still_missing`` — selected works still title-less after the pass.
    """

    queried: int = 0
    fetched: int = 0
    filled: int = 0
    still_missing: int = 0

    def to_section(self) -> dict:
        """The small additive ``title_backfill`` manifest section (D5): written
        ONCE by ``run_corpus``; this pass is its sole owner."""
        return {
            "queried": self.queried,
            "fetched": self.fetched,
            "filled": self.filled,
            "still_missing": self.still_missing,
        }


async def backfill_titles(
    h: "ProjectHandle",
    chain: "ProviderChain",
    *,
    batch_size: int = 50,
) -> BackfillResult:
    """Fill NULL/empty ``canonical_title`` on works that hold an ``openalex_id``.

    Selects every ``works`` row with a non-empty ``openalex_id`` and a NULL or
    empty ``canonical_title`` (both forms — the v1 lesson) that also OWNS its
    ``openalex`` row in ``identifiers`` — a parked ``duplicate_candidate``
    that merely mirrors a W-id another work owns is unreachable by the fold
    below, so selecting it would burn a provider call every run for a row this
    pass can never fill. Batch-fetches the selected W-ids through
    ``chain.by_openalex_ids`` (``ceil(N / batch_size)`` calls, NOT N), and
    folds each titled record back via ``identity.upsert_work``:
    the empty-only fill lands the title (and cheap co-metadata: year/authors)
    on the EXISTING W-id-owned row — no duplicate row, and a present title is
    never overwritten. ``chain`` need only expose
    ``async by_openalex_ids(ids) -> list[dict]``, so the pass is fully
    offline-testable with an injected double.

    Returns a :class:`BackfillResult`; re-runs are idempotent (``queried == 0``
    once every selectable row is titled).
    """
    result = BackfillResult()
    if batch_size < 1:
        batch_size = 1

    with Session(h.engine, expire_on_commit=False) as session:
        # The backfill universe: a resolvable W-id but no usable title. NULL and
        # empty string are both "untitled" (a once-NULL title can be persisted
        # as "" by an intermediate upsert — the v1 lesson), so match both. The
        # row must also OWN its openalex row in `identifiers`: a parked
        # duplicate_candidate mirrors the W-id onto its scalar column while the
        # identifiers row belongs to ANOTHER work, so the upsert fold below
        # could never reach it — selecting it would re-query the provider every
        # run without ever filling the row (idempotence is a docstring promise;
        # review resolution, not this pass, is that row's repair path).
        owns_wid = (
            select(Identifier.work_id)
            .where(
                Identifier.work_id == Work.work_id,
                Identifier.id_type == "openalex",
                Identifier.id_value == Work.openalex_id,
            )
            .exists()
        )
        untitled = list(
            session.exec(
                select(Work).where(
                    Work.openalex_id.is_not(None),
                    Work.openalex_id != "",
                    or_(Work.canonical_title.is_(None), Work.canonical_title == ""),
                    owns_wid,
                )
            ).all()
        )
        result.queried = len(untitled)
        if not untitled:
            return result

        # Order-preserving dedupe. UNIQUE(id_type, id_value) + the ownership
        # filter above make one selected row per W-id the invariant; this loop
        # is cheap insurance (and strips stray whitespace) rather than load-
        # bearing dedupe.
        wids: list[str] = []
        seen: set[str] = set()
        for work in untitled:
            wid = (work.openalex_id or "").strip()
            if wid and wid not in seen:
                seen.add(wid)
                wids.append(wid)

        # ceil(N/batch_size) batched calls through the chain's DISTINCT
        # by_openalex_ids: namespace — never by_doi, whose 30-day negatives
        # pinned the untitled state in the first place (D-8 cache hygiene).
        fetched: dict[str, dict] = {}
        for start in range(0, len(wids), batch_size):
            records = await chain.by_openalex_ids(wids[start : start + batch_size])
            for record in records or []:
                if not isinstance(record, dict):
                    continue
                rec_wid = identity.normalize_id(
                    "openalex", record.get("openalex_id") or record.get("openalex")
                )
                if rec_wid and (record.get("title") or "").strip():
                    fetched.setdefault(rec_wid, record)
        result.fetched = len(fetched)

        for wid, record in fetched.items():
            # The incoming carries the W-id, so upsert_work folds it into the
            # EXISTING row (merge; empty-only fill) — no duplicate. The record's
            # ``doi`` is DELIBERATELY dropped from the fold (v1 payload parity):
            # a DOI disagreeing with one the work already owns would demote the
            # merge to duplicate_candidate — a read-repair pass must never mint
            # duplicate works or review items just to fill a title.
            incoming: dict = {"openalex": wid, "title": record.get("title")}
            if record.get("year") is not None:
                incoming["year"] = record["year"]
            if record.get("authors"):
                incoming["authors"] = record["authors"]
            work, _outcome = identity.upsert_work(session, incoming)
            if (work.canonical_title or "").strip():
                result.filled += 1
        session.commit()

    result.still_missing = max(result.queried - result.filled, 0)
    return result
