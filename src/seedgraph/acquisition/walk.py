"""N-generation outbound corpus-growth walk (ported v1 ``walk.py``) — §6.3.

This phase is the SOLE owner of outbound corpus growth (binding r2-1 §4; D3):

1. resolve seeds -> expand outbound ``referenced_works``;
2. upsert EVERY discovered target as a metadata-only ``works`` row +
   ``project_documents(inclusion_status='metadata_only', inclusion_reason='citation_walk')``
   (decision 59/74 — so phase_2's edges never dangle);
3. rank the NEXT frontier by in-walk citation frequency; ``per_gen_cap`` / ``depth``
   bound the frontier expansion (mitigates v1's 122->1503 blowup);
4. DOI/strong-id backfill onto discovered works from the OpenAlex record already
   fetched (decision r2-1 §6);
5. cache each source work's reference list to ``provider_cache`` under
   ``key_referenced_works`` (§6.5) — done by the chain when it fetches.

Writes **NO** ``citation_edges`` row — phase_2 projects those offline from the
cache this walk populates (binding r2-1 §4). Inbound ``cited_by`` is a documented
seam only (decision 73/74).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlmodel import Session, select

from sqlalchemy import func, or_

from ..db.project_models import ProjectDocument, Work
from ..project import identity

if TYPE_CHECKING:
    from pathlib import Path

    from ..progress import Progress
    from ..project.service import ProjectHandle
    from ..providers.base import ProviderChain

#: Per-source batch size for the ch8 enrichment prefetch: a source's DOI-less
#: refs cost ceil(N/50) ``by_openalex_ids`` calls instead of N in-band ``by_doi``
#: calls (the §4.7 politeness rationale — hundreds-to-thousands of single-record
#: calls on a cold-cache run). 50 mirrors the v1 backfill chunking, so one
#: ``per-page=200`` request always covers a batch.
_ENRICH_BATCH_SIZE = 50


@dataclass
class WalkReport:
    """Walk-stage counters — written ONCE as the disjoint ``walk`` manifest section
    (D5). ``frontier_sizes`` is per-generation; ``per_provider_calls`` is for
    provenance. NO edge counts (this phase writes no edges).

    ``enriched`` / ``enrichment_failed`` count the DOI-less-reference enrichment
    outcomes (Build D ch8): a ref that folded a batched/fallback record vs one
    whose batch AND fallback both came up empty — a measured number, not the old
    silent swallow. ``untitled_after_walk`` is the post-walk residue (works with
    an ``openalex_id`` but NULL/empty ``canonical_title``), computed once, so the
    manifest distinguishes a fully-titled corpus from one where stubs lost their
    titles to a provider hang. All three serialize as ADDITIVE keys of the walk
    section this stage solely owns.

    ``citation_counts`` is the in-walk citation frequency per ``work_id`` (the
    per-generation ``next_counts`` accumulated across generations) — the Build D
    D-5 value-ordering hint ``run_corpus`` threads into the acquire pass.
    IN-MEMORY ONLY: deliberately excluded from :meth:`to_section` (the walk
    manifest section shape is pinned)."""

    depth: int = 0
    discovered: int = 0
    enriched: int = 0
    enrichment_failed: int = 0
    untitled_after_walk: int = 0
    frontier_sizes: list = field(default_factory=list)
    per_provider_calls: dict = field(default_factory=dict)
    citation_counts: dict = field(default_factory=dict)

    def to_section(self) -> dict:
        return {
            "depth": self.depth,
            "discovered": self.discovered,
            "enriched": self.enriched,
            "enrichment_failed": self.enrichment_failed,
            "untitled_after_walk": self.untitled_after_walk,
            "frontier_sizes": list(self.frontier_sizes),
            "per_provider_calls": dict(self.per_provider_calls),
        }


def _work_to_record(work: Work) -> dict:
    record: dict = {}
    for attr in ("openalex_id", "doi", "arxiv_id", "semantic_scholar_id", "ssrn_id"):
        value = getattr(work, attr, None)
        if value:
            record[attr] = value
    if work.canonical_title:
        record["title"] = work.canonical_title
    if work.year is not None:
        record["year"] = work.year
    return record


def _ref_to_incoming(ref: dict) -> dict:
    """Map a provider reference record to an identity ``incoming`` mapping."""
    incoming: dict = {}
    for src, dst in (
        ("doi", "doi"),
        ("openalex_id", "openalex"),
        ("openalex", "openalex"),
        ("arxiv_id", "arxiv"),
        ("arxiv", "arxiv"),
        ("semantic_scholar_id", "s2"),
        ("s2_id", "s2"),
        ("ssrn_id", "ssrn"),
    ):
        if ref.get(src) and dst not in incoming:
            incoming[dst] = ref[src]
    if ref.get("title"):
        incoming["title"] = ref["title"]
    if ref.get("year") is not None:
        incoming["year"] = ref["year"]
    if ref.get("authors"):
        incoming["authors"] = ref["authors"]
    # Build D ch10: pass the triage metadata through to the empty-only
    # identity fold (walk-discovered stubs ARE the unfetched papers the
    # abstract/OA-color triage exists for). Reaches works.* via upsert_work.
    if ref.get("abstract"):
        incoming["abstract"] = ref["abstract"]
    if ref.get("oa_status"):
        incoming["oa_status"] = ref["oa_status"]
    return incoming


async def expand_references(
    chain: "ProviderChain",
    project_session: Session,
    *,
    source_work_id: str,
    run_id: str,
    report: "WalkReport | None" = None,
) -> list[str]:
    """Expand one source work's outbound ``referenced_works`` (§6.3).

    For each referenced target: DOI/strong-id backfill from the OpenAlex record
    (decision r2-1 §6) — served from a per-source ``by_openalex_ids`` batch
    prefetch (ceil(N/50) calls, Build D ch8) with per-ref ``by_doi`` kept ONLY
    as fallback for ids the batch did not return — then ``upsert_work`` as
    ``metadata_only`` + write its ``project_documents`` row
    (``inclusion_reason='citation_walk'``); the chain caches the reference list
    under ``key_referenced_works``. ``report`` (additive, ch8) receives the
    ``enriched``/``enrichment_failed`` counts. Returns the cited ``work_id``
    list (deduped, order-preserving). Writes NO ``citation_edges``
    (binding r2-1 §4).
    """
    source = project_session.get(Work, source_work_id)
    if source is None:
        return []
    record = _work_to_record(source)
    refs = await chain.referenced_works(record)

    # Batched enrichment prefetch (Build D ch8): collect the DOI-less refs'
    # W-ids up front and fetch them in ceil(N/50) ``by_openalex_ids`` calls per
    # source instead of one in-band ``by_doi`` per ref (§4.7 politeness). Ch7's
    # ``select`` carries ``doi``, so the per-key ``setdefault`` fold below (the
    # r2-1 §6 DOI backfill) is unchanged. A raising/absent batch verb degrades
    # to the per-ref fallback — never fatal.
    need_ids: list[str] = []
    need_seen: set[str] = set()
    for ref in refs:
        if not isinstance(ref, dict) or ref.get("doi"):
            continue
        wid = identity.normalize_id("openalex", ref.get("openalex_id") or ref.get("openalex"))
        if wid and wid not in need_seen:
            need_seen.add(wid)
            need_ids.append(wid)
    prefetched: dict[str, dict] = {}
    for i in range(0, len(need_ids), _ENRICH_BATCH_SIZE):
        try:
            batch = await chain.by_openalex_ids(need_ids[i : i + _ENRICH_BATCH_SIZE])
        except Exception:  # noqa: BLE001 - a failed batch falls back per-ref below
            batch = []
        for rec in batch or []:
            if not isinstance(rec, dict):
                continue
            wid = identity.normalize_id("openalex", rec.get("openalex_id") or rec.get("openalex"))
            if wid:
                prefetched.setdefault(wid, rec)

    cited: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        # DOI/strong-id backfill: a bare ``{"openalex_id": "W.."}`` reference gets its
        # DOI (and title/year) from the OpenAlex record already reachable (r2-1 §6).
        if not ref.get("doi") and (ref.get("openalex_id") or ref.get("openalex")):
            raw_wid = ref.get("openalex_id") or ref.get("openalex")
            wid = identity.normalize_id("openalex", raw_wid)
            enriched = prefetched.get(wid) if wid else None
            if enriched is None:
                # Fallback ONLY for ids the batch did not return (ch8): preserves
                # multi-provider recovery and today's negative-cache semantics
                # for genuinely missing works.
                try:
                    enriched = await chain.by_doi(raw_wid)
                except Exception:  # noqa: BLE001 - one ref's enrichment failure is not fatal
                    enriched = None
            if isinstance(enriched, dict):
                for key, value in enriched.items():
                    ref.setdefault(key, value)
                if report is not None:
                    report.enriched += 1
            elif report is not None:
                # Batch AND fallback both came up empty/raised: the stub still
                # upserts title-less below (existing behavior preserved) — but
                # the failure is now a measured number, not a silent swallow (ch8).
                report.enrichment_failed += 1

        incoming = _ref_to_incoming(ref)
        if not identity.canonical_key(incoming) and not incoming.get("title"):
            continue  # nothing resolvable -> cannot become a node
        work, _outcome = identity.upsert_work(project_session, incoming)
        # Ensure a metadata-only / citation_walk membership row (never downgrade an
        # already-included work).
        doc = project_session.get(ProjectDocument, work.work_id)
        if doc is None:
            now = identity._now()
            project_session.add(
                ProjectDocument(
                    work_id=work.work_id,
                    inclusion_status="metadata_only",
                    inclusion_reason="citation_walk",
                    is_seed=0,
                    access_status=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            project_session.flush()
        if work.work_id not in seen:
            seen.add(work.work_id)
            cited.append(work.work_id)
    return cited


def _dump_frontier(frontier_dir: "Path | None", gen: int, frontier: list[str], ranked: list) -> None:
    if frontier_dir is None:
        return
    try:
        frontier_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generation": gen,
            "frontier": list(frontier),
            "ranked": [{"work_id": uid, "citation_count": c} for uid, c in ranked],
        }
        (frontier_dir / f"gen_{gen}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


async def walk_corpus(
    h: "ProjectHandle",
    chain: "ProviderChain",
    *,
    run_id: str,
    depth: int = 2,
    per_gen_cap: int = 50,
    frontier_dir: "Path | None" = None,
    progress: "Progress | None" = None,
) -> WalkReport:
    """Interleaved-BFS outbound walk to ``depth`` (ported v1).

    Grows the corpus (works + ``metadata_only`` membership + cached reference
    lists) without writing edges; ranks/caps the next frontier by citation
    frequency; optionally dumps ``frontier/gen_{i}.json`` under ``frontier_dir``.
    Builds fully offline from ``provider_cache`` when warm (decision 39/73). Writes
    the single ``walk`` manifest section once (D5). ``progress`` (additive, D-7)
    steps once per expanded source work per generation — the walk's total is only
    known generation-by-generation, so each generation extends ``progress.total``
    by its frontier size before stepping; ``emit=None`` (the CLI) keeps it
    stdout-only, ``None`` disables it.
    """
    report = WalkReport(depth=depth)
    with Session(h.engine, expire_on_commit=False) as session:
        seeds = list(
            session.exec(
                select(Work)
                .join(ProjectDocument, ProjectDocument.work_id == Work.work_id)
                .where(ProjectDocument.inclusion_status == "included")
                .order_by(Work.created_at)
            ).all()
        )
        frontier = [w.work_id for w in seeds]
        all_nodes: set[str] = set(frontier)
        report.frontier_sizes.append(len(frontier))
        _dump_frontier(frontier_dir, 0, frontier, [])

        for gen in range(1, depth + 1):
            if not frontier:
                break
            if progress is not None:
                # The walk's universe grows as it expands: extend the running
                # total by THIS generation's frontier before stepping it (D-7).
                progress.total += len(frontier)
            next_counts: Counter[str] = Counter()
            for source_uid in frontier:
                cited = await expand_references(
                    chain, session, source_work_id=source_uid, run_id=run_id, report=report
                )
                for cited_uid in cited:
                    if cited_uid == source_uid:
                        continue
                    next_counts[cited_uid] += 1
                if progress is not None:
                    # One frame per expanded source work per generation; emit=None
                    # (CLI) keeps this stdout-only so no non-terminal event can
                    # strand _status_from_events on 'running' (D-7).
                    progress.step(source_uid, generation=gen, cited=len(cited))

            # Accumulate the in-walk citation frequency across generations (D-5):
            # the SAME counts that rank the frontier below, kept per work_id so the
            # acquire pass can value-order non-seeds. In-memory hint only — never
            # serialized to the (pinned) walk manifest section.
            for cited_uid, count in next_counts.items():
                report.citation_counts[cited_uid] = (
                    report.citation_counts.get(cited_uid, 0) + count
                )

            for cited_uid in next_counts:
                if cited_uid not in all_nodes:
                    report.discovered += 1

            ranked = sorted(next_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            next_frontier: list[str] = []
            for cited_uid, _count in ranked:
                if cited_uid in all_nodes:
                    continue
                next_frontier.append(cited_uid)
                if len(next_frontier) >= per_gen_cap:
                    break
            # Every cited target is a NODE (its work row exists); only the *frontier*
            # we expand next is capped.
            for cited_uid in next_counts:
                all_nodes.add(cited_uid)

            report.frontier_sizes.append(len(next_frontier))
            _dump_frontier(frontier_dir, gen, next_frontier, ranked)
            frontier = next_frontier

        session.commit()

        # Residue counter (ch8): computed ONCE post-walk — works that resolved a
        # W-id but hold no title (a stub whose enrichment was lost to a provider
        # hang, then negative-cached). Nonzero here is the manifest's signal
        # that the corpus is NOT fully titled; the ch9 backfill is its repair.
        report.untitled_after_walk = int(
            session.exec(
                select(func.count())
                .select_from(Work)
                .where(Work.openalex_id.is_not(None))
                .where(or_(Work.canonical_title.is_(None), Work.canonical_title == ""))
            ).one()
        )

    report.per_provider_calls = dict(getattr(chain, "call_counts", {}))
    from .. import run as run_mod

    run_mod.update_manifest(h.slug, run_id, {"walk": report.to_section()}, root=h.root)
    return report
