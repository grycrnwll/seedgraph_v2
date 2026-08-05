"""Acquisition orchestration (§6.4) — the surface the ``corpus`` CLI calls.

``acquire_corpus`` / ``manual_upload`` / ``run_corpus`` thread ONE ``run_id`` from
``run.ensure_run`` and write each stage's DISJOINT top-level manifest section ONCE
via ``run.update_manifest`` (shallow section-keyed atomic replacement; D5 — NO
deep-merge, NO cross-stage accumulation; ``resolution`` / ``walk`` / ``acquisition``
each owned by exactly one stage). Content ingest/convert go through phase_1
``ingest_file`` / ``convert_source_file`` (no bespoke hashing/storage — §6.4). The
resolved ``access_class`` is mirrored onto ``project_documents.access_status``
(decision 30 / r2-1 §6).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from sqlmodel import Session, select

from ..cache.convert import convert_source_file
from ..cache.ingest import ingest_file
from ..db.project_models import ProjectDocument, Work, WorkSourceFile
from ..vocab import AccessClass, AcquisitionMethod
from . import bridge as bridge_mod
from .backfill import backfill_titles
from .fetch import fetch_oa
from .resolve import resolve_corpus
from .walk import walk_corpus

if TYPE_CHECKING:
    from ..progress import Progress
    from ..project.service import ProjectHandle
    from ..providers.base import ProviderChain


def corpus_io(root: Path | None, run_id: str | None):
    """Build ``(chain, http_client, marker_backend)`` for a corpus run.

    ``SEEDGRAPH_FAKE_PROVIDERS`` (mirrors ``SEEDGRAPH_FAKE_MARKER``) selects a
    deterministic OFFLINE chain + fake OA transport + fake Marker backend so the CLI
    round-trips with no network and no LLM (the build-plan VERIFY step). Otherwise a
    real ``ProviderChain`` is built from the global config.
    """
    from ..providers.base import ProviderChain

    if os.environ.get("SEEDGRAPH_FAKE_PROVIDERS"):
        from ..cache.marker_backend import FakeMarkerBackend
        from ..providers.fake import FakeProvider, fake_http_client

        chain = ProviderChain([FakeProvider()], cache_engine=root, run_id=run_id)
        return chain, fake_http_client(), FakeMarkerBackend()

    from ..config.loader import load_global_config

    cfg = load_global_config(root)
    chain = ProviderChain.from_config(cfg, cache_engine=root, run_id=run_id)
    return chain, None, None


@dataclass
class AcquireReport:
    """Acquire-stage counters — written ONCE as the disjoint ``acquisition`` manifest
    section (D5). ``state_summary`` tallies ``vocab.ACQUISITION_STATES``;
    ``access_class_summary`` tallies the resolved access classes.
    ``skipped_budget`` counts the targets a ``max_papers``/``budget_seconds`` budget
    cut — accounted, never silently skipped (D-6; additive section key)."""

    acquired: int = 0
    already_cached: int = 0
    requires_upload: int = 0
    failed: int = 0
    skipped_budget: int = 0
    access_class_summary: dict = field(default_factory=dict)
    state_summary: dict = field(default_factory=dict)

    def to_section(self) -> dict:
        return {
            "acquired": self.acquired,
            "already_cached": self.already_cached,
            "requires_upload": self.requires_upload,
            "failed": self.failed,
            "skipped_budget": self.skipped_budget,
            "access_class_summary": dict(self.access_class_summary),
            "state_summary": dict(self.state_summary),
        }


def _record_from_work(work: Work) -> dict:
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


def _bump(summary: dict, key: str) -> None:
    summary[key] = summary.get(key, 0) + 1


async def _acquire_async(
    h: "ProjectHandle",
    chain: "ProviderChain",
    *,
    run_id: str,
    statuses: tuple[str, ...] = ("included", "metadata_only"),
    promote: bool = True,
    cache_root: "Path | str | None" = None,
    http_client=None,
    backend=None,
    max_papers: "int | None" = None,
    budget_seconds: "float | None" = None,
    order_hint: "dict | None" = None,
    progress: "Progress | None" = None,
) -> AcquireReport:
    report = AcquireReport()

    with Session(h.engine, expire_on_commit=False) as session:
        results = session.exec(
            select(Work, ProjectDocument)
            .join(ProjectDocument, ProjectDocument.work_id == Work.work_id)
            .where(ProjectDocument.inclusion_status.in_(tuple(statuses)))
            .order_by(Work.created_at)
        ).all()
        targets = [(w.work_id, _record_from_work(w), bool(d.is_seed)) for w, d in results]

    if order_hint is not None:
        # Value-ordered pass (D-5): seeds first, then descending in-walk citation
        # frequency (the walk's accumulated frontier-ranking counts, threaded by
        # run_corpus), deterministic work_id tie-break — a capped/budgeted pass
        # spends its attempts on the highest-value papers. No hint (standalone
        # `corpus acquire`) keeps today's created_at order untouched — no
        # in-degree fallback (cut as over-engineering, D-5).
        targets.sort(key=lambda t: (not t[2], -order_hint.get(t[0], 0), t[0]))

    if progress is not None:
        # This pass owns the true target count — the caller constructs Progress
        # BEFORE the select, so its total is a placeholder we reconcile here (D-7).
        progress.total = len(targets)

    def _step(work_id: str) -> None:
        # One frame per target, detail=work_id, carrying the running outcome
        # counters (D-7). With emit=None (the CLI verbs) this is stdout-only —
        # no non-terminal 'progress' event ever lands in events.jsonl, so a
        # CLI run can never strand _status_from_events on 'running'; the
        # planner's emit-backed Progress streams the same frame to the run page.
        if progress is not None:
            progress.step(
                work_id,
                acquired=report.acquired,
                failed=report.failed,
                requires_upload=report.requires_upload,
                skipped_budget=report.skipped_budget,
            )

    attempted = 0
    budget_exhausted = False
    start = time.monotonic()
    for work_id, record, _is_seed in targets:
        # Budget enforcement (D-6 — v1 pipeline.py:1143 semantics ported verbatim):
        # `is not None` (not truthiness) so max_papers=0 / budget_seconds=0.0 mean
        # "attempt nothing"; on expiry the remainder is COUNTED skipped_budget,
        # never silently skipped. Defaults None/None leave behavior unchanged.
        # ponytail: an already-bridged target still consumes an attempt (v1 parity —
        # the bound is loop iterations, not fetches); exempting cached skips from
        # the budget is the ceiling if re-run ergonomics ever need it.
        if not budget_exhausted and (
            (max_papers is not None and attempted >= max_papers)
            or (budget_seconds is not None and (time.monotonic() - start) >= budget_seconds)
        ):
            budget_exhausted = True
        if budget_exhausted:
            report.skipped_budget += 1
            _step(work_id)
            continue
        attempted += 1

        # 1. Skip if already bridged + reconciles ok (idempotent; criterion 7).
        with Session(h.engine, expire_on_commit=False) as session:
            if bridge_mod.resolve_work_markdown(session, work_id=work_id) is not None:
                state = bridge_mod.derive_acquisition_state(session, cache_root, work_id)
            else:
                state = None
        if state == "already_cached_local":
            report.already_cached += 1
            _bump(report.state_summary, state)
            _step(work_id)
            continue

        # 2.+3. Build the provider record + fetch an OA PDF (OA-only; never paywall).
        result = await fetch_oa(record, chain=chain, client=http_client)
        if result.status == "failed":
            # TRANSIENT acquisition failure (every tried candidate transport-errored,
            # or the per-paper deadline expired) — honest 'failed' accounting, NOT a
            # requires_upload mislabel (D-4). No bridge row is written either way,
            # so a re-run naturally retries; fetch_oa RETURNS (never raises) so one
            # hostile host cannot abort the whole corpus pass.
            report.failed += 1
            _bump(report.state_summary, "failed")
            _step(work_id)
            continue
        if result.status != "downloaded" or not result.content:
            # No OA copy -> metadata-only stub + manual-upload invite (no file, no raise).
            with Session(h.engine, expire_on_commit=False) as session:
                derived = bridge_mod.derive_acquisition_state(session, cache_root, work_id)
            report.requires_upload += 1
            _bump(report.state_summary, derived)
            _step(work_id)
            continue

        # 4. Ingest the bytes via phase_1 (SHA-256 + dedup; no bespoke hashing).
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp.write(result.content)
            tmp.close()
            src = ingest_file(
                Path(tmp.name),
                access_class=AccessClass.open_access,
                acquisition_method=AcquisitionMethod.open_access_fetch,
                root=cache_root,
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        # 5. Convert to markdown via phase_1 (Marker; dedup on fingerprint).
        from ..cache.convert import ConversionError

        try:
            md = convert_source_file(src.source_file_id, backend=backend, root=cache_root)

            # 6.+7. Write the single bridge row (upsert on work_id) + mirror access_status.
            with Session(h.engine, expire_on_commit=False) as session:
                bridge_mod.write_bridge(
                    session,
                    work_id=work_id,
                    source_file_id=src.source_file_id,
                    file_hash=src.file_hash,
                    markdown_id=md.markdown_id,
                    markdown_hash=md.markdown_hash,
                    acquisition_method="open_access_fetch",
                )
                doc = session.get(ProjectDocument, work_id)
                if doc is not None:
                    doc.access_status = AccessClass.open_access.value
                    # A fetched+converted+bridged OA work is now part of the local corpus.
                    if promote and doc.inclusion_status == "metadata_only":
                        doc.inclusion_status = "included"
                        doc.inclusion_reason = "provided_pdf"
                    session.add(doc)
                session.commit()
        except ConversionError:
            # Marker produced garbage/empty markdown (the validate_markdown gate). A
            # single bad OA PDF must NOT abort the whole corpus pass — same rationale
            # as fetch_oa RETURNING (never raising) on a hostile host above. Count it
            # 'failed' and move on; no bridge row is written, so a re-run retries, and
            # the already-ingested source file staying in cache is fine (dedup handles
            # it). Kept specific to ConversionError so unrelated bugs still surface.
            report.failed += 1
            _bump(report.state_summary, "failed")
            _step(work_id)
            continue

        report.acquired += 1
        _bump(report.access_class_summary, AccessClass.open_access.value)
        _bump(report.state_summary, "already_cached_local")
        _step(work_id)

    from .. import run as run_mod

    run_mod.update_manifest(h.slug, run_id, {"acquisition": report.to_section()}, root=h.root)
    return report


def acquire_corpus(
    h: "ProjectHandle",
    chain: "ProviderChain",
    *,
    run_id: str,
    statuses: tuple[str, ...] = ("included", "metadata_only"),
    promote: bool = True,
    cache_root: "Path | None" = None,
    http_client=None,
    backend=None,
    max_papers: "int | None" = None,
    budget_seconds: "float | None" = None,
    progress: "Progress | None" = None,
) -> AcquireReport:
    """Acquire OA full text for every target work lacking a reconciling bridge row
    (§6.4 load-bearing algorithm).

    Per paper: skip if already bridged + reconciles ``ok`` (idempotent); else
    ``fetch_oa`` -> ``ingest_file(access_class=open_access,
    acquisition_method='open_access_fetch')`` -> ``convert_source_file`` ->
    ``write_bridge(+markdown)``; no-OA/SSRN -> stub + manual-upload invite (no file,
    no raise). Mirrors ``access_status='open_access'``; ``promote`` (default True)
    flips a fetched+converted ``metadata_only -> included``
    (``inclusion_reason='provided_pdf'``); ``promote=False`` opts out. Writes the
    single ``acquisition`` section once (D5).

    ``http_client`` (additive) is an injectable ``httpx.AsyncClient`` for offline
    tests; ``backend`` (additive) is the phase_1 Marker backend (tests pass a
    ``FakeMarkerBackend``). ``max_papers`` / ``budget_seconds`` (additive, D-6)
    bound the pass — ``0`` means "attempt nothing", ``None`` (default) means
    unbounded; the cut remainder is counted ``skipped_budget``. Standalone
    acquire runs in today's ``created_at`` order (no ``order_hint`` — D-5).
    ``progress`` (additive, D-7) steps once per target with ``detail=work_id``
    — ``emit=None`` (the CLI) keeps it stdout-only; ``None`` disables it.
    Sync wrapper over the async core (``asyncio.run``).
    """
    return asyncio.run(
        _acquire_async(
            h, chain, run_id=run_id, statuses=statuses, promote=promote,
            cache_root=cache_root, http_client=http_client, backend=backend,
            max_papers=max_papers, budget_seconds=budget_seconds,
            progress=progress,
        )
    )


def ingest_upload(pdf_path, *, cache_root=None):
    """Ingest a user-supplied PDF (fast: hash/dedup/store). Returns the SourceFile.

    The fast front-half of :func:`manual_upload`; the slow Marker convert is the
    back-half (:func:`convert_and_bridge`). The web upload route runs this inline and
    enqueues the convert; the CLI ``corpus upload`` still runs both synchronously via
    :func:`manual_upload`."""
    return ingest_file(
        Path(pdf_path),
        access_class=AccessClass.user_supplied_private,
        acquisition_method=AcquisitionMethod.upload,
        root=cache_root,
    )


def convert_and_bridge(
    h: "ProjectHandle",
    *,
    work_id: str,
    source_file_id: str,
    file_hash: str,
    cache_root: "Path | None" = None,
    backend=None,
    promote: bool = True,
) -> "WorkSourceFile":
    """Convert an already-ingested source file to markdown and bridge it to the work
    (the slow back-half of :func:`manual_upload`). Sets
    ``access_status='user_supplied_private'``. Converted no-LLM by default.

    ``promote`` (default True): a PROVIDED work (uploaded PDF now converted+bridged to
    markdown) joins the local corpus — ``metadata_only -> included`` with
    ``inclusion_reason='provided_pdf'``. Pass ``promote=False`` to keep the prior
    membership. ``included`` / ``excluded`` works are left untouched (only the
    walk-frontier ``metadata_only`` default is promoted)."""
    md = convert_source_file(source_file_id, backend=backend, root=cache_root)
    with Session(h.engine, expire_on_commit=False) as session:
        bridge_mod.write_bridge(
            session,
            work_id=work_id,
            source_file_id=source_file_id,
            file_hash=file_hash,
            markdown_id=md.markdown_id,
            markdown_hash=md.markdown_hash,
            acquisition_method="manual_upload",
        )
        doc = session.get(ProjectDocument, work_id)
        if doc is not None:
            doc.access_status = AccessClass.user_supplied_private.value
            if promote and doc.inclusion_status == "metadata_only":
                doc.inclusion_status = "included"
                doc.inclusion_reason = "provided_pdf"
            session.add(doc)
        session.commit()
        row = session.exec(
            select(WorkSourceFile).where(WorkSourceFile.work_id == work_id)
        ).first()
        if row is not None:
            session.expunge(row)
    return row


def manual_upload(
    h: "ProjectHandle",
    *,
    work_id: str,
    pdf_path,
    cache_root: "Path | None" = None,
    backend=None,
    promote: bool = True,
) -> "WorkSourceFile":
    """Lawful manual upload of a user-supplied PDF (§6.4; doc 01 §7).

    Steps 4-7 of ``acquire_corpus`` with ``access_class=user_supplied_private`` and
    ``acquisition_method='manual_upload'``: ``ingest_file`` -> ``convert_source_file``
    -> ``write_bridge`` -> set ``project_documents.access_status='user_supplied_private'``.
    Converted no-LLM by default (phase_1 content-access gate). Composes
    :func:`ingest_upload` + :func:`convert_and_bridge` (the web route runs those two
    halves separately, with the convert on a background queue). ``promote`` (default
    True) promotes a provided ``metadata_only`` work to ``included`` — see
    :func:`convert_and_bridge`; ``promote=False`` opts out."""
    src = ingest_upload(pdf_path, cache_root=cache_root)
    return convert_and_bridge(
        h,
        work_id=work_id,
        source_file_id=src.source_file_id,
        file_hash=src.file_hash,
        cache_root=cache_root,
        backend=backend,
        promote=promote,
    )


def import_markdown_and_bridge(
    h: "ProjectHandle",
    *,
    work_id: str,
    pdf_path,
    markdown_path,
    cache_root: "Path | None" = None,
    promote: bool = True,
) -> "WorkSourceFile":
    """Bridge a work to an EXTERNALLY-converted markdown file WITHOUT running Marker.

    The GPU-free sibling of :func:`manual_upload`: it ingests the source PDF for
    provenance (``access_class=user_supplied_private``,
    ``acquisition_method='upload'`` — same as :func:`ingest_upload`), stores the
    PROVIDED markdown through :func:`cache.convert.import_external_markdown` (an honest
    ``conversion_runs`` row with ``converter_name='external_import'`` + a
    ``markdown_documents`` row indistinguishable in shape from a converted one — no
    Marker, no GPU), then bridges + promotes exactly like :func:`convert_and_bridge`
    (``access_status='user_supplied_private'``; a ``metadata_only`` work flips to
    ``included`` with ``inclusion_reason='provided_pdf'`` when ``promote``).

    Idempotent: a work already bridged to markdown is left untouched (mirrors the
    pre-convert idempotency in :func:`ingest_one_pdf`) — its existing bridge row is
    returned and no second import runs."""
    from ..cache.convert import import_external_markdown

    # Idempotency: a work already bridged to markdown short-circuits (no re-import).
    with Session(h.engine, expire_on_commit=False) as session:
        if bridge_mod.resolve_work_markdown(session, work_id=work_id) is not None:
            row = session.exec(
                select(WorkSourceFile).where(WorkSourceFile.work_id == work_id)
            ).first()
            if row is not None:
                session.expunge(row)
            return row

    # 1. Ingest the PDF for provenance (user_supplied_private / upload).
    src = ingest_upload(pdf_path, cache_root=cache_root)

    # 2. Store the PROVIDED markdown WITHOUT Marker (external_import provenance).
    markdown_text = Path(markdown_path).read_text(encoding="utf-8")
    md = import_external_markdown(
        src.source_file_id, markdown_text=markdown_text, root=cache_root
    )

    # 3. Bridge + promote (mirrors convert_and_bridge exactly).
    with Session(h.engine, expire_on_commit=False) as session:
        bridge_mod.write_bridge(
            session,
            work_id=work_id,
            source_file_id=src.source_file_id,
            file_hash=src.file_hash,
            markdown_id=md.markdown_id,
            markdown_hash=md.markdown_hash,
            acquisition_method="manual_upload",
        )
        doc = session.get(ProjectDocument, work_id)
        if doc is not None:
            doc.access_status = AccessClass.user_supplied_private.value
            if promote and doc.inclusion_status == "metadata_only":
                doc.inclusion_status = "included"
                doc.inclusion_reason = "provided_pdf"
            session.add(doc)
        session.commit()
        row = session.exec(
            select(WorkSourceFile).where(WorkSourceFile.work_id == work_id)
        ).first()
        if row is not None:
            session.expunge(row)
    return row


# ---------------------------------------------------------------------------
# Bulk folder ingest + deterministic auto-match (corpus ingest-folder / UI drop)
# ---------------------------------------------------------------------------

# First ATX '# ' H1 line — a deterministic (NO-LLM) title candidate.
_H1_TITLE_RE = re.compile(r"(?m)^#\s+(.+?)\s*$")


@dataclass
class IngestFolderReport:
    """Aggregate counters + per-PDF audit rows for ``corpus ingest-folder``.

    ``per_pdf`` is the ordered list of the per-PDF outcome dicts
    (:func:`ingest_one_pdf`); the scalar fields tally them for the CLI summary.
    """

    matched_doi: int = 0
    matched_arxiv: int = 0
    matched_title: int = 0
    promoted: int = 0
    unmatched_review: int = 0
    ambiguous_review: int = 0
    already_bridged: int = 0
    conversion_failed: int = 0
    ocr_recovered: int = 0
    skipped_non_pdf: int = 0
    per_pdf: list = field(default_factory=list)


def convert_with_ocr_fallback(
    source_file_id: str,
    *,
    backend=None,
    cache_root: "Path | str | None" = None,
) -> "tuple[MarkdownDocument, bool]":
    """Convert ``source_file_id`` to markdown, retrying with OCR on the garbage gate.

    Try the default (no-OCR) ``MarkerConfig()`` first. If ``convert_source_file``
    RAISES ``ConversionError`` (the ``validate_markdown`` garbage gate — empty output
    or word-like char ratio < 0.20, i.e. an embedded-text-layer OCR failure), retry
    ONCE with ``MarkerConfig(force_ocr=True)`` + ``force=True`` (``force_ocr``
    perturbs the conversion fingerprint, so this is a DISTINCT run — never deduped
    against the default). Returns ``(markdown, ocr_used)``; a second
    ``ConversionError`` on the OCR retry PROPAGATES (the caller records
    ``conversion_failed``). Stays no-LLM / local (force_ocr uses local Surya OCR,
    not an external service)."""
    from ..cache.convert import ConversionError, MarkerConfig, convert_source_file

    try:
        md = convert_source_file(
            source_file_id, config=MarkerConfig(), backend=backend, root=cache_root
        )
        return md, False
    except ConversionError:
        md = convert_source_file(
            source_file_id,
            config=MarkerConfig(force_ocr=True),
            backend=backend,
            force=True,
            root=cache_root,
        )
        return md, True


def _first_h1_title(markdown_head: str) -> "str | None":
    """First ATX '# ' H1 line as a deterministic title candidate.

    Rejects a heading that carries no alphabetic character (a page-number / OCR
    artifact) so it is never treated as a title."""
    m = _H1_TITLE_RE.search(markdown_head or "")
    if m is None:
        return None
    title = m.group(1).strip()
    if not re.search(r"[A-Za-z]", title):
        return None
    return title or None


def extract_identity(markdown_text: str) -> "dict[str, str | None]":
    """Deterministic (NO-LLM) identity from the markdown HEAD.

    Returns ``{'doi', 'arxiv', 'title'}`` — the paper's OWN identity, cover-region
    bounded (``metadata.HEAD_CHARS``) so a CITED work's id / heading from a later
    reference list is never picked up. ``doi`` / ``arxiv`` reuse
    ``metadata.deterministic_identifiers``; ``title`` is the first ``# `` H1."""
    from ..extraction import metadata

    head = (markdown_text or "")[: metadata.HEAD_CHARS]
    det = metadata.deterministic_identifiers(head)
    return {"doi": det.get("doi"), "arxiv": det.get("arxiv"), "title": _first_h1_title(head)}


def _pending_title_matches(session: "Session", title_hash_value: str) -> "list[Work]":
    """Works with ``title_hash == title_hash_value`` scoped to a pending inclusion
    status (``metadata_only`` / ``included``) — an ``excluded`` work is never a
    match target (edge case: no auto-promote of an excluded work)."""
    return list(
        session.exec(
            select(Work)
            .join(ProjectDocument, ProjectDocument.work_id == Work.work_id)
            .where(Work.title_hash == title_hash_value)
            .where(ProjectDocument.inclusion_status.in_(("metadata_only", "included")))
        ).all()
    )


def match_pending_work(
    session: "Session",
    *,
    doi: "str | None" = None,
    arxiv: "str | None" = None,
    title: "str | None" = None,
) -> "tuple[str | None, str]":
    """Match an extracted identity to ONE pending work: strong-id first, then title.

    Strong ids (``identity.match_works``) win: exactly one distinct work ->
    ``(work_id, 'doi'|'arxiv')``; >=2 -> ``(None, 'ambiguous')`` (never auto-merge
    across a cross-id collision). With no strong-id match, a ``title_hash`` lookup
    scoped to pending works: exactly one -> ``(work_id, 'title')``; >=2 ->
    ``(None, 'ambiguous')`` (a title collision NEVER auto-merges — identity rule).
    Otherwise ``(None, 'none')``."""
    from ..project import identity

    incoming: dict = {}
    if doi:
        incoming["doi"] = doi
    if arxiv:
        incoming["arxiv"] = arxiv

    distinct: list = []
    if incoming:
        seen: set = set()
        for w in identity.match_works(session, incoming):
            if w.work_id not in seen:
                seen.add(w.work_id)
                distinct.append(w)
    if len(distinct) == 1:
        return distinct[0].work_id, ("doi" if doi else "arxiv")
    if len(distinct) >= 2:
        return None, "ambiguous"

    if title:
        matches = _pending_title_matches(session, identity.title_hash(title))
        if len(matches) == 1:
            return matches[0].work_id, "title"
        if len(matches) >= 2:
            return None, "ambiguous"
    return None, "none"


def _match_candidates(
    session: "Session",
    *,
    doi: "str | None" = None,
    arxiv: "str | None" = None,
    title: "str | None" = None,
) -> "list[str]":
    """The candidate work_ids behind an ``ambiguous`` match (for the review payload)."""
    from ..project import identity

    incoming: dict = {}
    if doi:
        incoming["doi"] = doi
    if arxiv:
        incoming["arxiv"] = arxiv
    ids: list[str] = []
    if incoming:
        seen: set = set()
        for w in identity.match_works(session, incoming):
            if w.work_id not in seen:
                seen.add(w.work_id)
                ids.append(w.work_id)
    if len(ids) >= 2:
        return ids
    if title:
        return [w.work_id for w in _pending_title_matches(session, identity.title_hash(title))]
    return ids


def ingest_one_pdf(
    h: "ProjectHandle",
    pdf_path: "Path | str",
    *,
    cache_root: "Path | str | None" = None,
    backend=None,
    promote: bool = True,
    promote_unmatched: bool = False,
) -> dict:
    """Per-PDF auto-match pipeline shared by the folder loop and the web drop-zone.

    Steps: (a) ``ingest_file`` (fail-closed ``user_supplied_private`` /
    ``folder_import``); (b) pre-convert idempotency — a file already bridged (its
    ``file_hash`` present in ``work_source_files``) short-circuits to
    ``already_bridged`` (no second Marker run, no duplicate review item);
    (c) :func:`convert_with_ocr_fallback` (``conversion_failed`` on a second
    failure); (d) :func:`extract_identity` over the converted head; (e)
    :func:`match_pending_work`; (f) a matched work that is ALREADY bridged is left
    untouched (``already_bridged`` — never clobber a newer/other acquisition);
    (g) a unique match writes the single bridge row, mirrors
    ``access_status='user_supplied_private'``, and (``promote``) flips a
    ``metadata_only`` work to ``included`` (``inclusion_reason='provided_pdf'``);
    (h) a miss/ambiguous match either promotes to a NEW included work
    (``promote_unmatched``) or routes to ``review_queue`` as a typed
    ``unmatched_upload`` item.

    Returns a per-PDF outcome dict: ``{filename, outcome, work_id, matched_by,
    ocr_used, markdown_id, review_item_id}`` with ``outcome`` one of
    ``matched_doi|matched_arxiv|matched_title|promoted|unmatched_review|
    ambiguous_review|already_bridged|conversion_failed``. Deterministic + local — no
    LLM / network on this path."""
    from .. import cache_access
    from ..cache.convert import ConversionError
    from ..project import review as review_mod

    pdf_path = Path(pdf_path)
    filename = pdf_path.name
    outcome: dict = {
        "filename": filename,
        "outcome": None,
        "work_id": None,
        "matched_by": None,
        "ocr_used": False,
        "markdown_id": None,
        "review_item_id": None,
    }

    # (a) ingest bytes — fail-closed user_supplied_private (folder_import default).
    src = ingest_file(
        pdf_path,
        access_class=AccessClass.user_supplied_private,
        acquisition_method=AcquisitionMethod.folder_import,
        root=cache_root,
    )

    # (b) pre-convert idempotency: this exact file is already bridged -> skip Marker.
    with Session(h.engine, expire_on_commit=False) as session:
        existing = session.exec(
            select(WorkSourceFile).where(WorkSourceFile.file_hash == src.file_hash)
        ).first()
        if existing is not None:
            outcome["outcome"] = "already_bridged"
            outcome["work_id"] = existing.work_id
            outcome["markdown_id"] = existing.markdown_id
            return outcome

    # (c) convert (OCR-fallback on the validate garbage gate).
    try:
        md, ocr_used = convert_with_ocr_fallback(
            src.source_file_id, backend=backend, cache_root=cache_root
        )
    except ConversionError:
        outcome["outcome"] = "conversion_failed"
        return outcome
    outcome["ocr_used"] = ocr_used
    outcome["markdown_id"] = md.markdown_id

    # (d) read the converted head + extract the paper's OWN deterministic identity.
    cache_conn = cache_access.open_cache_ro(cache_root)
    try:
        mdrow = cache_access.read_markdown(cache_conn, cache_root, md.markdown_id)
    finally:
        cache_conn.close()
    ident = extract_identity(mdrow.text if mdrow is not None else "")

    # (e) match to a pending work (strong-id first, then title).
    with Session(h.engine, expire_on_commit=False) as session:
        work_id, kind = match_pending_work(
            session, doi=ident["doi"], arxiv=ident["arxiv"], title=ident["title"]
        )
        matched_already = (
            work_id is not None
            and bridge_mod.resolve_work_source(session, work_id=work_id) is not None
        )
        candidates = (
            _match_candidates(
                session, doi=ident["doi"], arxiv=ident["arxiv"], title=ident["title"]
            )
            if kind == "ambiguous"
            else []
        )

    # (f) a matched work already bridged -> never clobber a newer/other acquisition.
    if work_id is not None and matched_already:
        outcome["outcome"] = "already_bridged"
        outcome["work_id"] = work_id
        return outcome

    # (g) unique match -> bridge (+ optional promote to included).
    if work_id is not None:
        with Session(h.engine, expire_on_commit=False) as session:
            bridge_mod.write_bridge(
                session,
                work_id=work_id,
                source_file_id=src.source_file_id,
                file_hash=src.file_hash,
                markdown_id=md.markdown_id,
                markdown_hash=md.markdown_hash,
                acquisition_method="manual_upload",
            )
            doc = session.get(ProjectDocument, work_id)
            if doc is not None:
                doc.access_status = AccessClass.user_supplied_private.value
                if promote and doc.inclusion_status == "metadata_only":
                    doc.inclusion_status = "included"
                    doc.inclusion_reason = "provided_pdf"
                session.add(doc)
            session.commit()
        outcome["outcome"] = f"matched_{kind}"
        outcome["work_id"] = work_id
        outcome["matched_by"] = kind
        return outcome

    # (h) miss / ambiguous -> promote-to-new-work OR route to review.
    reason = "ambiguous_match" if kind == "ambiguous" else "no_match"
    if promote_unmatched:
        from ..project import service as project_service

        ids: dict = {}
        if ident["doi"]:
            ids["doi"] = ident["doi"]
        if ident["arxiv"]:
            ids["arxiv"] = ident["arxiv"]
        title = (
            ident["title"]
            or Path(filename).stem.replace("_", " ").replace("-", " ").strip()
            or filename
        )
        work = project_service.add_work(
            h,
            ids=ids or None,
            title=title,
            inclusion_status="included",
            inclusion_reason="provided_pdf",
        )
        with Session(h.engine, expire_on_commit=False) as session:
            bridge_mod.write_bridge(
                session,
                work_id=work.work_id,
                source_file_id=src.source_file_id,
                file_hash=src.file_hash,
                markdown_id=md.markdown_id,
                markdown_hash=md.markdown_hash,
                acquisition_method="manual_upload",
            )
            doc = session.get(ProjectDocument, work.work_id)
            if doc is not None:
                doc.access_status = AccessClass.user_supplied_private.value
                session.add(doc)
            session.commit()
        outcome["outcome"] = "promoted"
        outcome["work_id"] = work.work_id
        return outcome

    payload = {
        "kind": "unmatched_upload",
        "reason": reason,
        "source_file_id": src.source_file_id,
        "file_hash": src.file_hash,
        "markdown_id": md.markdown_id,
        "filename": filename,
        "doi": ident["doi"],
        "arxiv": ident["arxiv"],
        "extracted_title": ident["title"],
        "candidates": candidates,
    }
    with Session(h.engine, expire_on_commit=False) as session:
        item_id = review_mod.enqueue_in_session(
            session,
            "unmatched_upload",
            target_type="source_file",
            target_id=src.source_file_id,
            payload=payload,
        )
        session.commit()
    outcome["outcome"] = "ambiguous_review" if reason == "ambiguous_match" else "unmatched_review"
    outcome["review_item_id"] = item_id
    return outcome


def _tally_ingest(report: "IngestFolderReport", result: dict) -> None:
    """Fold one :func:`ingest_one_pdf` outcome into the running report."""
    counter = {
        "matched_doi": "matched_doi",
        "matched_arxiv": "matched_arxiv",
        "matched_title": "matched_title",
        "promoted": "promoted",
        "unmatched_review": "unmatched_review",
        "ambiguous_review": "ambiguous_review",
        "already_bridged": "already_bridged",
        "conversion_failed": "conversion_failed",
    }.get(result.get("outcome"))
    if counter is not None:
        setattr(report, counter, getattr(report, counter) + 1)
    if result.get("ocr_used"):
        report.ocr_recovered += 1


def ingest_folder(
    h: "ProjectHandle",
    folder: "Path | str",
    *,
    cache_root: "Path | str | None" = None,
    backend=None,
    promote: bool = True,
    promote_unmatched: bool = False,
    recursive: bool = False,
) -> IngestFolderReport:
    """Bulk-ingest a folder of PDFs, auto-matching each to a pending work.

    Loops ``*.pdf`` files (``rglob`` when ``recursive``; a non-``.pdf`` file is
    tallied ``skipped_non_pdf`` and skipped), calls :func:`ingest_one_pdf` per file,
    and returns an :class:`IngestFolderReport`. A missing / non-existent folder
    yields an empty report (no error); one bad PDF (conversion failure) never aborts
    the batch."""
    folder = Path(folder)
    report = IngestFolderReport()
    if not folder.is_dir():
        return report
    entries = sorted(p for p in (folder.rglob("*") if recursive else folder.iterdir()) if p.is_file())
    for pdf_path in entries:
        if pdf_path.suffix.lower() != ".pdf":
            report.skipped_non_pdf += 1
            report.per_pdf.append(
                {
                    "filename": pdf_path.name,
                    "outcome": "skipped_non_pdf",
                    "work_id": None,
                    "matched_by": None,
                    "ocr_used": False,
                    "markdown_id": None,
                    "review_item_id": None,
                }
            )
            continue
        result = ingest_one_pdf(
            h,
            pdf_path,
            cache_root=cache_root,
            backend=backend,
            promote=promote,
            promote_unmatched=promote_unmatched,
        )
        report.per_pdf.append(result)
        _tally_ingest(report, result)
    return report


def corpus_rows(
    h: "ProjectHandle",
    *,
    statuses: "tuple[str, ...] | None" = None,
    cache_root: "Path | str | None" = None,
) -> list[dict]:
    """Return the dense corpus table shared by ``corpus status`` (CLI) and the UI.

    Lifts the per-work projection the CLI ``corpus status`` command rendered inline
    (``work_id | title | inclusion_status | access_status | acquisition_state |
    has_markdown``) and adds the boolean enrichment flags the corpus screen needs:
    ``has_citations`` / ``has_extraction`` / ``has_concepts`` / ``has_review``. Flag
    sets are computed once (set-based) over project.db; the per-work
    ``acquisition_state`` + ``has_markdown`` still resolve through the cross-DB bridge
    (``derive_acquisition_state`` / ``resolve_work_markdown``) exactly as the CLI did.
    Read-only; no full text is included here (the gated drawer serves text).
    """
    from ..db.adapter import raw_conn
    from .bridge import derive_acquisition_state, resolve_work_markdown

    wanted = tuple(statuses) if statuses else ("included", "metadata_only", "excluded")
    rows: list[dict] = []
    with Session(h.engine, expire_on_commit=False) as session:
        conn = raw_conn(session)
        cited: set[str] = set()
        for src, tgt in conn.execute(
            "SELECT source_work_id, target_work_id FROM citation_edges"
        ).fetchall():
            cited.add(src)
            cited.add(tgt)
        extracted = {
            r[0] for r in conn.execute("SELECT DISTINCT work_id FROM extracted_claims").fetchall()
        }
        concepted = {
            r[0] for r in conn.execute("SELECT DISTINCT work_id FROM claim_concepts").fetchall()
        }
        reviewed = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT lo.work_id FROM review_queue rq "
                "JOIN lens_outputs lo ON lo.lens_output_id = rq.target_id "
                "WHERE rq.status = 'open'"
            ).fetchall()
        }

        results = session.exec(
            select(Work, ProjectDocument)
            .join(ProjectDocument, ProjectDocument.work_id == Work.work_id)
            .where(ProjectDocument.inclusion_status.in_(wanted))
            .order_by(Work.created_at)
        ).all()
        for work, doc in results:
            wid = work.work_id
            state = derive_acquisition_state(session, cache_root, wid)
            has_md = resolve_work_markdown(session, work_id=wid) is not None
            rows.append(
                {
                    "work_id": wid,
                    "title": work.canonical_title,
                    "year": work.year,
                    "inclusion_status": doc.inclusion_status,
                    "is_seed": bool(doc.is_seed),
                    "access_status": doc.access_status,
                    # Build D ch10: the OA color (triage badge; the ch12 frontier
                    # pane's "OA exists" vs "genuinely paywalled" split). Additive key.
                    "oa_status": work.oa_status,
                    "acquisition_state": state,
                    "has_markdown": has_md,
                    "has_citations": wid in cited,
                    "has_extraction": wid in extracted,
                    "has_concepts": wid in concepted,
                    "has_review": wid in reviewed,
                }
            )
    return rows


def run_corpus(
    h: "ProjectHandle",
    chain: "ProviderChain",
    *,
    depth: int = 2,
    per_gen_cap: int = 50,
    root: "Path | None" = None,
    http_client=None,
    backend=None,
    promote: bool = True,
    max_papers: "int | None" = None,
    budget_seconds: "float | None" = None,
    walk_progress: "Progress | None" = None,
    acquire_progress: "Progress | None" = None,
) -> dict:
    """``corpus run`` umbrella (§6.1): ``ensure_run`` -> resolve -> walk ->
    title-backfill -> acquire under ONE ``run_id``; each stage writes its OWN
    disjoint top-level manifest section once (D5). Returns
    ``{run_id, resolution, walk, title_backfill, acquisition}`` reports
    (``title_backfill`` is ``None`` when its fail-soft pass errored).

    Threads the walk's in-memory ``citation_counts`` into the acquire pass as its
    ``order_hint`` (D-5: seeds first, then in-walk citation frequency) and passes
    ``max_papers`` / ``budget_seconds`` through (D-6 budget semantics).
    ``walk_progress`` / ``acquire_progress`` (additive, D-7) are per-stage
    :class:`Progress` handles — the planner passes emit-backed ones, the CLI
    stdout-only ones, ``None`` (default) disables per-item frames.

    The post-walk title backfill (§5.10, Build D ch9) runs FAIL-SOFT between
    walk and acquire: an exception there logs and continues (the pass is
    re-runnable standalone via ``corpus backfill-titles``), and its four
    counters land once as the small additive ``title_backfill`` section.
    """
    from .. import paths
    from ..run import ensure_run, update_manifest

    run_id = ensure_run(h.slug, root=h.root)
    frontier_dir = paths.project_dir(h.slug, h.root) / "runs" / run_id / "frontier"

    async def _run_async() -> dict:
        resolution = await resolve_corpus(h, chain, run_id=run_id)
        walk = await walk_corpus(
            h, chain, run_id=run_id, depth=depth, per_gen_cap=per_gen_cap,
            frontier_dir=frontier_dir, progress=walk_progress,
        )
        # Post-walk title backfill (§5.10, ch9): a read-repair pass between walk
        # and acquire, FAIL-SOFT — a provider outage here must not cost the run
        # its acquire stage. Counters are written ONCE as the additive
        # ``title_backfill`` section this pass solely owns (D5).
        title_backfill = None
        try:
            title_backfill = await backfill_titles(h, chain)
            update_manifest(
                h.slug, run_id,
                {"title_backfill": title_backfill.to_section()},
                root=h.root,
            )
        except Exception:  # noqa: BLE001 - fail-soft: log and continue to acquire
            logging.getLogger(__name__).warning(
                "post-walk title backfill failed; continuing to acquire",
                exc_info=True,
            )
        acquisition = await _acquire_async(
            h, chain, run_id=run_id, promote=promote, cache_root=root,
            http_client=http_client, backend=backend,
            max_papers=max_papers, budget_seconds=budget_seconds,
            order_hint=walk.citation_counts, progress=acquire_progress,
        )
        return {
            "run_id": run_id,
            "resolution": resolution,
            "walk": walk,
            "title_backfill": title_backfill,
            "acquisition": acquisition,
        }

    return asyncio.run(_run_async())
