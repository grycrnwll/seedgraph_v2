"""Run planner (Track 2 Stage C) — job preview + launch for every task type.

:func:`plan_job` is a pure read that previews a prospective build job: the affected
works, each work's resolved route/profile (via ``llm.routing.resolve_route`` ->
local-vs-API destination + a "content may leave this machine" flag from
``Route.external_full_text``), an estimated token/cost projection (executor
``preflight`` + ``llm.cost.preflight_estimate``), the current monthly spend, and the
budget confirmation gate (``require_confirmation_above_usd``). It never dispatches a
model and never writes — safe offline/keyless.

:func:`launch_planned_job` dispatches the matching service call through
:func:`web.jobs.launch_job`, which mints a fresh ``run_id``, runs the body on a
daemon thread with its OWN ``ProjectHandle``, and guarantees a terminal event. The
LLM-shaped tasks (extraction / chunked / lenses / concepts-with-profile) run for
real through the Track-1 executor and degrade honestly (``skipped_no_llm`` etc.)
when no backend is configured — they never crash the worker.

This module REUSES the service layer (``project.service``, ``acquisition.service``,
``extraction.runner``, ``lenses.runner``, ``semantic``) — it does not re-implement
CLI logic. The corpus/graph run-view helpers it shares with the CLI live in their
own homes (``acquisition.service.corpus_io``, ``graph.run_view.build_and_export``,
``db.connection.connect_project_raw``, ``cache_access.open_cache_ro``), so neither
adapter reaches into the other.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..cache_access import open_cache_ro
from ..errors import SeedgraphError
from ..progress import Progress

# --- preview value objects ---------------------------------------------------


@dataclass
class WorkPreview:
    """One affected work's resolved routing + cost projection (preview only)."""

    work_id: str
    title: Optional[str]
    access_class: Optional[str]
    profile_id: Optional[str]
    provider: Optional[str]
    destination: str  # local | external | deterministic | blocked
    external_full_text: bool  # content may leave this machine for THIS work
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float
    note: Optional[str] = None


@dataclass
class JobPlan:
    """A previewed (un-launched) build job over one project + task type."""

    slug: str
    task: str
    task_type: Optional[str]  # the LLM route task (None for deterministic tasks)
    is_llm_task: bool
    label: str
    works: list[WorkPreview]
    affected_count: int
    route_profile_id: Optional[str]
    route_provider: Optional[str]
    route_destination: str
    external_full_text: bool  # ANY affected work may leave the machine
    needs_llm_backend: bool  # the resolved route has no usable profile
    est_total_input_tokens: int
    est_total_output_tokens: int
    est_total_cost_usd: float
    monthly_spend_usd: float
    monthly_soft_limit_usd: Optional[float]
    over_monthly_soft_limit: bool
    require_confirmation_above_usd: Optional[float]
    requires_confirmation: bool
    flags: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: Conversion task only (Build D ch13): the closed-form no-network acquisition
    #: budget (``preview_acquisition`` dict — papers / API calls / disk). ``None``
    #: for every other task; the template falls back to the token/cost lines.
    acquisition_preview: Optional[dict] = None


# --- worklists (reuse the service layer; small read queries only) ------------


def _query_pairs(h, sql: str) -> list[tuple[str, Optional[str]]]:
    conn = sqlite3.connect(str(h.db_path))
    try:
        return [(row[0], row[1]) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def _titles_for(h, work_ids: list[str]) -> list[tuple[str, Optional[str]]]:
    conn = sqlite3.connect(str(h.db_path))
    try:
        out: list[tuple[str, Optional[str]]] = []
        for wid in work_ids:
            row = conn.execute(
                "SELECT canonical_title FROM works WHERE work_id = ?", (wid,)
            ).fetchone()
            out.append((wid, row[0] if row is not None else None))
        return out
    finally:
        conn.close()


def _included_pairs(h, cache_root) -> list[tuple[str, Optional[str]]]:
    return _query_pairs(
        h,
        "SELECT w.work_id, w.canonical_title FROM works w "
        "JOIN project_documents d ON d.work_id = w.work_id "
        "WHERE d.inclusion_status = 'included' ORDER BY w.created_at",
    )


def _claim_pairs(h, cache_root) -> list[tuple[str, Optional[str]]]:
    return _query_pairs(
        h,
        "SELECT DISTINCT w.work_id, w.canonical_title FROM works w "
        "JOIN extracted_claims c ON c.work_id = w.work_id ORDER BY w.work_id",
    )


def _markdown_pairs(h, cache_root) -> list[tuple[str, Optional[str]]]:
    """Included works that have a bridged markdown (the extraction/sections worklist)."""
    from sqlmodel import Session

    from ..acquisition.bridge import resolve_work_markdown

    out: list[tuple[str, Optional[str]]] = []
    with Session(h.engine) as session:
        for wid, title in _included_pairs(h, cache_root):
            if resolve_work_markdown(session, work_id=wid) is not None:
                out.append((wid, title))
    return out


def _oversize_pairs(h, cache_root) -> list[tuple[str, Optional[str]]]:
    """phase_4 ``skipped_oversize`` works with no surviving note (the chunked worklist)."""
    return _query_pairs(
        h,
        "SELECT DISTINCT r.work_id, w.canonical_title FROM extraction_runs r "
        "JOIN works w ON w.work_id = r.work_id "
        "WHERE r.run_status = 'skipped_oversize' "
        "AND NOT EXISTS (SELECT 1 FROM structured_notes n WHERE n.work_id = r.work_id) "
        "ORDER BY r.work_id",
    )


# --- per-work routing + token estimate ---------------------------------------


def _access_and_tokens(session, cache_conn, cache_root, work_id: str) -> tuple[str, int]:
    """Best-effort source access-class + input-token estimate for ``work_id``.

    Resolves the work's bridged markdown; reads its access class + text from the
    read-only cache. Unresolved markdown ⇒ fail-closed ``user_supplied_private`` /
    zero tokens (the preview never blocks on a pruned cache)."""
    from .. import cache_access
    from ..acquisition.bridge import resolve_work_markdown
    from ..extraction.runner import resolve_source_access_class
    from ..llm.tokens import estimate_tokens

    access_class = "user_supplied_private"
    tokens = 0
    try:
        resolved = resolve_work_markdown(session, work_id=work_id)
    except Exception:  # noqa: BLE001 — preview is best-effort
        resolved = None
    if resolved is not None and cache_conn is not None:
        markdown_id, _hash = resolved
        try:
            access_class = resolve_source_access_class(cache_conn, markdown_id)
        except Exception:  # noqa: BLE001
            access_class = "user_supplied_private"
        try:
            md = cache_access.read_markdown(cache_conn, cache_root, markdown_id)
            if md is not None:
                tokens = estimate_tokens(md.text)
        except Exception:  # noqa: BLE001
            tokens = 0
    return access_class, tokens


# --- task registry -----------------------------------------------------------


@dataclass
class _TaskSpec:
    label: str
    llm_task: Optional[str]
    phase: str
    worklist: Callable
    job: Callable  # (plan, flags) -> fn(emit, handle)


#: Conversion-job walk bounds. Module-level so the plan preview and the launched
#: job read the SAME values and cannot disagree (Build D ch13 / D-13). Deliberately
#: NOT planner-exposed controls yet ("expose acquisition in the UI" remains an
#: open backlog item; scope discipline).
_CONVERSION_DEPTH = 2
_CONVERSION_PER_GEN_CAP = 50


def _job_conversion(plan: JobPlan, flags: dict) -> Callable:
    def fn(emit, handle) -> None:
        from ..acquisition.service import run_corpus
        from ..acquisition.service import corpus_io

        emit("conversion", "resolve -> walk -> acquire -> convert")
        chain, client, backend = corpus_io(handle.root, emit.run_id)
        # Per-item observability (Build D ch6 / D-7): one emit-backed Progress per
        # stage, streaming default 'progress' frames through the run page's
        # existing events poller. Totals start at 0 placeholders — the walk grows
        # its total per generation; the acquire pass reconciles its total once
        # the target list is selected.
        walk_prog = Progress(0, "works", emit=emit, echo=False)
        acquire_prog = Progress(0, "works", emit=emit, echo=False)
        out = run_corpus(
            handle, chain, depth=_CONVERSION_DEPTH, per_gen_cap=_CONVERSION_PER_GEN_CAP,
            root=handle.root,
            http_client=client, backend=backend, promote=False,
            walk_progress=walk_prog, acquire_progress=acquire_prog,
        )
        emit(
            "conversion",
            f"resolved={out['resolution'].resolved} "
            f"walked={out['walk'].discovered} acquired={out['acquisition'].acquired}",
        )

    return fn


def _job_sections(plan: JobPlan, flags: dict) -> Callable:
    work_ids = [w.work_id for w in plan.works]

    def fn(emit, handle) -> None:
        from sqlmodel import Session

        from .. import cache_access
        from ..acquisition.bridge import resolve_work_markdown
        from ..db.adapter import raw_conn
        from ..sections.parser import parse_sections
        from ..sections.store import replace_sections

        cache_conn = cache_access.open_cache_ro(handle.root)
        total = 0
        prog = Progress(len(work_ids), "works", emit=emit, event="work", echo=False)
        try:
            with Session(handle.engine) as session:
                conn = raw_conn(session)
                for wid in work_ids:
                    resolved = resolve_work_markdown(session, work_id=wid)
                    if resolved is None:
                        prog.step(f"{wid}: no markdown", work_id=wid)
                        continue
                    mid, _h = resolved
                    md = cache_access.read_markdown(cache_conn, handle.root, mid)
                    if md is None:
                        prog.step(f"{wid}: markdown unresolvable", work_id=wid)
                        continue
                    sections = parse_sections(
                        md.text, markdown_id=mid, markdown_hash=md.markdown_hash,
                        source_file_id=md.source_file_id,
                        source_file_hash=md.source_file_hash, work_id=wid,
                    )
                    n = replace_sections(conn, mid, sections)
                    session.commit()
                    total += n
                    prog.step(f"{wid}: sections={n}", work_id=wid, sections=n)
        finally:
            cache_conn.close()
        emit("summary", f"{len(work_ids)} works, {total} sections")

    return fn


def _job_cite(plan: JobPlan, flags: dict) -> Callable:
    def fn(emit, handle) -> None:
        from ..cache_access import open_cache_ro
        from ..citation.project_edges import project_provider_edges
        from ..db.connection import connect_project_raw
        from ..graph.run_view import build_and_export

        run_id = emit.run_id
        emit("cite", "projecting provider edges (offline)")
        conn = connect_project_raw(handle.db_path)
        cache_conn = open_cache_ro(handle.root)
        try:
            result = project_provider_edges(conn, cache_conn, run_id=run_id)
        finally:
            conn.close()
            cache_conn.close()
        graph, out_dir = build_and_export(
            handle.slug, handle.root, handle.root, run_id, open_world=False
        )
        emit(
            "cite",
            f"edges={result['edges']} nodes={graph.number_of_nodes()}",
            edges=result["edges"],
        )

    return fn


def _job_concepts(plan: JobPlan, flags: dict) -> Callable:
    profile_id = flags.get("profile_id")

    def fn(emit, handle) -> None:
        from ..config.loader import load_project_config
        from ..semantic import build_semantic_overlay
        from ..semantic.export import update_graph_manifest

        run_id = emit.run_id
        prof = None
        cfg = None
        # Deterministic by default (offline-safe); routes through the executor only
        # when an explicit profile is chosen (then it degrades honestly if down).
        if profile_id:
            cfg = load_project_config(handle.slug, handle.root)
            prof = cfg.llm.profiles.get(profile_id)
        emit("concepts", "extract -> cluster -> guard -> merge")
        conn = sqlite3.connect(str(handle.db_path))
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            report = build_semantic_overlay(conn, run_id=run_id, profile=prof, config=cfg)
        finally:
            conn.close()
        update_graph_manifest(slug=handle.slug, run_id=run_id, report=report, root=handle.root)
        emit(
            "concepts",
            f"mode={report.concept_mode} concepts={report.concepts_written} "
            f"co_occurs={report.co_occurs_edges}",
            concepts=report.concepts_written,
            mode=report.concept_mode,
        )

    return fn


def _job_extraction(plan: JobPlan, flags: dict) -> Callable:
    work_ids = [w.work_id for w in plan.works]
    profile_id = flags.get("profile_id")
    force = bool(flags.get("force"))

    def fn(emit, handle) -> None:
        from sqlmodel import Session

        from .. import cache_access
        from ..config.loader import load_project_config
        from ..errors import ConfigError
        from ..extraction.runner import BudgetState, extract_note
        from ..extraction.schema import SCHEMA_ID

        cfg = load_project_config(handle.slug, handle.root)
        cache_conn = cache_access.open_cache_ro(handle.root)
        budget = BudgetState()
        n_ok = 0
        prog = Progress(len(work_ids), "works", emit=emit, event="work", echo=False)
        try:
            with Session(handle.engine) as session:
                for wid in work_ids:
                    try:
                        res = extract_note(
                            session, cache_conn, work_id=wid, schema_id=SCHEMA_ID,
                            profile_id=profile_id, force=force, budget_state=budget,
                            config=cfg, cache_root=handle.root,
                        )
                    except ConfigError as exc:
                        prog.step(f"{wid}: config_error: {exc}", level="error", work_id=wid)
                        continue
                    if res.run_status == "success":
                        n_ok += 1
                    prog.step(f"{wid}: {res.run_status}", work_id=wid, status=res.run_status)
        finally:
            cache_conn.close()
        emit("summary", f"{n_ok}/{len(work_ids)} extracted", ok=n_ok, total=len(work_ids))

    return fn


def _job_chunked(plan: JobPlan, flags: dict) -> Callable:
    work_ids = [w.work_id for w in plan.works]
    profile_id = flags.get("profile_id")
    force = bool(flags.get("force"))

    def fn(emit, handle) -> None:
        from sqlmodel import Session

        from .. import cache_access
        from ..config.loader import load_project_config
        from ..errors import ConfigError
        from ..extraction.chunked_runner import extract_note_chunked
        from ..extraction.runner import BudgetState
        from ..extraction.schema import SCHEMA_ID
        from ..run import update_manifest

        run_id = emit.run_id
        cfg = load_project_config(handle.slug, handle.root)
        cache_conn = cache_access.open_cache_ro(handle.root)
        budget = BudgetState()
        notes = 0
        prog = Progress(len(work_ids), "works", emit=emit, event="work", echo=False)
        try:
            with Session(handle.engine) as session:
                for wid in work_ids:
                    try:
                        res = extract_note_chunked(
                            session, cache_conn, work_id=wid, schema_id=SCHEMA_ID,
                            profile_id=profile_id, force=force, build_run_id=run_id,
                            budget_state=budget, config=cfg, cache_root=handle.root,
                        )
                    except ConfigError as exc:
                        prog.step(f"{wid}: config_error: {exc}", level="error", work_id=wid)
                        continue
                    if res.status == "success":
                        notes += 1
                    prog.step(
                        f"{wid}: {res.status} chunks={res.chunk_count}",
                        work_id=wid, status=res.status,
                    )
        finally:
            cache_conn.close()
        update_manifest(
            handle.slug, run_id,
            {"chunked_extraction": {"works": len(work_ids), "notes_written": notes, "run_id": run_id}},
            root=handle.root,
        )
        emit("summary", f"{notes}/{len(work_ids)} chunked notes", notes=notes)

    return fn


def _job_lenses(plan: JobPlan, flags: dict) -> Callable:
    work_ids = [w.work_id for w in plan.works]
    lens_id = flags.get("lens_id")
    force = bool(flags.get("force"))

    def fn(emit, handle) -> None:
        from sqlmodel import Session

        from ..config.loader import load_project_config
        from ..lenses import registry
        from ..lenses.runner import build_lens_router, included_lens_work_ids, run_lens
        from ..run import update_manifest

        run_id = emit.run_id
        if not lens_id:
            emit("lenses", "no lens_id supplied — nothing to run", level="error")
            return
        cfg = load_project_config(handle.slug, handle.root)
        project_dir = handle.root / "projects" / handle.slug
        targets = work_ids or included_lens_work_ids(handle)
        prog = Progress(len(targets), "works", emit=emit, event="work", echo=False)
        emit("lenses", f"running {lens_id} over {len(targets)} works")
        with Session(handle.engine) as session:
            registry.sync_lens(session, project_dir, lens_id)
            lens = registry.load_lens(project_dir, lens_id)
            result = run_lens(
                session, handle.root, lens, targets, build_lens_router(cfg),
                force=force, on_step=lambda wid: prog.step(wid, work_id=wid),
            )
            registry.promote_to_active(session, lens)
        update_manifest(
            handle.slug, run_id,
            {"lens_run": {
                "lens_id": lens_id, "found": result.found, "not_found": result.not_found,
                "ambiguous": result.ambiguous, "extraction_failed": result.extraction_failed,
                "degraded": result.degraded,
            }},
            root=handle.root,
        )
        emit(
            "lenses",
            f"found={result.found} not_found={result.not_found} failed={result.extraction_failed}"
            + (" [degraded:no-LLM]" if result.degraded else ""),
            found=result.found,
        )

    return fn


_TASKS: dict[str, _TaskSpec] = {
    "conversion": _TaskSpec("Corpus conversion", None, "conversion", _included_pairs, _job_conversion),
    "sections": _TaskSpec("Sections build", None, "sections", _markdown_pairs, _job_sections),
    "cite": _TaskSpec("Citation graph", None, "cite", _included_pairs, _job_cite),
    "concepts": _TaskSpec("Concept overlay", "semantic_graph_extraction", "concepts", _claim_pairs, _job_concepts),
    "extraction": _TaskSpec("Note extraction", "note_extraction", "extraction", _markdown_pairs, _job_extraction),
    "chunked": _TaskSpec("Chunked extraction", "note_extraction", "chunked", _oversize_pairs, _job_chunked),
    "lenses": _TaskSpec("Lens run", "project_lens_extraction", "lenses", _included_pairs, _job_lenses),
}

TASK_NAMES: list[str] = list(_TASKS)
TASK_LABELS: dict[str, str] = {name: spec.label for name, spec in _TASKS.items()}


# --- public API --------------------------------------------------------------


def plan_job(
    h,
    task: str,
    *,
    work_ids: Optional[list[str]] = None,
    profile_id: Optional[str] = None,
    force: bool = False,
    only_failed: bool = False,
    only_stale: bool = False,
    cache_root: Path | str | None = None,
) -> JobPlan:
    """Preview the prospective ``task`` build over project ``h`` (pure read).

    Resolves the affected works, each work's route/profile + "content may leave this
    machine" flag, a token/cost projection, and the budget confirmation gate. Raises
    :class:`SeedgraphError` for an unknown task; otherwise never dispatches or writes.
    """
    from ..config.loader import load_llm_capabilities, load_project_config
    from ..llm import cost as _cost
    from ..llm.executor import preflight
    from ..llm.profiles import is_local_profile
    from ..llm.routing import NoLlmRoute, resolve_route

    spec = _TASKS.get(task)
    if spec is None:
        raise SeedgraphError(f"unknown plan task {task!r} (known: {sorted(_TASKS)})")

    root = getattr(h, "root", None)
    if cache_root is None:
        cache_root = root
    cfg = load_project_config(h.slug, root)
    try:
        caps = load_llm_capabilities()
    except SeedgraphError:
        caps = None

    pairs = _titles_for(h, work_ids) if work_ids is not None else spec.worklist(h, cache_root)

    # Task-level route (banner + default destination, valid even with 0 works).
    route_profile_id = route_provider = None
    route_destination = "deterministic"
    needs_backend = False
    task_external = False
    if spec.llm_task:
        try:
            troute = resolve_route(spec.llm_task, "open_access", cfg)
        except Exception:  # noqa: BLE001
            troute = NoLlmRoute(task_type=spec.llm_task, deterministic_fallback=False)
        if isinstance(troute, NoLlmRoute):
            needs_backend = True
            route_destination = "deterministic" if troute.deterministic_fallback else "none"
        else:
            route_profile_id = troute.profile_id
            route_provider = troute.provider
            prof = cfg.llm.profiles.get(troute.profile_id)
            local = bool(prof is not None and is_local_profile(prof))
            route_destination = "local" if local else "external"
            task_external = bool(troute.external_full_text)

    works: list[WorkPreview] = []
    total_in = total_out = 0
    total_cost = 0.0
    ext_any = task_external

    if spec.llm_task:
        from sqlmodel import Session

        cache_conn = None
        try:
            cache_conn = open_cache_ro(cache_root)
        except Exception:  # noqa: BLE001 — preview without a cache is fine
            cache_conn = None
        try:
            with Session(h.engine) as session:
                for wid, title in pairs:
                    access_class, in_toks = _access_and_tokens(session, cache_conn, cache_root, wid)
                    plan_obj, block = preflight(
                        spec.llm_task, access_class, cfg,
                        capabilities=caps, profile_id_override=profile_id,
                    )
                    if plan_obj is not None:
                        local = is_local_profile(plan_obj.profile)
                        dest = "local" if local else "external"
                        ext = bool(plan_obj.external_full_text)
                        out_toks = in_toks
                        cost = _cost.preflight_estimate(plan_obj.cap, in_toks, out_toks)
                        pid = plan_obj.route.profile_id
                        provider = plan_obj.route.provider
                        note = None
                    else:
                        code = block.error.code if (block and block.error) else "config_error"
                        dest = "blocked" if code in ("policy_blocked", "budget_exceeded") else "deterministic"
                        ext = False
                        out_toks = 0
                        cost = 0.0
                        pid = provider = None
                        note = block.error.message if (block and block.error) else None
                    works.append(WorkPreview(
                        wid, title, access_class, pid, provider, dest, ext,
                        in_toks, out_toks, cost, note,
                    ))
                    total_in += in_toks
                    total_out += out_toks
                    total_cost += cost
                    ext_any = ext_any or ext
        finally:
            if cache_conn is not None:
                cache_conn.close()
    else:
        for wid, title in pairs:
            works.append(WorkPreview(
                wid, title, None, None, None, "deterministic", False, 0, 0, 0.0, None,
            ))

    # Budget snapshot (monthly soft limit + confirmation gate).
    conn = sqlite3.connect(str(h.db_path))
    try:
        bstatus = _cost.budget_status(conn, cfg.budget, total_cost)
    finally:
        conn.close()

    # Build D ch13 (D-13): the conversion task is the one phase that fans out
    # against rate-limited public APIs — attach its closed-form no-network
    # budget. Seed count is the worklist just computed; depth/cap are the SAME
    # module constants _job_conversion passes to run_corpus, so the preview and
    # the launched job cannot disagree.
    acquisition_preview = None
    if task == "conversion":
        from ..acquisition.preview import preview_acquisition

        acquisition_preview = preview_acquisition(
            len(pairs), _CONVERSION_DEPTH, _CONVERSION_PER_GEN_CAP
        )

    return JobPlan(
        slug=h.slug,
        task=task,
        task_type=spec.llm_task,
        is_llm_task=spec.llm_task is not None,
        label=spec.label,
        works=works,
        affected_count=len(works),
        route_profile_id=route_profile_id,
        route_provider=route_provider,
        route_destination=route_destination,
        external_full_text=ext_any,
        needs_llm_backend=needs_backend,
        est_total_input_tokens=total_in,
        est_total_output_tokens=total_out,
        est_total_cost_usd=total_cost,
        monthly_spend_usd=bstatus.prior_spend_usd,
        monthly_soft_limit_usd=bstatus.monthly_soft_limit_usd,
        over_monthly_soft_limit=bstatus.over_monthly_soft_limit,
        require_confirmation_above_usd=bstatus.confirmation_threshold_usd,
        requires_confirmation=bstatus.requires_confirmation,
        flags={
            "profile_id": profile_id, "force": force,
            "only_failed": only_failed, "only_stale": only_stale,
        },
        acquisition_preview=acquisition_preview,
    )


def launch_planned_job(h, plan: JobPlan, flags: Optional[dict] = None) -> str:
    """Dispatch ``plan``'s task body through :func:`web.jobs.launch_job`; return run_id.

    Raises :class:`web.jobs.ProjectBusyError` when a job is already running for the
    project (surfaced as a friendly busy message by the caller)."""
    from .jobs import launch_job

    merged = dict(plan.flags)
    if flags:
        merged.update({k: v for k, v in flags.items() if v is not None})
    spec = _TASKS[plan.task]
    body = spec.job(plan, merged)
    return launch_job(h.slug, phase=spec.phase, fn=body, root=getattr(h, "root", None))


__all__ = [
    "JobPlan",
    "WorkPreview",
    "TASK_NAMES",
    "TASK_LABELS",
    "plan_job",
    "launch_planned_job",
]
