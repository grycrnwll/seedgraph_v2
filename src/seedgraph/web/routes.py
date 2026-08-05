"""Localhost-only FastAPI read routes for the citation graph (phase_2, §6.2).

A thin localhost READ view over the latest run's authoritative, shareable
included->included edges — the CLI remains the complete write/answer surface
(decision 81; no HTTP write/answer endpoints). The handlers are exposed on an
``APIRouter`` so the wiring stage can ``app.include_router(router)`` into the
existing FastAPI app (``api/app.py``) without this phase editing that shared,
other-phase file (see needs_wiring).

Routes:
* ``GET /projects/{slug}/citations`` -> JSON of the latest run's authoritative
  included->included edges (read view).
* ``GET /projects/{slug}/runs/{run_id}/graph.json`` -> serve the exported artifact.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from ..errors import SeedgraphError
from ..project import service as project_service
from ..run import _status_from_events, latest_run_id, read_events
from .serve import require_csrf, require_local_session

router = APIRouter(tags=["citations"])


# ---------------------------------------------------------------------------
# settings_view (Track 2) — the read-model behind the settings screen + its JSON
# helper, lifted here so the CLI ``llm`` group, the UI, and the API share one
# source of truth. Pure read over the runtime LLM/budget/privacy config + the
# bundled pricing snapshot + a doctor badge; NEVER returns a raw secret value
# (env-var NAMES + a presence boolean only).
# ---------------------------------------------------------------------------


#: Overridable fields grouped by their runtime-config section. Each entry is the
#: leaf field name; the render type (bool vs number) is derived from the group.
_OVERRIDE_CONTENT_POLICY = (
    "external_llm_for_private_full_text",
    "external_llm_for_answer_generation",
    "allow_external_llm",
)
_OVERRIDE_BUDGET = (
    "usd_limit",
    "max_tokens",
    "per_run_soft_limit_usd",
    "monthly_soft_limit_usd",
    "require_confirmation_above_usd",
    "allow_unverified_pricing",
)
_OVERRIDE_ANSWER = (
    "max_evidence_tokens",
    "max_fragment_chars",
    "prompt_overhead_tokens",
    "reserved_output_tokens",
    "weak_evidence_floor",
    "absent_floor",
    "support_floor",
)


def _project_overrides_view(
    merged_cfg: Any, root: Path | str | None, slug: str
) -> dict[str, Any]:
    """Per-field inherit-vs-override read-model for the PROJECT settings screen.

    For each overridable field it reports the effective (merged + clamped) value,
    the value it WOULD inherit from the current global config, and whether the raw
    ``project.yaml`` overlay explicitly carries that key (``overridden``). The
    ``ceiling`` marker (from :data:`CEILING_SPEC`) flags fields a project may only
    tighten. Pure read; the write path lives in ``web.ui``.
    """
    from .. import paths
    from ..config.loader import _read_yaml, load_global_config
    from ..config.models import CEILING_SPEC

    raw = _read_yaml(paths.project_dir(slug, root) / "project.yaml")
    global_cfg = load_global_config(root)
    cp_raw = raw.get("content_policy") if isinstance(raw.get("content_policy"), dict) else {}
    bud_raw = raw.get("budget") if isinstance(raw.get("budget"), dict) else {}
    ans_raw = raw.get("answer") if isinstance(raw.get("answer"), dict) else {}
    llm_raw = raw.get("llm") if isinstance(raw.get("llm"), dict) else {}
    routes_raw = llm_raw.get("routes") if isinstance(llm_raw.get("routes"), dict) else {}

    def _leaf(group_raw, eff_obj, glob_obj, field, ftype, dotted):
        return {
            "field": field,
            "type": ftype,
            "ceiling": CEILING_SPEC.get(dotted),
            "effective": getattr(eff_obj, field),
            "global": getattr(glob_obj, field),
            "overridden": field in group_raw,
        }

    content_policy = [
        _leaf(cp_raw, merged_cfg.content_policy, global_cfg.content_policy, f,
              "bool", f"content_policy.{f}")
        for f in _OVERRIDE_CONTENT_POLICY
    ]
    budget = [
        _leaf(bud_raw, merged_cfg.budget, global_cfg.budget, f,
              "bool" if f == "allow_unverified_pricing" else "number", f"budget.{f}")
        for f in _OVERRIDE_BUDGET
    ]
    answer = [
        _leaf(ans_raw, merged_cfg.answer, global_cfg.answer, f, "number", f"answer.{f}")
        for f in _OVERRIDE_ANSWER
    ]
    routes = []
    for task in sorted(merged_cfg.llm.routes.keys()):
        eff = merged_cfg.llm.routes[task]
        glob = global_cfg.llm.routes.get(task)
        task_raw = routes_raw.get(task) if isinstance(routes_raw.get(task), dict) else {}
        routes.append(
            {
                "task": task,
                "preferred": {
                    "effective": eff.preferred_profile,
                    "global": glob.preferred_profile if glob else None,
                    "overridden": "preferred_profile" in task_raw,
                },
                "fallback": {
                    "effective": eff.fallback_profile,
                    "global": glob.fallback_profile if glob else None,
                    "overridden": "fallback_profile" in task_raw,
                },
            }
        )
    return {
        "content_policy": content_policy,
        "budget": budget,
        "answer": answer,
        "routes": routes,
    }


def settings_view(root: Path | str | None = None, slug: str | None = None) -> dict[str, Any]:
    """Compose the settings read-model for the global home or one project ``slug``.

    Returns the profile/route matrix by task, the privacy toggles, the budget +
    this-month DB spend, key-ref presence **by NAME** (never the value), the
    pricing-snapshot freshness, an Ollama health hint, and the doctor results. Safe
    offline/keyless; a project ``slug`` adds the project's monthly spend and routes
    the doctor checks at the project scope.
    """
    from ..config.loader import (
        load_global_config,
        load_llm_capabilities,
        load_project_config as load_runtime_config,
    )
    from ..llm import cost as _cost
    from ..llm.profiles import is_local_profile, is_profile_available
    from ..llm.providers.ollama import DEFAULT_BASE_URL
    from ..llm.secrets import resolve_profile_key
    from .. import doctor

    cfg = load_runtime_config(slug, root) if slug else load_global_config(root)

    profiles: list[dict] = []
    ollama_base_url = DEFAULT_BASE_URL
    for pid, profile in sorted(cfg.llm.profiles.items()):
        local = is_local_profile(profile)
        key_present = False
        if not local:
            # keyring → env chain (ADR-0002); a keyring-only profile (no env_var)
            # now reports presence truthfully.
            key_present = resolve_profile_key(profile) is not None
        if profile.provider == "ollama" and profile.base_url:
            ollama_base_url = profile.base_url
        profiles.append(
            {
                "profile_id": pid,
                "provider": profile.provider,
                "model": profile.model,
                "is_local": local,
                "env_var": profile.env_var,  # NAME only — never the secret value
                "key_present": key_present,
                "available": is_profile_available(profile),
            }
        )

    routes: list[dict] = []
    for task, route in sorted(cfg.llm.routes.items()):
        pref = cfg.llm.profiles.get(route.preferred_profile)
        routes.append(
            {
                "task": task,
                "preferred_profile": route.preferred_profile,
                "fallback_profile": route.fallback_profile,
                "preferred_available": bool(pref is not None and is_profile_available(pref)),
                "preferred_local": bool(pref is not None and is_local_profile(pref)),
                "requires_source_text": route.requires_source_text,
                "deterministic_fallback": route.deterministic_fallback,
            }
        )

    year_month = _cost.this_month()
    monthly_spend = 0.0
    if slug:
        try:
            handle = project_service.open_project(slug, root=Path(root) if root else None)
            conn = sqlite3.connect(str(handle.db_path))
            try:
                monthly_spend = _cost.monthly_spend(conn, year_month)
            finally:
                conn.close()
        except SeedgraphError:
            monthly_spend = 0.0

    budget = cfg.budget
    privacy = cfg.content_policy.model_dump()

    try:
        caps = load_llm_capabilities()
        from datetime import date as _date

        age_days = (_date.today() - caps.snapshot_date).days
        pricing = {
            "snapshot_date": caps.snapshot_date.isoformat(),
            "age_days": age_days,
            "fresh": age_days <= 180,
        }
    except SeedgraphError as exc:
        pricing = {"snapshot_date": None, "age_days": None, "fresh": False, "error": str(exc)}

    checks = doctor.collect_checks(root, slug)
    doctor_results = [
        {"name": c.name, "ok": c.ok, "severity": c.severity, "detail": c.detail}
        for c in checks
    ]
    n_fail = sum(1 for c in checks if (not c.ok) and c.severity == "error")
    n_warn = sum(1 for c in checks if (not c.ok) and c.severity == "warning")

    # Settings-inheritance read-model (additive, backward-compatible). In PROJECT
    # scope we surface, per overridable field, BOTH the effective (merged, clamped)
    # value AND whether it is inherited-from-global or explicitly overridden — by
    # comparing the RAW project.yaml overlay keys against the merged effective
    # config. Global scope leaves ``overrides`` as ``None`` (nothing to inherit).
    overrides = _project_overrides_view(cfg, root, slug) if slug else None

    return {
        "scope": "project" if slug else "global",
        "slug": slug,
        "profiles": profiles,
        "routes": routes,
        "profile_ids": sorted(cfg.llm.profiles.keys()),
        "privacy": privacy,
        "answer": cfg.answer.model_dump(),
        "answer_policy": cfg.answer_policy.model_dump(),
        "overrides": overrides,
        "budget": {
            "usd_limit": budget.usd_limit,
            "max_tokens": budget.max_tokens,
            "monthly_soft_limit_usd": budget.monthly_soft_limit_usd,
            "per_run_soft_limit_usd": budget.per_run_soft_limit_usd,
            "require_confirmation_above_usd": budget.require_confirmation_above_usd,
            "allow_unverified_pricing": budget.allow_unverified_pricing,
            "year_month": year_month,
            "monthly_spend_usd": monthly_spend,
        },
        "pricing_snapshot": pricing,
        "ollama": {
            "base_url": ollama_base_url,
            "hint": "run `seedgraph doctor --probe-ollama` for a live reachability probe",
        },
        "doctor": {"results": doctor_results, "n_fail": n_fail, "n_warn": n_warn, "ok": n_fail == 0},
    }


# --- phase_5 project-model read routes (decision 81 — read-only this phase) ---
# Mounted via the same router api/app.py already includes, so no shared-file edit
# is needed. All mutation goes through the CLI over the same service core.

@router.get("/projects")
def list_projects() -> list[str]:
    """List the slugs of every project under the seedgraph home (read-only)."""
    return project_service.list_projects()


@router.get("/projects/{slug}/documents", dependencies=[Depends(require_local_session)])
def list_project_documents(slug: str, status: str | None = None) -> list[dict]:
    """Return the project's corpus rows, optionally filtered to one inclusion
    ``status`` (read-only view over ``project.service.list_documents``)."""
    try:
        handle = project_service.open_project(slug)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    statuses = (status,) if status else ("included", "metadata_only", "excluded")
    return project_service.list_documents(handle, statuses)


# --- phase_7 concept read routes (decision 81 — read-only) -----------------


def _open_project_conn(slug: str) -> sqlite3.Connection:
    handle = project_service.open_project(slug)
    conn = sqlite3.connect(str(handle.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@router.get("/projects/{slug}/concepts")
def list_project_concepts(
    slug: str, type: str | None = None, status: str | None = None
) -> list[dict]:
    """List the project's concepts (read-only view over ``semantic.query``)."""
    from ..semantic.query import list_concepts

    try:
        conn = _open_project_conn(slug)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return list_concepts(conn, concept_type=type, status=status)
    finally:
        conn.close()


@router.get(
    "/projects/{slug}/concepts/{concept_id:path}",
    dependencies=[Depends(require_local_session)],
)
def get_project_concept(slug: str, concept_id: str) -> dict:
    """Return ``{concept, papers[], claims[], spans[]}`` for one concept (the
    doc 10 §11 milestone query as a read route)."""
    from ..semantic.query import concept_detail

    try:
        conn = _open_project_conn(slug)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        detail = concept_detail(conn, concept_id)
    finally:
        conn.close()
    if detail is None:
        raise HTTPException(status_code=404, detail=f"no concept {concept_id!r}")
    return detail


@router.get("/projects/{slug}/citations")
def list_citations(slug: str) -> dict:
    """Return the latest run's authoritative included->included edges as JSON.

    Read-only view: resolves the project's latest ``run_id``, computes
    ``authoritative_edges`` (highest provenance per source->target), filters to the
    closed-world included corpus, and returns the shareable
    (``is_shareable_edge``) edge list. Never mints edges or a ``run_id``.
    """
    from ..citation.edges import authoritative_edges, is_shareable_edge

    try:
        handle = project_service.open_project(slug)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    run_id = latest_run_id(slug, root=handle.root)
    if run_id is None:
        return {"run_id": None, "edges": []}

    conn = sqlite3.connect(str(handle.db_path))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        edges = authoritative_edges(conn, run_id)
        included = {
            row[0]
            for row in conn.execute(
                "SELECT work_id FROM project_documents WHERE inclusion_status = 'included'"
            ).fetchall()
        }
    finally:
        conn.close()

    out = [
        {
            "source_work_id": edge["source_work_id"],
            "target_work_id": edge["target_work_id"],
            "edge_type": edge["edge_type"],
            "provenance": edge["provenance"],
            "confidence": edge["confidence"],
        }
        for edge in edges
        if edge["source_work_id"] in included
        and edge["target_work_id"] in included
        and is_shareable_edge(edge["provenance"])
    ]
    return {"run_id": run_id, "edges": out}


# --- phase_8 answer read route (decision 81 — thin in-process read view) -----

@router.get("/projects/{slug}/answer", dependencies=[Depends(require_local_session)])
def get_answer(slug: str, q: str, mode: str = "project_only", no_llm: bool = False) -> dict:
    """Answer ``q`` over project ``slug`` in-process and return the envelope as JSON.

    A thin localhost read view over ``answer.answer()`` (no POST/streaming/external
    surface over private artifacts). Returns the full :class:`AnswerEnvelope` dict so
    the caller can inspect citations / ``insufficient_evidence`` deterministically.
    """
    from ..answer.harness import answer as answer_fn
    from ..answer.types import AnswerMode

    if mode not in ("project_only", "allow_outside"):
        raise HTTPException(status_code=400, detail="mode must be project_only or allow_outside")
    try:
        handle = project_service.open_project(slug)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Localhost JSON API stays ephemeral in v1 — trace dropped visibly (00 §5).
    env, _ = answer_fn(q, handle, mode=AnswerMode(mode), no_llm=no_llm)
    return env.model_dump()


@router.get(
    "/projects/{slug}/runs/{run_id}/graph.json",
    dependencies=[Depends(require_local_session)],
)
def get_run_graph(slug: str, run_id: str) -> dict:
    """Serve the exported ``runs/{run_id}/graph.json`` artifact for ``slug``.

    Read-only: returns the previously exported, shareable-filtered node-link
    document for the given run (404 if the run or artifact is absent).
    """
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=404, detail=f"invalid run id {run_id!r}")
    try:
        handle = project_service.open_project(slug)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    path = handle.db_path.parent / "runs" / run_id / "graph.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no graph.json for run {run_id!r}")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get(
    "/api/projects/{slug}/runs/{run_id}/events",
    dependencies=[Depends(require_local_session)],
)
def get_run_events(slug: str, run_id: str, after: int = -1) -> dict:
    """Return run progress events with ``seq > after`` for UI polling (review #7).

    Read-only view over :func:`run.read_events`; ``status`` is the run's coarse state
    derived from its terminal event (``finished``/``failed``/``running``/``None``) and
    is always computed over the **full** log, not just the slice after ``after``.
    Behind ``require_local_session`` like every other private-project view.
    """
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=404, detail=f"invalid run id {run_id!r}")
    try:
        events = read_events(slug, run_id)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    status = _status_from_events(events)
    visible = [e for e in events if int(e.get("seq", -1)) > after]
    return {"run_id": run_id, "events": visible, "status": status}


# ---------------------------------------------------------------------------
# Track 2 read-only JSON helpers (all gated). Thin views over the shared service
# layer (dashboard / corpus / lenses / settings) so the UI never re-implements a
# read; mutation stays on the CLI / service core (decision 81).
# ---------------------------------------------------------------------------


def _open_handle(slug: str):
    try:
        return project_service.open_project(slug)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/projects/{slug}/dashboard",
    dependencies=[Depends(require_local_session)],
)
def api_project_dashboard(slug: str) -> dict:
    """Project dashboard read-model (counts, coverage, runs, models, budget, doctor)."""
    return project_service.project_dashboard(_open_handle(slug))


@router.get(
    "/api/projects/{slug}/corpus",
    dependencies=[Depends(require_local_session)],
)
def api_project_corpus(slug: str, status: str | None = None) -> list[dict]:
    """Dense corpus table (metadata + enrichment flags), optionally one ``status``."""
    from ..acquisition.service import corpus_rows

    statuses = (status,) if status else None
    return corpus_rows(_open_handle(slug), statuses=statuses)


@router.post(
    "/ui/projects/{slug}/corpus/{work_id}/upload",
    dependencies=[Depends(require_local_session), Depends(require_csrf)],
)
async def ui_frontier_upload(request: Request, slug: str, work_id: str) -> RedirectResponse:
    """Per-row PDF upload for the corpus frontier pane (Build D ch12).

    Bridges ONE existing work with a user-supplied PDF via the existing
    :func:`acquisition.service.manual_upload` (ingest → convert → bridge — NO
    new ingest logic; ``promote`` stays at its default ``True``, the same
    default the ``corpus upload`` CLI verb ships). Membership-validated first
    (404 for a work outside this project's corpus, so no orphan bridge row can
    be minted), ``%PDF``-guarded (400), then 303→ the corpus page, where the
    now-provided row has dropped out of the pane. Gated like every mutating
    POST (session + CSRF). ponytail: the convert runs synchronously in the
    request — acceptable for a one-row drop (the bulk paths own the background
    queue); ceiling is one Marker conversion per click.
    """
    import os
    import tempfile

    from sqlmodel import Session

    from ..acquisition.service import manual_upload
    from ..db.project_models import ProjectDocument

    handle = _open_handle(slug)
    with Session(handle.engine) as session:
        if session.get(ProjectDocument, work_id) is None:
            raise HTTPException(status_code=404, detail="work not in corpus")
    form = await request.form()
    up = form.get("pdf")
    read = getattr(up, "read", None)
    data = await read() if read is not None else b""
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="not a PDF")
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        manual_upload(handle, work_id=work_id, pdf_path=Path(tmp.name), cache_root=None)
    except SeedgraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return RedirectResponse(url=f"/ui/projects/{slug}/corpus", status_code=303)


@router.get(
    "/api/projects/{slug}/lenses",
    dependencies=[Depends(require_local_session)],
)
def api_project_lenses(slug: str) -> list[dict]:
    """Registered lenses (id/name/status/object_type/hash) for the project."""
    from sqlmodel import Session

    from ..lenses import registry

    handle = _open_handle(slug)
    with Session(handle.engine) as session:
        rows = registry.list_lenses(session)
    return [
        {
            "lens_id": r.lens_id,
            "name": r.name,
            "scope": r.scope,
            "object_type": r.object_type,
            "status": r.status,
            "definition_hash": r.definition_hash,
        }
        for r in rows
    ]


@router.get(
    "/api/projects/{slug}/lenses/{lens_id}",
    dependencies=[Depends(require_local_session)],
)
def api_project_lens_detail(slug: str, lens_id: str, status: str = "found") -> dict:
    """One lens: its result rows (``status`` filter) + coverage/staleness read-model."""
    from sqlmodel import Session

    from ..lenses import registry
    from ..lenses import runner as lens_runner

    handle = _open_handle(slug)
    project_dir = handle.root / "projects" / handle.slug
    with Session(handle.engine) as session:
        results = lens_runner.lens_results(session, lens_id, status=status)
        staleness = registry.lens_staleness(session, project_dir, lens_id)
    return {"lens_id": lens_id, "results": results, "staleness": staleness}


@router.get("/api/settings", dependencies=[Depends(require_local_session)])
def api_global_settings() -> dict:
    """Global settings read-model (profile/route matrix, privacy, budget, doctor)."""
    return settings_view()


@router.get(
    "/api/projects/{slug}/settings",
    dependencies=[Depends(require_local_session)],
)
def api_project_settings(slug: str) -> dict:
    """Project-scoped settings read-model (adds the project's monthly spend)."""
    try:
        project_service.open_project(slug)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return settings_view(slug=slug)
