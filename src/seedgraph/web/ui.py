"""Jinja2 HTML UI surface for ``seedgraph serve`` (Track 2).

``attach_ui(app)`` mounts the static asset tree at ``/static`` and includes the
``/ui`` HTML router. :func:`render` is the single template entry point — it always
injects the CSRF token, the remote-bind banner hint, and nav basics (never the
session token) so a template can't leak it.

Stage A ships the read-only screens over the shared service layer: first-run setup,
the project dashboard, the corpus table + its gated full-text drawer, the answer
workspace (synchronous ``no_llm`` / retrieval-only render), the lens list + detail,
and the global/project settings views. The project-list landing page and the graph
placeholder (which 302→ the latest run's graph view) are the Track-2 shell. Every
screen that can surface private evidence is behind ``require_local_session``;
mutating forms + the job planner land in a later stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import paths
from ..errors import SeedgraphError
from ..project import service as project_service
from ..run import latest_run_id
from . import jobs, marker_queue
from .serve import require_csrf, require_local_session

_WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = _WEB_DIR / "templates"
STATIC_DIR = _WEB_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
ui_router = APIRouter(tags=["ui"])

#: Applied to every screen that can render private evidence (cross-cutting #1).
_GATE = [Depends(require_local_session)]
#: Applied to every mutating POST: the loopback+cookie gate AND the synchronizer
#: CSRF token (``_csrf == app.state.csrf_token``). Both fail-closed (403).
_POST_GATE = [Depends(require_local_session), Depends(require_csrf)]


def _redirect(url: str) -> RedirectResponse:
    """303 See Other so a POST result is followed with a GET (no form re-submit)."""
    return RedirectResponse(url=url, status_code=303)


def render(
    request: Request, name: str, *, status_code: int = 200, **ctx: Any
) -> HTMLResponse:
    """Render template ``name`` with the always-on base context merged over ``ctx``.

    Injects ``csrf_token`` (for hidden form fields), ``remote_bind`` (banner hint),
    and ``nav`` basics on every page. **Never** injects ``session_token`` — the
    HttpOnly cookie is the only place that value lives. ``status_code`` lets a
    handler render a non-200 body (e.g. a 409 friendly busy page).
    """
    base: dict[str, Any] = {
        "csrf_token": getattr(request.app.state, "csrf_token", ""),
        "remote_bind": getattr(request.app.state, "remote_bind", False),
        "nav": {"brand": "seedgraph", "home_url": "/ui"},
    }
    base.update(ctx)
    return templates.TemplateResponse(request, name, base, status_code=status_code)


def _enrich_envelope(env_dict: dict[str, Any], handle) -> dict[str, Any]:
    """Annotate each citation with its work's ``access_class`` (shared by every
    answer-rendering surface — ask GET/POST + the run page)."""
    access_by_work = {
        d["work_id"]: d.get("access_status")
        for d in project_service.list_documents(handle)
    }
    for cit in env_dict.get("citations", []):
        cit["access_class"] = access_by_work.get(cit.get("work_id"))
    return env_dict


def _answer_backend_available(cfg) -> bool:
    """True when ``answer_generation`` resolves to a usable (non-no-LLM) profile.

    Decides whether a real-LLM ask runs as a background job (prose) or degrades to a
    synchronous retrieval-only answer (no backend) — never a crash either way."""
    from ..llm.routing import NoLlmRoute, resolve_route

    try:
        route = resolve_route("answer_generation", "open_access", cfg)
    except Exception:  # noqa: BLE001 — any resolution failure ⇒ no backend
        return False
    return not isinstance(route, NoLlmRoute)


def _load_run_answer(handle, run_id: str) -> dict[str, Any] | None:
    """Load a saved :class:`AnswerEnvelope` for ``run_id`` (the answer-job artifact)."""
    import json

    from ..project import layout

    base = layout.project_runs_dir(handle.slug, handle.root) / run_id / "answers"
    if not base.exists():
        return None
    files = sorted(base.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _validate_slug(slug: str) -> None:
    try:
        paths.validate_slug(slug)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _open_project(slug: str):
    _validate_slug(slug)
    try:
        return project_service.open_project(slug)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _title_from_filename(filename: str) -> str:
    """Human-ish title from an uploaded filename: drop the extension, spaces for _/-."""
    from pathlib import PurePosixPath

    stem = PurePosixPath(filename).stem or filename
    return stem.replace("_", " ").replace("-", " ").strip() or filename


# --- shell: project list + graph placeholder (ungated; no private evidence) ---


@ui_router.get("/ui", response_class=HTMLResponse)
def ui_projects(request: Request) -> HTMLResponse:
    """Project-list landing page (read-only over ``project_service.list_projects``).

    Lists slugs + links only (no private evidence), so it stays reachable without a
    session cookie as the entry point; every screen it links to is gated."""
    return render(request, "projects.html", projects=project_service.list_projects())


@ui_router.post("/ui/projects", dependencies=_POST_GATE)
async def ui_create_project(request: Request):
    """Create a new project from the landing-page form, then 303 to its corpus."""
    form = await request.form()
    slug = str(form.get("slug", "")).strip()
    name = str(form.get("name", "")).strip() or None
    description = str(form.get("description", "")).strip() or None
    try:
        project_service.create_project(slug, name=name, description=description)
    except SeedgraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _redirect(f"/ui/projects/{slug}/corpus")


@ui_router.get("/ui/projects/{slug}/graph")
def ui_project_graph(request: Request, slug: str):
    """Graph placeholder: 302→ the latest run's run-scoped graph view.

    Renders a friendly no-runs page when the project has no runs yet. Track 3 owns
    the actual ``/ui/projects/{slug}/runs/{run_id}/graph`` view; run-scoped is the
    canonical graph URL.
    """
    _validate_slug(slug)
    latest = latest_run_id(slug)
    if latest is None:
        return render(request, "no_runs.html", slug=slug)
    return RedirectResponse(
        url=f"/ui/projects/{slug}/runs/{latest}/graph", status_code=302
    )


# --- first-run setup (gated) -------------------------------------------------


@ui_router.get("/ui/setup", response_class=HTMLResponse, dependencies=_GATE)
def ui_setup(request: Request) -> HTMLResponse:
    """First-run setup: doctor checks + the current global config (read-only form).

    Stage A renders the form + diagnostics; the write happens via
    ``config.loader.write_global_config`` in the mutating-forms stage."""
    from .. import doctor
    from ..config.loader import load_global_config

    checks = [
        {"name": c.name, "ok": c.ok, "severity": c.severity, "detail": c.detail}
        for c in doctor.collect_checks()
    ]
    cfg = load_global_config()
    config_summary = {
        "home": str(paths.resolve_home(None)),
        "log_level": cfg.log_level,
        "n_profiles": len(cfg.llm.profiles),
        "n_routes": len(cfg.llm.routes),
        "external_llm_for_private_full_text": cfg.content_policy.external_llm_for_private_full_text,
        "external_llm_for_answer_generation": cfg.content_policy.external_llm_for_answer_generation,
    }
    n_fail = sum(1 for c in checks if (not c["ok"]) and c["severity"] == "error")
    return render(
        request,
        "setup.html",
        checks=checks,
        config=config_summary,
        n_fail=n_fail,
    )


def _global_config_patch(form) -> dict[str, Any]:
    """Build the ``write_global_config`` overlay patch from a submitted setup/settings
    form: the privacy toggles (checkboxes — present ⇒ true), the budget soft-limits
    (only non-empty values, validated by the config model), an optional
    ``note_extraction`` route preference, and the log level.

    Only env-var NAMES ever appear in the patch — never a raw key value (the writer
    re-asserts this, raising :class:`ConfigError` on any embedded secret)."""
    patch: dict[str, Any] = {}
    log_level = str(form.get("log_level", "")).strip()
    if log_level:
        patch["log_level"] = log_level
    patch["content_policy"] = {
        "external_llm_for_private_full_text": form.get("external_llm_for_private_full_text")
        is not None,
        "external_llm_for_answer_generation": form.get("external_llm_for_answer_generation")
        is not None,
    }
    budget: dict[str, Any] = {}
    for key in (
        "usd_limit",
        "monthly_soft_limit_usd",
        "per_run_soft_limit_usd",
        "require_confirmation_above_usd",
    ):
        val = form.get(key)
        if val is not None and str(val).strip() != "":
            budget[key] = str(val).strip()  # coerced + validated by LlmBudget
    if budget:
        patch["budget"] = budget
    # answer-harness tunables (free, global-only editor): only non-empty values are
    # included, so an untouched field stays at the current global default.
    answer: dict[str, Any] = {}
    for key in (
        "max_evidence_tokens",
        "max_fragment_chars",
        "prompt_overhead_tokens",
        "reserved_output_tokens",
        "weak_evidence_floor",
        "absent_floor",
        "support_floor",
    ):
        val = form.get(key)
        if val is not None and str(val).strip() != "":
            answer[key] = str(val).strip()  # coerced + validated by AnswerConfig
    if answer:
        patch["answer"] = answer
    # answer_policy is a checkbox group: only interpreted when the section was
    # actually rendered (``_has_answer_policy`` marker), so a partial POST that omits
    # it never silently flips a default (e.g. require_project_sources True→False).
    if form.get("_has_answer_policy") is not None:
        patch["answer_policy"] = {
            "require_project_sources": form.get("require_project_sources") is not None,
            "allow_external_search_by_default": form.get("allow_external_search_by_default")
            is not None,
            "distinguish_source_claims_from_synthesis": form.get(
                "distinguish_source_claims_from_synthesis"
            )
            is not None,
        }
    # Routing profile CHOICE editor (per task): ``route_<task>_preferred`` /
    # ``route_<task>_fallback`` select fields. Only endpoint-free profile references
    # ever reach the patch (the writer re-asserts the credential boundary).
    routes: dict[str, Any] = {}
    for key in list(form.keys()):
        if not (key.startswith("route_") and key.endswith("_preferred")):
            continue
        task = key[len("route_") : -len("_preferred")]
        pref = str(form.get(key, "")).strip()
        if not pref:
            continue
        entry = {"task_type": task, "preferred_profile": pref}
        fb = str(form.get(f"route_{task}_fallback", "")).strip()
        if fb:
            entry["fallback_profile"] = fb
        routes[task] = entry
    note_profile = str(form.get("note_extraction_profile", "")).strip()
    if note_profile:
        routes["note_extraction"] = {
            "task_type": "note_extraction",
            "preferred_profile": note_profile,
        }
    if routes:
        patch.setdefault("llm", {}).setdefault("routes", {}).update(routes)
    return patch


def _coerce_scalar(text: str) -> Any:
    """Coerce a numeric form string to ``int``/``float`` so the project overlay stores
    a real number (not a quoted string); leave anything non-numeric untouched so the
    downstream config validation surfaces a clean error."""
    t = text.strip()
    try:
        return float(t) if any(c in t for c in ".eE") else int(t)
    except ValueError:
        return t


def _project_config_patch(form) -> dict[str, Any]:
    """Build the thin ``write_project_overrides`` patch from a project settings form.

    Covers the four inheritance override groups — ``content_policy`` (privacy bools),
    ``budget``, ``answer`` tunables, and the routing profile CHOICE. Each field is
    guarded by an ``override_<field>`` checkbox: when that box is UNCHECKED the field
    is OMITTED entirely, so the project keeps inheriting the live global default (the
    current global value is never materialized into ``project.yaml``). Reverting an
    existing override to inheritance is a separate action (``revert=…``); this builder
    only ever ADDS/updates a checked override.
    """
    patch: dict[str, Any] = {}
    content_policy: dict[str, Any] = {}
    for field in (
        "external_llm_for_private_full_text",
        "external_llm_for_answer_generation",
        "allow_external_llm",
    ):
        if form.get(f"override_{field}") is not None:
            content_policy[field] = form.get(field) is not None
    if content_policy:
        patch["content_policy"] = content_policy

    budget: dict[str, Any] = {}
    for field in (
        "usd_limit",
        "max_tokens",
        "per_run_soft_limit_usd",
        "monthly_soft_limit_usd",
        "require_confirmation_above_usd",
    ):
        if form.get(f"override_{field}") is not None:
            val = str(form.get(field, "")).strip()
            if val != "":
                budget[field] = _coerce_scalar(val)  # validated by LlmBudget
    if form.get("override_allow_unverified_pricing") is not None:
        budget["allow_unverified_pricing"] = form.get("allow_unverified_pricing") is not None
    if budget:
        patch["budget"] = budget

    answer: dict[str, Any] = {}
    for field in (
        "max_evidence_tokens",
        "max_fragment_chars",
        "prompt_overhead_tokens",
        "reserved_output_tokens",
        "weak_evidence_floor",
        "absent_floor",
        "support_floor",
    ):
        if form.get(f"override_{field}") is not None:
            val = str(form.get(field, "")).strip()
            if val != "":
                answer[field] = _coerce_scalar(val)  # validated by AnswerConfig
    if answer:
        patch["answer"] = answer

    # Routing profile CHOICE per task: a project may only choose an existing global
    # profile (never define/redirect one). No task_type key — it inherits from the
    # global route via the loader deep-merge.
    routes: dict[str, Any] = {}
    for key in list(form.keys()):
        if not key.startswith("override_route_"):
            continue
        task = key[len("override_route_") :]
        entry: dict[str, Any] = {}
        pref = str(form.get(f"route_{task}_preferred", "")).strip()
        if pref:
            entry["preferred_profile"] = pref
        fb = str(form.get(f"route_{task}_fallback", "")).strip()
        if fb:
            entry["fallback_profile"] = fb
        if entry:
            routes[task] = entry
    if routes:
        patch.setdefault("llm", {}).setdefault("routes", {}).update(routes)
    return patch


def _assert_project_override_not_loosening(patch: dict[str, Any], global_cfg) -> None:
    """Reject a project override patch that would LOOSEN a tighten-only ceiling.

    ``config.loader.write_project_overrides`` clamps a loosening ceiling silently at
    load time (defense-in-depth) rather than raising, so the settings screen makes the
    attempt an explicit, visible failure: it reuses the real ceiling rule
    (:func:`config.loader._apply_ceiling_rule` + :data:`CEILING_SPEC`) on the exact
    fields the patch touches and raises :class:`ConfigError` if the clamp would alter
    the requested value (i.e. the request tried to widen the global ceiling). Free
    overrides (routing choice, ``answer.*``) are never checked. Answer tunables are
    free; only ``content_policy`` + ``budget`` carry UI-editable ceilings.
    """
    from ..config.loader import _apply_ceiling_rule
    from ..config.models import CEILING_SPEC, ContentPolicy, LlmBudget
    from ..errors import ConfigError

    checks: list[tuple[str, str, str, Any, Any]] = []
    if isinstance(patch.get("content_policy"), dict):
        cp = ContentPolicy(**patch["content_policy"])
        for field in patch["content_policy"]:
            rule = CEILING_SPEC.get(f"content_policy.{field}")
            if rule:
                checks.append(
                    ("content_policy", field, rule,
                     getattr(cp, field), getattr(global_cfg.content_policy, field))
                )
    if isinstance(patch.get("budget"), dict):
        bud = LlmBudget(**patch["budget"])
        for field in patch["budget"]:
            rule = CEILING_SPEC.get(f"budget.{field}")
            if rule:
                checks.append(
                    ("budget", field, rule,
                     getattr(bud, field), getattr(global_cfg.budget, field))
                )
    for group, field, rule, requested, global_val in checks:
        if _apply_ceiling_rule(rule, global_val, requested) != requested:
            raise ConfigError(
                f"project override may not loosen {group}.{field}: the global ceiling "
                f"is {global_val!r} but the project requested {requested!r} — a project "
                f"may only make this setting MORE restrictive, never looser"
            )


def _revert_project_override(slug: str, revert: str, root=None) -> None:
    """Remove one override key from the raw ``project.yaml`` overlay (revert-to-global).

    ``revert`` is ``"<group>:<field>"`` (a single ``content_policy`` / ``budget`` /
    ``answer`` leaf), ``"route:<task>"`` (a whole route choice), or a bare group name
    (the whole group). Emptied parents are pruned so the overlay stays sparse. Removing
    an override can only make the field inherit the (valid) global default, so no
    re-validation is needed. Atomic + overlay-preserving (every other key survives)."""
    import yaml as _yaml

    from ..config.loader import _atomic_write_text
    from ..project import layout

    path = layout.project_yaml_path(slug, root)
    if not path.exists():
        return
    raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return
    changed = False
    if revert.startswith("route:"):
        task = revert[len("route:") :]
        llm = raw.get("llm")
        routes = llm.get("routes") if isinstance(llm, dict) else None
        if isinstance(routes, dict) and task in routes:
            del routes[task]
            changed = True
            if not routes:
                llm.pop("routes", None)
            if isinstance(llm, dict) and not llm:
                raw.pop("llm", None)
    elif ":" in revert:
        group, field = revert.split(":", 1)
        grp = raw.get(group)
        if isinstance(grp, dict) and field in grp:
            del grp[field]
            changed = True
            if not grp:
                raw.pop(group, None)
    elif revert in raw:
        raw.pop(revert)
        changed = True
    if changed:
        _atomic_write_text(
            path, _yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
        )


@ui_router.post("/ui/setup", dependencies=_POST_GATE)
async def ui_setup_save(request: Request):
    """Persist the first-run setup form via ``write_global_config`` then re-run doctor.

    Same validation as the shared writer (a bad patch — e.g. an unknown route
    profile — raises :class:`ConfigError`, surfaced verbatim as a 400; nothing is
    written). On success the diagnostics are refreshed and we 303→ the setup screen."""
    from .. import doctor
    from ..config.loader import write_global_config
    from ..errors import ConfigError

    form = await request.form()
    try:
        write_global_config(_global_config_patch(form))
    except (ConfigError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    doctor.collect_checks()  # re-run diagnostics (fresh on the next GET)
    return _redirect("/ui/setup")


# --- dashboard (gated) -------------------------------------------------------


@ui_router.get("/ui/projects/{slug}", response_class=HTMLResponse, dependencies=_GATE)
def ui_dashboard(request: Request, slug: str) -> HTMLResponse:
    """Project dashboard (counts, extraction coverage, reviews, runs, models, budget)."""
    handle = _open_project(slug)
    dash = project_service.project_dashboard(handle, cache_root=None)
    return render(request, "dashboard.html", slug=slug, dash=dash)


# --- corpus table + full-text drawer (gated) ---------------------------------

#: Build D ch12 — v1 ``report.py:_KLASS_RANK`` triage order translated to v2's
#: DERIVED acquisition states: the genuinely-paywalled rows always lead ("get
#: THESE first"), unknown-status stubs follow, and the OA-available re-run set
#: sorts last (a re-run / budget bump fetches those with no manual work).
#: Unlisted states rank with "unknown" (the v1 default rank).
_FRONTIER_STATE_RANK: dict[str, int] = {
    "requires_user_upload": 0,  # v1 "paywalled"
    "failed": 1,  # bridge no longer reconciles — needs attention, like unknown
    "metadata_only": 1,  # v1 "unknown"
    "available_open_access": 3,  # v1 "oa_unfetched" — always last
}


def _frontier_rows(handle, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Missing-paper frontier read-model for the corpus page (Build D ch12).

    Folds the SAME ``corpus_rows`` read-model the table + conversion bar render
    (one source of truth): keeps the unprovided rows (``has_markdown=False``,
    inclusion ≠ excluded), attaches lawful-access ``links`` per row — the public
    resolvers plus the OPTIONAL OpenURL/EZproxy institutional links, with
    ``openurl_resolver`` / ``ezproxy_host`` threaded from the EFFECTIVE
    inheritance-resolved project config (``config.loader.load_project_config``,
    the same merged+clamped loader the rest of the web layer uses — never a
    fresh ``load_global_config()``, so project overrides are honored) — and
    sorts by the v1 ``_KLASS_RANK`` triage order (stable sort: ``created_at``
    order is preserved within a rank). Identifier fields for the links come
    from one ``works`` SELECT here; ``corpus_rows`` itself stays id-less.
    """
    from sqlmodel import Session, select

    from ..acquisition.links import resolver_links
    from ..config.loader import load_project_config
    from ..db.project_models import Work

    missing = [
        r
        for r in rows
        if not r.get("has_markdown") and r.get("inclusion_status") != "excluded"
    ]
    if not missing:
        return []
    cfg = load_project_config(handle.slug, getattr(handle, "root", None))
    ids = [r["work_id"] for r in missing]
    with Session(handle.engine) as session:
        works = {
            w.work_id: w
            for w in session.exec(select(Work).where(Work.work_id.in_(ids))).all()
        }
    out: list[dict[str, Any]] = []
    for r in missing:
        w = works.get(r["work_id"])
        id_row = {
            "doi": w.doi if w else None,
            "arxiv_id": w.arxiv_id if w else None,
            "openalex_id": w.openalex_id if w else None,
            "semantic_scholar_id": w.semantic_scholar_id if w else None,
            "ssrn_id": w.ssrn_id if w else None,
            "title": r.get("title"),
            "authors": (w.authors if w else None) or [],
            "year": r.get("year"),
            "venue": w.venue if w else None,
        }
        out.append(
            {
                **r,
                "links": resolver_links(
                    id_row,
                    openurl_resolver=cfg.openurl_resolver,
                    ezproxy_host=cfg.ezproxy_host,
                ),
            }
        )
    out.sort(key=lambda r: _FRONTIER_STATE_RANK.get(r["acquisition_state"], 1))
    return out


@ui_router.get(
    "/ui/projects/{slug}/corpus", response_class=HTMLResponse, dependencies=_GATE
)
def ui_corpus(request: Request, slug: str, status: str | None = None) -> HTMLResponse:
    """Corpus table (read-only) with an optional ``inclusion_status`` filter,
    plus the missing-paper frontier pane (Build D ch12, gap scan §5.6 gaps 2+3):
    K-of-N provided counter + the park-and-drop triage rows. Graph building was
    never blocked on unprovided papers and stays that way — the pane only lists
    what to do next per row."""
    from ..acquisition.service import corpus_rows
    from ..progress import conversion_summary

    handle = _open_project(slug)
    statuses = (status,) if status else None
    rows = corpus_rows(handle, statuses=statuses, cache_root=None)
    queue_status = marker_queue.status(slug)
    # K-of-N folds over the SAME rows the table + conversion bar render (one
    # source of truth with ``conversion_summary``): K = rows with markdown,
    # N = all non-excluded rows in this view.
    eligible = [r for r in rows if r.get("inclusion_status") != "excluded"]
    provided_counts = {
        "provided": sum(1 for r in eligible if r.get("has_markdown")),
        "total": len(eligible),
    }
    return render(
        request,
        "corpus.html",
        slug=slug,
        rows=rows,
        filter_status=status,
        statuses=("included", "metadata_only", "excluded"),
        queue_status=queue_status,
        conversion=conversion_summary(rows, queue_status),
        frontier_rows=_frontier_rows(handle, rows),
        provided_counts=provided_counts,
    )


@ui_router.get(
    "/ui/projects/{slug}/corpus/{work_id}",
    response_class=HTMLResponse,
    dependencies=_GATE,
)
def ui_corpus_drawer(request: Request, slug: str, work_id: str) -> HTMLResponse:
    """Row-detail drawer for one work — the ONLY screen that renders full text.

    Membership-validated: the work must be in this project's corpus (else 404, no
    path in the message). Full markdown text is read through the read-only cache and
    is rendered only here (behind ``require_local_session``); ``access_class`` is
    surfaced so a private source is visibly private."""
    import sqlite3

    from .. import cache_access
    from ..acquisition.bridge import resolve_work_markdown
    from ..db.adapter import raw_conn
    from ..extraction.runner import resolve_source_access_class
    from sqlmodel import Session

    handle = _open_project(slug)
    full_text: str | None = None
    access_class: str | None = None
    row: dict[str, Any] | None = None
    has_markdown = False
    with Session(handle.engine, expire_on_commit=False) as session:
        conn = raw_conn(session)
        meta = conn.execute(
            "SELECT w.work_id, w.canonical_title, w.year, d.inclusion_status, "
            "d.access_status, d.is_seed FROM works w "
            "JOIN project_documents d ON d.work_id = w.work_id WHERE w.work_id = ?",
            (work_id,),
        ).fetchone()
        if meta is None:
            raise HTTPException(status_code=404, detail="work not in corpus")
        row = {
            "work_id": meta[0],
            "title": meta[1],
            "year": meta[2],
            "inclusion_status": meta[3],
            "access_status": meta[4],
            "is_seed": bool(meta[5]),
        }
        resolved = resolve_work_markdown(session, work_id=work_id)
        if resolved is not None:
            markdown_id, _markdown_hash = resolved
            cache_conn = cache_access.open_cache_ro(None)
            try:
                access_class = resolve_source_access_class(cache_conn, markdown_id)
                md = cache_access.read_markdown(cache_conn, None, markdown_id)
            finally:
                cache_conn.close()
            if md is not None:
                full_text = md.text
                has_markdown = True
                access_class = md.access_class
    return render(
        request,
        "corpus_drawer.html",
        slug=slug,
        row=row,
        full_text=full_text,
        has_markdown=has_markdown,
        access_class=access_class,
        statuses=("included", "metadata_only", "excluded"),
    )


@ui_router.post(
    "/ui/projects/{slug}/corpus/{work_id}/status", dependencies=_POST_GATE
)
async def ui_corpus_set_status(request: Request, slug: str, work_id: str):
    """Transition a work's ``inclusion_status`` (the ``project set-status`` lift).

    Calls the same :func:`project.service.set_inclusion_status` the CLI does, so an
    invalid status or an unknown ``work_id`` is rejected with the identical message
    (surfaced as a 400). ``access_status`` stays NULL (decision 30). 303→ the corpus."""
    handle = _open_project(slug)
    form = await request.form()
    status = str(form.get("status", ""))
    reason = form.get("reason")
    reason = str(reason) if reason not in (None, "") else None
    try:
        project_service.set_inclusion_status(handle, work_id, status, reason)
    except SeedgraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _redirect(f"/ui/projects/{slug}/corpus")


@ui_router.post("/ui/projects/{slug}/upload", dependencies=_POST_GATE)
async def ui_upload_seeds(request: Request, slug: str):
    """Upload one or more seed PDFs: each becomes a seed work, is INGESTED inline
    (fast: hash/dedup/store), then ENQUEUED for background Marker conversion.

    The slow PDF->markdown convert no longer blocks the request: a process-global
    worker pool (N = GPU slots) drains the queue at most N papers at a time (the rest
    wait FIFO). The corpus page shows per-work conversion status (queued / converting
    / failed) and live-updates as conversions complete. Non-PDF uploads are skipped
    (``failed``). 303-> the corpus with ``queued`` / ``failed`` counts.
    """
    import os
    import tempfile
    from pathlib import Path

    from ..acquisition.service import ingest_upload

    handle = _open_project(slug)
    form = await request.form()
    uploads = [f for f in form.getlist("files") if getattr(f, "filename", "")]
    queued = failed = 0
    for up in uploads:
        data = await up.read()
        if not data.startswith(b"%PDF"):
            failed += 1
            continue
        work = project_service.add_work(
            handle, title=_title_from_filename(up.filename), is_seed=True,
            inclusion_status="included",
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp.write(data)
            tmp.close()
            src = ingest_upload(Path(tmp.name), cache_root=None)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        marker_queue.enqueue(slug, work.work_id, src.source_file_id, src.file_hash)
        queued += 1
    return _redirect(f"/ui/projects/{slug}/corpus?queued={queued}&failed={failed}")


@ui_router.post("/ui/projects/{slug}/ingest-folder", dependencies=_POST_GATE)
async def ui_ingest_folder(request: Request, slug: str):
    """Bulk drop-zone: ingest + auto-match one or more PDFs (Phase-2 sibling of /upload).

    Each %PDF-guarded upload is written to a temp file (preserving its basename for
    provenance) and run SYNCHRONOUSLY through :func:`acquisition.service.ingest_one_pdf`
    (deterministic DOI→arXiv→title match, OCR-fallback convert, bridge/route). Matched
    works are bridged in place; misses route to ``review_queue`` as ``unmatched_upload``
    items. Non-PDF uploads are skipped (``failed``). 303→ the corpus with outcome
    counts. Synchronous convert is acceptable for the small batches a drop-zone submits;
    the queue-backed variant is deferred."""
    import os
    import shutil
    import tempfile
    from pathlib import Path, PurePosixPath

    from ..acquisition.service import ingest_one_pdf

    handle = _open_project(slug)
    form = await request.form()
    uploads = [f for f in form.getlist("files") if getattr(f, "filename", "")]
    matched = promoted = review = failed = 0
    for up in uploads:
        data = await up.read()
        if not data.startswith(b"%PDF"):
            failed += 1
            continue
        tmpdir = tempfile.mkdtemp()
        safe_name = PurePosixPath(up.filename).name or "upload.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        tmp = Path(tmpdir) / safe_name
        try:
            tmp.write_bytes(data)
            result = ingest_one_pdf(
                handle, tmp, cache_root=None, promote=True, promote_unmatched=False
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        outcome = result["outcome"]
        if outcome == "promoted":
            promoted += 1
        elif outcome in ("unmatched_review", "ambiguous_review"):
            review += 1
        elif outcome and (outcome.startswith("matched") or outcome == "already_bridged"):
            matched += 1
        else:  # conversion_failed / unexpected
            failed += 1
    return _redirect(
        f"/ui/projects/{slug}/corpus?matched={matched}&promoted={promoted}"
        f"&review={review}&failed={failed}"
    )


@ui_router.get("/ui/projects/{slug}/upload/status", dependencies=_GATE)
def ui_upload_status(slug: str) -> JSONResponse:
    """Per-work background-conversion status for the corpus poller (JSON, gated).

    Returns ``{work_id: {"state","detail"}}`` for this project's works that are
    currently queued / converting / failed; a work absent from the map has either
    never been enqueued or has finished (its corpus row's ``has_markdown`` is then the
    source of truth). Same loopback+cookie gate as every private-evidence screen."""
    return JSONResponse(marker_queue.status(slug))


# --- answer workspace (gated; synchronous no_llm / retrieval-only) -----------


@ui_router.get("/ui/projects/{slug}/ask", response_class=HTMLResponse, dependencies=_GATE)
def ui_ask(
    request: Request,
    slug: str,
    q: str | None = None,
    mode: str = "project_only",
    no_llm: bool = True,
) -> HTMLResponse:
    """Question form + a synchronous answer render path.

    With no ``q`` it renders the empty form. With a ``q`` it runs the in-process
    :func:`answer.harness.answer` (``no_llm`` defaults true in Stage A — the
    real-LLM launch lands once Track 1 is wired into the planner), then renders the
    :class:`AnswerEnvelope`: prose (empty in retrieval-only) + citations whose
    work-access-class is enriched from ``list_documents``, plus recommendations and
    warnings. Never crashes when no model is configured — it surfaces the honest
    retrieval-only / insufficient-evidence state."""
    handle = _open_project(slug)
    if mode not in ("project_only", "allow_outside"):
        raise HTTPException(status_code=400, detail="mode must be project_only or allow_outside")

    envelope: dict[str, Any] | None = None
    trace_url: str | None = None
    if q:
        from ..answer.harness import answer as answer_fn, save_answer
        from ..answer.trace import save_trace
        from ..answer.types import AnswerMode

        env, trace = answer_fn(q, handle, mode=AnswerMode(mode), no_llm=no_llm)
        # GET-with-q persists envelope + trace, then renders (00 §5; a refresh re-runs
        # the ask and mints a new pair — accepted).
        root = getattr(handle, "root", None)
        save_answer(env, slug=slug, root=root)
        save_trace(trace, slug=slug, root=root)
        envelope = _enrich_envelope(env.model_dump(), handle)
        trace_url = f"/ui/projects/{slug}/answers/{env.answer_id}/trace"

    return render(
        request,
        "ask.html",
        slug=slug,
        question=q or "",
        mode=mode,
        no_llm=no_llm,
        envelope=envelope,
        trace_url=trace_url,
    )


@ui_router.post("/ui/projects/{slug}/ask", dependencies=_POST_GATE)
async def ui_ask_submit(request: Request, slug: str):
    """Submit a question (the answer workspace mutating path; require_local_session + CSRF).

    Without the ``use_llm`` toggle — or when no usable answer backend is configured —
    the answer is computed **synchronously** and rendered inline (retrieval-only /
    no-LLM honest degradation; never a crash). With the toggle AND a usable backend a
    real prose answer runs as a background job (the worker calls
    :func:`answer.harness.answer` through the executor, saves the envelope under the
    job's run, and emits progress) and we 303→ the run page, which streams progress
    then re-renders with the saved envelope."""
    from ..answer.harness import answer as answer_fn, save_answer
    from ..answer.trace import save_trace
    from ..answer.types import AnswerMode
    from ..config.loader import load_project_config

    handle = _open_project(slug)
    form = await request.form()
    q = str(form.get("q", "")).strip()
    mode = str(form.get("mode", "project_only"))
    use_llm = form.get("use_llm") is not None
    if mode not in ("project_only", "allow_outside"):
        raise HTTPException(status_code=400, detail="mode must be project_only or allow_outside")
    if not q:
        return _redirect(f"/ui/projects/{slug}/ask")

    cfg = load_project_config(slug, getattr(handle, "root", None))
    backend_ok = _answer_backend_available(cfg)

    # Synchronous path: no-LLM requested, or no usable backend (degrade to retrieval-only).
    if not use_llm or not backend_ok:
        env, trace = answer_fn(q, handle, mode=AnswerMode(mode), no_llm=not use_llm, config=cfg)
        # Sync POST persists envelope + trace, then renders (00 §5).
        root = getattr(handle, "root", None)
        save_answer(env, slug=slug, root=root)
        save_trace(trace, slug=slug, root=root)
        return render(
            request,
            "ask.html",
            slug=slug,
            question=q,
            mode=mode,
            no_llm=not use_llm,
            envelope=_enrich_envelope(env.model_dump(), handle),
            degraded_no_backend=bool(use_llm and not backend_ok),
            trace_url=f"/ui/projects/{slug}/answers/{env.answer_id}/trace",
        )

    # Real LLM prose answer → background job (own connection inside the worker).
    def _answer_job(emit, h) -> None:
        run_id = emit.run_id
        emit("answer", "generating prose answer")
        env, trace = answer_fn(
            q, h, mode=AnswerMode(mode), no_llm=False,
            config=load_project_config(h.slug, getattr(h, "root", None)),
            run_id=run_id,
        )
        # Background LLM job saves envelope + trace under the run (00 §5).
        save_answer(env, slug=h.slug, root=getattr(h, "root", None), run_id=run_id)
        save_trace(trace, slug=h.slug, root=getattr(h, "root", None), run_id=run_id)
        emit(
            "answer",
            f"answer ready ({env.answer_category.value})",
            answer_id=env.answer_id,
            category=env.answer_category.value,
        )

    try:
        run_id = jobs.launch_job(
            slug, phase="answer", fn=_answer_job, root=getattr(handle, "root", None)
        )
    except jobs.ProjectBusyError as exc:
        return render(
            request, "ask.html", slug=slug, question=q, mode=mode, no_llm=False,
            envelope=None, busy_message=str(exc), status_code=409,
        )
    return _redirect(f"/ui/projects/{slug}/runs/{run_id}?kind=answer")


# --- run planner + run progress (gated) --------------------------------------


@ui_router.get("/ui/projects/{slug}/plan", response_class=HTMLResponse, dependencies=_GATE)
def ui_plan(request: Request, slug: str, task: str = "extraction") -> HTMLResponse:
    """Run-planner preview for ``task`` (affected works + route/profile + cost + privacy).

    Pure read over :func:`web.planner.plan_job`; surfaces an "awaiting LLM backend"
    banner only when the resolved route has no usable profile. Launching happens via
    the POST below."""
    from .planner import TASK_LABELS, TASK_NAMES, plan_job

    handle = _open_project(slug)
    if task not in TASK_NAMES:
        raise HTTPException(status_code=404, detail=f"unknown task {task!r}")
    try:
        plan = plan_job(handle, task)
    except SeedgraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return render(
        request, "planner.html", slug=slug, plan=plan,
        tasks=TASK_NAMES, labels=TASK_LABELS, task=task,
    )


@ui_router.post("/ui/projects/{slug}/plan/{task}", dependencies=_POST_GATE)
async def ui_plan_launch(request: Request, slug: str, task: str):
    """Launch the planned ``task`` as a background job (require_local_session + CSRF).

    Re-plans with the submitted flags, dispatches through
    :func:`web.planner.launch_planned_job`, and 303→ the new run's progress page. A
    second concurrent job for the project is rejected with a friendly busy message
    (409 — :class:`web.jobs.ProjectBusyError`), not a crash."""
    from .planner import TASK_LABELS, TASK_NAMES, launch_planned_job, plan_job

    handle = _open_project(slug)
    if task not in TASK_NAMES:
        raise HTTPException(status_code=404, detail=f"unknown task {task!r}")
    form = await request.form()
    profile_id = str(form.get("profile_id", "")).strip() or None
    flags = {
        "profile_id": profile_id,
        "force": form.get("force") is not None,
        "only_failed": form.get("only_failed") is not None,
        "only_stale": form.get("only_stale") is not None,
        "lens_id": str(form.get("lens_id", "")).strip() or None,
    }
    try:
        plan = plan_job(
            handle, task, profile_id=profile_id, force=flags["force"],
            only_failed=flags["only_failed"], only_stale=flags["only_stale"],
        )
    except SeedgraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        run_id = launch_planned_job(handle, plan, flags)
    except jobs.ProjectBusyError as exc:
        return render(
            request, "planner.html", slug=slug, plan=plan,
            tasks=TASK_NAMES, labels=TASK_LABELS, task=task,
            busy_message=str(exc), status_code=409,
        )
    return _redirect(f"/ui/projects/{slug}/runs/{run_id}")


@ui_router.get(
    "/ui/projects/{slug}/runs/{run_id}", response_class=HTMLResponse, dependencies=_GATE
)
def ui_run_progress(
    request: Request, slug: str, run_id: str, kind: str | None = None
) -> HTMLResponse:
    """Run progress page: server-rendered events + JS polling until a terminal status.

    Polls ``GET /api/.../runs/{run_id}/events`` (poll.js) until ``finished``/``failed``;
    on the first terminal status the page reloads so the final server state — including
    any saved answer envelope (the answer-job artifact) — renders. ``run_id`` is path-
    validated (never concatenated unsafely)."""
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=404, detail="invalid run id")
    from ..run import _status_from_events, read_events

    handle = _open_project(slug)
    events = read_events(slug, run_id, root=getattr(handle, "root", None))
    status = _status_from_events(events)
    envelope = _load_run_answer(handle, run_id)
    # Link the run-scoped trace view only when the sibling trace file exists (a
    # pre-feature run answer has an envelope but no trace — risk #6).
    trace_url = None
    if envelope is not None:
        _, trace_path = trace_answer_paths(handle, envelope["answer_id"], run_id=run_id)
        if trace_path.exists():
            trace_url = (
                f"/ui/projects/{slug}/runs/{run_id}/answers/{envelope['answer_id']}/trace"
            )
        envelope = _enrich_envelope(envelope, handle)
    return render(
        request, "run.html", slug=slug, run_id=run_id,
        events=events, status=status, kind=kind, envelope=envelope,
        trace_url=trace_url,
    )


# --- answer trace view + explorable subgraph (gated) -------------------------


class InvalidTraceAnswerId(ValueError):
    """``answer_id`` contained a path separator (``/``, ``\\``) or ``..``.

    Raised at the choke point (:func:`trace_answer_paths`) so both the web trace routes
    and the CLI ``answer trace-export`` inherit the guard through their shared call: the
    ``answer_id`` is interpolated into a filename, so a traversal-shaped value must never
    reach the filesystem. A dedicated subclass (not a bare ``ValueError``) keeps the catch
    precise — a corrupt trace JSON (``json.JSONDecodeError``, also a ``ValueError``) is not
    swallowed as "invalid id"."""


def trace_answer_paths(handle, answer_id: str, *, run_id: str | None = None):
    """(envelope_path, trace_path) for ``answer_id`` — mirrors ``save_trace``/``save_answer``
    path logic (ad-hoc ``answers/`` vs run-nested ``runs/{run_id}/answers/``). Public: the
    chunk-4 CLI export command also calls this to derive its default output path.

    ``answer_id`` is interpolated into the filename, so it is validated here (the choke
    point every caller shares): a ``/``, ``\\``, or ``..`` raises :class:`InvalidTraceAnswerId`
    before any path is built — matching the ``run_id`` traversal guard on the routes."""
    from ..project import layout

    if "/" in answer_id or "\\" in answer_id or ".." in answer_id:
        raise InvalidTraceAnswerId(f"invalid answer id: {answer_id!r}")
    if run_id:
        base = layout.project_runs_dir(handle.slug, handle.root) / run_id / "answers"
    else:
        base = layout.project_dir(handle.slug, handle.root) / "answers"
    return base / f"{answer_id}.json", base / f"{answer_id}.trace.json"


def _work_titles(handle, work_ids) -> dict[str, dict[str, Any]]:
    """``{work_id: {"title", "year"}}`` for the given ids — the trace-view enrichment
    query. A work absent from ``works`` is simply omitted (the view renders "unknown")."""
    import sqlite3

    ids = [w for w in dict.fromkeys(work_ids) if w]
    if not ids:
        return {}
    conn = sqlite3.connect(str(handle.db_path))
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT work_id, canonical_title, year FROM works WHERE work_id IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        conn.close()
    return {r[0]: {"title": r[1], "year": r[2]} for r in rows}


def build_trace_context(
    handle, answer_id: str, *, run_id: str | None = None
) -> dict[str, Any] | None:
    """Shared trace-view read-model (trace_plans 00 §6.1).

    Loads the envelope + trace JSON for ``answer_id`` (ad-hoc or run-scoped), runs the
    enrichment query (work titles/years for candidate rows + neighborhood nodes), builds
    the works-only subgraph payload via the pure
    :func:`web.trace_view.build_trace_subgraph_payload`, and returns the ``trace.html``
    context dict. Returns ``None`` when the answer or its trace file is absent (the route
    404s, and pre-feature answers have no trace). The chunk-4 export command reuses this
    verbatim to build the same context outside FastAPI — keep it request-free.
    """
    import json

    from .trace_view import build_trace_subgraph_payload

    env_path, trace_path = trace_answer_paths(handle, answer_id, run_id=run_id)
    if not env_path.exists() or not trace_path.exists():
        return None
    envelope = json.loads(env_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    candidates = sorted(
        trace.get("candidates", []) or [], key=lambda c: c.get("rank_position", 0)
    )
    neighborhood = trace.get("neighborhood")

    # enrichment: candidate work ids + neighborhood node ids -> title/year.
    work_ids: set[str] = {c["work_id"] for c in candidates}
    if neighborhood:
        work_ids.update(neighborhood.get("nodes", []) or [])
    work_meta = _work_titles(handle, work_ids)
    for c in candidates:  # attach display title/year (None -> "unknown" in the template)
        meta = work_meta.get(c["work_id"]) or {}
        c["title"] = meta.get("title")
        c["year"] = meta.get("year")

    subgraph = None
    if neighborhood:
        rec_ids = [r.get("work_id") for r in envelope.get("recommendations", []) or []]
        subgraph = build_trace_subgraph_payload(
            neighborhood, recommendation_work_ids=rec_ids, work_meta=work_meta
        )

    # shown rows expand to their full fragment; a miss (no prompt built on this path)
    # falls back to the row's text_preview (advisor trap: shown_evidence can be None).
    evidence_by_item = {
        se["item_id"]: se for se in (trace.get("shown_evidence") or [])
    }
    divergence = trace.get("spec", {}).get("protocol_hint") != envelope.get("query_type")
    back_url = (
        f"/ui/projects/{handle.slug}/runs/{run_id}"
        if run_id
        else f"/ui/projects/{handle.slug}/ask"
    )
    return {
        "slug": handle.slug,
        "answer_id": answer_id,
        "run_id": run_id,
        "envelope": envelope,
        "trace": trace,
        "candidates": candidates,
        "work_meta": work_meta,
        "evidence_by_item": evidence_by_item,
        "neighborhood": neighborhood,
        "subgraph": subgraph,
        "divergence": divergence,
        "back_url": back_url,
    }


def resolve_trace_context(handle, answer_id: str) -> dict[str, Any] | None:
    """Locate ``answer_id``'s trace context without knowing in advance whether it is
    ad-hoc or run-scoped (chunk 4's export command; the web routes already know from
    the URL, so they call :func:`build_trace_context` directly). Tries the ad-hoc
    ``answers/`` location first, then searches ``runs/*/answers/`` for the sibling
    pair. Returns ``None`` when neither location has it — the caller turns that into
    a clean error, never a traceback."""
    try:
        ctx = build_trace_context(handle, answer_id, run_id=None)
    except InvalidTraceAnswerId:
        # Traversal-shaped answer_id: no location can hold it — treat as "not found" so the
        # CLI export prints its clean error and exits non-zero (never a traceback), and the
        # runs/*/answers/ scan below (which also interpolates answer_id) is never reached.
        return None
    if ctx is not None:
        return ctx

    from ..project import layout

    runs_dir = layout.project_runs_dir(handle.slug, handle.root)
    if not runs_dir.exists():
        return None
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if (run_dir / "answers" / f"{answer_id}.trace.json").exists():
            return build_trace_context(handle, answer_id, run_id=run_dir.name)
    return None


# Vendored subgraph JS files, keyed for the standalone export's inline block (must
# match the ``vendor_js.*`` lookups in ``trace.html``'s ``{% if standalone %}`` arm).
_VENDOR_JS_FILES = {
    "three": "three.min.js",
    "spritetext": "three-spritetext.min.js",
    "force_graph": "3d-force-graph.min.js",
}


def render_trace_standalone(ctx: dict[str, Any]) -> str:
    """Render ``trace.html`` with ``standalone=True`` for the export command (chunk 4,
    trace_plans 00 §6.1: one template, two consumers). Same Jinja environment as the
    served route (``templates`` module-level above) — the export is not a fork of the
    view, only a different value for the one ``standalone`` conditional.

    The vendored subgraph JS (~1.4 MB) is read from disk and inlined ONLY when
    ``ctx["neighborhood"]`` is present — a retrieval-only trace exports without that
    payload, same as the neighborhood guard already does for the served page.
    """
    vendor_js = None
    if ctx.get("neighborhood"):
        vendor_js = {
            key: (STATIC_DIR / "vendor" / filename).read_text(encoding="utf-8")
            for key, filename in _VENDOR_JS_FILES.items()
        }
    template = templates.get_template("trace.html")
    return template.render(**ctx, standalone=True, vendor_js=vendor_js)


def _render_trace(
    request: Request, slug: str, answer_id: str, *, run_id: str | None
) -> HTMLResponse:
    handle = _open_project(slug)
    try:
        ctx = build_trace_context(handle, answer_id, run_id=run_id)
    except InvalidTraceAnswerId:
        # Traversal-shaped answer_id: the same plain 404 the run_id guard gives, never a
        # file read or the friendly "no trace" page (which would imply a valid-but-missing id).
        raise HTTPException(status_code=404, detail="invalid answer id")
    if ctx is None:
        back_url = (
            f"/ui/projects/{slug}/runs/{run_id}" if run_id else f"/ui/projects/{slug}/ask"
        )
        return render(
            request, "trace_missing.html", slug=slug, answer_id=answer_id,
            back_url=back_url, status_code=404,
        )
    return render(request, "trace.html", **ctx)


@ui_router.get(
    "/ui/projects/{slug}/answers/{answer_id}/trace",
    response_class=HTMLResponse,
    dependencies=_GATE,
)
def ui_answer_trace(request: Request, slug: str, answer_id: str) -> HTMLResponse:
    """AnswerTrace view for an ad-hoc ask (``answers/{answer_id}.trace.json``).

    Renders the query classification, disposition-tagged candidate table, and — when
    the trace captured a citation neighborhood — the explorable works-only subgraph.
    A pre-feature answer (envelope but no trace) or an unknown id yields a friendly
    404 page."""
    return _render_trace(request, slug, answer_id, run_id=None)


@ui_router.get(
    "/ui/projects/{slug}/runs/{run_id}/answers/{answer_id}/trace",
    response_class=HTMLResponse,
    dependencies=_GATE,
)
def ui_run_answer_trace(
    request: Request, slug: str, run_id: str, answer_id: str
) -> HTMLResponse:
    """AnswerTrace view for a run-scoped answer (background LLM job artifact)."""
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=404, detail="invalid run id")
    return _render_trace(request, slug, answer_id, run_id=run_id)


# --- lens list + detail (gated) ----------------------------------------------


@ui_router.get(
    "/ui/projects/{slug}/lenses", response_class=HTMLResponse, dependencies=_GATE
)
def ui_lenses(request: Request, slug: str) -> HTMLResponse:
    """Registered-lens list (id/name/status/object_type) over ``registry.list_lenses``."""
    from sqlmodel import Session

    from ..lenses import registry

    handle = _open_project(slug)
    with Session(handle.engine) as session:
        rows = registry.list_lenses(session)
    lenses = [
        {
            "lens_id": r.lens_id,
            "name": r.name,
            "scope": r.scope,
            "object_type": r.object_type,
            "status": r.status,
        }
        for r in rows
    ]
    return render(request, "lenses.html", slug=slug, lenses=lenses)


@ui_router.get(
    "/ui/projects/{slug}/lenses/{lens_id}",
    response_class=HTMLResponse,
    dependencies=_GATE,
)
def ui_lens_detail(
    request: Request, slug: str, lens_id: str, status: str = "found"
) -> HTMLResponse:
    """One lens: result rows + coverage/staleness (lifted services; no duplication)."""
    from sqlmodel import Session

    from ..lenses import registry
    from ..lenses import runner as lens_runner

    handle = _open_project(slug)
    project_dir = handle.root / "projects" / handle.slug
    with Session(handle.engine) as session:
        results = lens_runner.lens_results(session, lens_id, status=status)
        staleness = registry.lens_staleness(session, project_dir, lens_id)
    project_dir = handle.root / "projects" / handle.slug
    yaml_text = ""
    yaml_path = registry.lens_yaml_path(project_dir, lens_id)
    if yaml_path.exists():
        yaml_text = yaml_path.read_text(encoding="utf-8")
    return render(
        request,
        "lens_detail.html",
        slug=slug,
        lens_id=lens_id,
        status=status,
        results=results,
        staleness=staleness,
        yaml_text=yaml_text,
    )


@ui_router.post("/ui/projects/{slug}/lenses", dependencies=_POST_GATE)
async def ui_lens_new(request: Request, slug: str):
    """Scaffold a new lens from a built-in template (the ``lens new`` lift).

    Delegates to :func:`lenses.registry.create_lens_from_template` (the same path the
    CLI uses); an unknown template or an already-existing lens id is rejected with the
    identical message (surfaced as a 400). 303→ the lens list."""
    from sqlmodel import Session

    from ..lenses import registry

    handle = _open_project(slug)
    form = await request.form()
    lens_id = str(form.get("lens_id", "")).strip()
    from_template = str(form.get("from_template", "")).strip()
    project_dir = handle.root / "projects" / handle.slug
    try:
        with Session(handle.engine) as session:
            registry.create_lens_from_template(session, project_dir, lens_id, from_template)
    except SeedgraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _redirect(f"/ui/projects/{slug}/lenses")


@ui_router.post(
    "/ui/projects/{slug}/lenses/{lens_id}/validate", dependencies=_POST_GATE
)
async def ui_lens_validate(request: Request, slug: str, lens_id: str):
    """Validate a lens YAML textarea and (if valid) write it to the lens file.

    Validation is the same :func:`registry.load_lens`/``LensDefinition`` parse the
    ``lens validate`` CLI runs; an invalid document is rejected with the identical
    ``INVALID`` message (400) and **nothing is written**. On success the file is
    written, the registry row resynced, and we 303→ the lens detail."""
    from sqlmodel import Session

    from ..lenses import registry

    handle = _open_project(slug)
    form = await request.form()
    yaml_text = str(form.get("yaml", ""))
    project_dir = handle.root / "projects" / handle.slug
    try:
        with Session(handle.engine) as session:
            registry.write_lens_yaml(session, project_dir, lens_id, yaml_text)
    except SeedgraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _redirect(f"/ui/projects/{slug}/lenses/{lens_id}")


# --- review queue (gated) ----------------------------------------------------


@ui_router.get(
    "/ui/projects/{slug}/review", response_class=HTMLResponse, dependencies=_GATE
)
def ui_review(request: Request, slug: str) -> HTMLResponse:
    """Open review-queue items + a per-item resolve form (the ``review list`` lift)."""
    from ..project import review as review_mod
    from ..project.review import review_rows  # the shared row-shaping seam

    handle = _open_project(slug)
    # Same rows as `seedgraph review list`: each dict carries the human
    # ``decision`` line (merge pair / citation fields) plus the per-item-type
    # action/input menu (``it.menu``) alongside the ids.
    items = review_rows(handle, review_mod.list_open(handle))
    return render(
        request,
        "review.html",
        slug=slug,
        items=items,
    )


@ui_router.post(
    "/ui/projects/{slug}/review/{item_id}/resolve", dependencies=_POST_GATE
)
async def ui_review_resolve(request: Request, slug: str, item_id: str):
    """Resolve a review item (the ``review resolve`` lift; idempotent).

    Calls :func:`project.review.resolve` exactly as the CLI does, so an invalid action
    or an unknown item id is rejected with the identical message (400). 303→ review."""
    from ..project import review as review_mod

    handle = _open_project(slug)
    form = await request.form()
    action = str(form.get("action", ""))
    # Optional applier knobs (Build A ch7/8) — absent/blank fields keep defaults.
    survivor_id = str(form.get("survivor_id", "")).strip() or None
    target_work_id = str(form.get("target_work_id", "")).strip() or None
    raw_candidate = str(form.get("candidate_index", "")).strip()
    candidate_index: int | None = None
    if raw_candidate:
        try:
            candidate_index = int(raw_candidate)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"candidate_index must be an integer, got {raw_candidate!r}"
            ) from exc
    try:
        review_mod.resolve(
            handle, item_id, action,
            survivor_id=survivor_id, candidate_index=candidate_index,
            target_work_id=target_work_id,
        )
    except SeedgraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _redirect(f"/ui/projects/{slug}/review")


# --- settings (gated) --------------------------------------------------------


@ui_router.get("/ui/settings", response_class=HTMLResponse, dependencies=_GATE)
def ui_global_settings(request: Request) -> HTMLResponse:
    """Global settings view (profile/route matrix, privacy, budget, doctor)."""
    from .routes import settings_view

    return render(request, "settings.html", slug=None, view=settings_view())


@ui_router.post("/ui/settings", dependencies=_POST_GATE)
async def ui_global_settings_save(request: Request):
    """Persist global settings (privacy/budget/routes) via ``write_global_config``.

    Same validation as the writer (a bad patch raises :class:`ConfigError`, surfaced
    verbatim as a 400; nothing is written). 303→ the settings screen."""
    from ..config.loader import write_global_config
    from ..errors import ConfigError

    form = await request.form()
    try:
        write_global_config(_global_config_patch(form))
    except (ConfigError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _redirect("/ui/settings")


@ui_router.post("/ui/projects/{slug}/settings", dependencies=_POST_GATE)
async def ui_project_settings_save(request: Request, slug: str):
    """Persist project settings with LIVE inheritance (never the full-dump clobber path).

    Three concerns, three clobber-safe writers on the SHARED ``project.yaml``:

    * ``revert=<group>[:field]`` — remove one override key so the field re-inherits the
      live global default (:func:`_revert_project_override`); 303 back.
    * identity + ``answer_policy`` — the DECLARATIVE fields, written via the
      overlay-preserving :func:`project.config.write_project_config` (which reads the
      file and overlays only the keys it owns, so it never deletes the override groups).
    * the four inheritance override groups (routing choice / ``content_policy`` /
      ``budget`` / ``answer``) — written as a THIN overlay via
      :func:`config.loader.write_project_overrides`, where an unchecked ``override``
      box OMITS the field (keeps inheriting global).

    A bad identity value, a loosening ceiling override, or an unknown route profile is
    rejected as a 400 BEFORE anything is written (validate-then-write ordering), so a
    rejected save leaves the file untouched. Preserves ``_POST_GATE`` (session + CSRF).
    """
    from pydantic import ValidationError as PydanticValidationError

    from ..config.loader import load_global_config, write_project_overrides
    from ..errors import ConfigError, ValidationError
    from ..project import layout
    from ..project.config import (
        ProjectConfig,
        load_project_config,
        write_project_config,
    )

    _open_project(slug)  # 404 (same message as the CLI) on a bad/unknown slug
    form = await request.form()

    # (a) revert-to-global: remove one override key, then 303 back.
    revert = form.get("revert")
    if revert not in (None, ""):
        _revert_project_override(slug, str(revert))
        return _redirect(f"/ui/projects/{slug}/settings")

    try:
        cfg = load_project_config(slug)
    except SeedgraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # (b) build + VALIDATE the declarative identity/answer_policy block (no write yet).
    data = cfg.model_dump()
    name = str(form.get("project_name", "")).strip()
    if name:
        data["project_name"] = name
    description = form.get("description")
    data["description"] = str(description).strip() or None if description is not None else data.get("description")
    data["answer_policy"] = {
        "require_project_sources": form.get("require_project_sources") is not None,
        "allow_external_search_by_default": form.get("allow_external_search_by_default")
        is not None,
        "distinguish_source_claims_from_synthesis": form.get(
            "distinguish_source_claims_from_synthesis"
        )
        is not None,
    }
    try:
        new_cfg = ProjectConfig(**data)
    except PydanticValidationError as exc:
        raise HTTPException(
            status_code=400, detail=str(ValidationError(f"invalid project.yaml: {exc}"))
        ) from exc

    # (c) the four override groups — validate (loosening + write validation) BEFORE any
    # write so a rejected save leaves project.yaml untouched (no partial write).
    patch = _project_config_patch(form)
    if patch:
        try:
            _assert_project_override_not_loosening(patch, load_global_config())
            write_project_overrides(slug, patch)
        except (ConfigError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # (d) declarative write LAST — already validated above, overlay-preserving so the
    # override groups written in (c) survive.
    write_project_config(new_cfg, layout.project_yaml_path(slug))
    return _redirect(f"/ui/projects/{slug}/settings")


@ui_router.get(
    "/ui/projects/{slug}/settings", response_class=HTMLResponse, dependencies=_GATE
)
def ui_project_settings(request: Request, slug: str) -> HTMLResponse:
    """Project-scoped settings view (adds the project's monthly spend).

    Also surfaces the editable ``project.yaml`` fields (name / description / answer
    policy) so the project-settings form pre-populates from the live config."""
    from ..project.config import load_project_config
    from .routes import settings_view

    _open_project(slug)
    try:
        project_cfg = load_project_config(slug).model_dump()
    except SeedgraphError:
        project_cfg = None
    return render(
        request,
        "settings.html",
        slug=slug,
        view=settings_view(slug=slug),
        project_cfg=project_cfg,
    )


def attach_ui(app: FastAPI) -> None:
    """Mount the static tree at ``/static`` and include the ``/ui`` HTML router."""
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(ui_router)
