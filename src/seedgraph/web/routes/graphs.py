"""Context graphs (UI design §2.3; stage 4 design §6-§8): the list and the graph page's
tabs - Overview, Concepts, Graph, Grow, Lenses.

GETs read :mod:`seedgraph.web.views.graphs` and never write. Each POST calls one deep
function: :func:`seedgraph.library.lifecycle.remove_from_graph`,
:func:`seedgraph.library.graphs.set_membership` (add, add a suggestion),
:func:`seedgraph.library.literatures.regroup` / ``step_resolution`` /
``rename_literature``, :func:`seedgraph.library.graphs.choose_lens` / ``unchoose_lens``,
or submits one :class:`~seedgraph.web.jobs.JobRunner` job (the growth walk, the growth
approval). A refused action redirects back with the reason as an error flash.
"""

from __future__ import annotations

from functools import partial

from starlette.routing import Route

from ...errors import SeedgraphError
from ...library import graphs, growth, lifecycle
from ...library import literatures as lits
from ..app import Flash, get_web
from ..jobs import JobResult
from ..views import graph_svg
from ..views import graphs as graphs_view
from ._pages import form, not_found, render
from .library import provider_chain

STATUSES = ("member", "frontier", "excluded")
NAV = "graphs"


def _tab(request, template: str, title: str, ctx: dict, tab: str):
    return render(request, f"graphs/{template}", {"page_title": title, "tabs": graphs_view.TABS, "tab": tab,
                                                  "concept_href": graphs_view.concept_href, **ctx}, nav=NAV)


def _back(request, tab: str = "", text: str = "", level: str = "success"):
    slug = request.path_params["slug"]
    url = f"/graphs/{slug}" + (f"/{tab}" if tab else "")
    return get_web(request).redirect(url, flash=[Flash(text, level)] if text else [])


def graphs_page(request):
    web = get_web(request)
    rows = graphs_view.list_page(web.library())
    return render(request, "graphs/list.html", {"page_title": "Context graphs", "rows": rows}, nav=NAV)


def graph_page(request):
    web = get_web(request)
    view = graphs_view.overview(web.library(), request.path_params["slug"])
    if view is None:
        return not_found("no such graph")
    return _tab(request, "detail.html", view.detail.display_name, {"view": view, "statuses": STATUSES}, "")


# --- papers -----------------------------------------------------------------------------------------

def remove(request):
    web = get_web(request)
    slug = request.path_params["slug"]
    if not graphs_view.exists(web.library(), slug):
        return not_found("no such graph")
    paper_id = str(form(request).get("paper_id") or "")
    try:
        result = lifecycle.remove_from_graph(web.library(), slug, paper_id)
    except SeedgraphError as exc:
        return web.redirect(f"/graphs/{slug}", flash=[Flash(f"not removed: {exc}", "error")])
    if not result.removed:
        return web.redirect(f"/graphs/{slug}", flash=[Flash(f"{paper_id} was not in this graph", "warning")])
    seed = " (it was a seed)" if result.was_seed else ""
    return web.redirect(f"/graphs/{slug}", flash=[Flash(f"removed {paper_id} from the graph{seed}", "success")])


def add(request):
    web = get_web(request)
    slug = request.path_params["slug"]
    if not graphs_view.exists(web.library(), slug):
        return not_found("no such graph")
    data = form(request)
    paper_id = str(data.get("paper_id") or "").strip()
    status = str(data.get("status") or "member")
    if status not in STATUSES:
        return web.redirect(f"/graphs/{slug}", flash=[Flash(f"unknown graph status {status!r}", "error")])
    try:
        with graphs.open_graph(web.library(), slug) as g:
            graphs.set_membership(g, paper_id, status, is_seed=bool(data.get("is_seed")))  # type: ignore[arg-type]
    except SeedgraphError as exc:
        return web.redirect(f"/graphs/{slug}", flash=[Flash(f"not added: {exc}", "error")])
    return web.redirect(f"/graphs/{slug}", flash=[Flash(f"added {paper_id} as {status}", "success")])


# --- literatures ------------------------------------------------------------------------------------

def _literature_post(request, action, done: str):
    web = get_web(request)
    slug = request.path_params["slug"]
    if not graphs_view.exists(web.library(), slug):
        return not_found("no such graph")
    data = form(request)
    try:
        with graphs.open_graph(web.library(), slug) as g:
            action(g, data)
    except SeedgraphError as exc:
        return _back(request, "", f"not done: {exc}", "error")
    return _back(request, "", done)


def regroup(request):
    return _literature_post(request, lambda g, data: lits.regroup(g), "literatures regrouped")


def step(request):
    return _literature_post(request, lambda g, data: lits.step_resolution(g, str(data.get("direction") or "")),
                            "literatures regrouped at the next resolution")


def rename_literature(request):
    def action(g, data):
        lits.rename_literature(g, str(data.get("literature_id") or ""), str(data.get("name") or ""))

    return _literature_post(request, action, "literature renamed")


# --- concepts and outside-graph suggestions ---------------------------------------------------------

def concepts_page(request):
    web = get_web(request)
    view = graphs_view.concepts_tab(web.library(), request.path_params["slug"])
    if view is None:
        return not_found("no such graph")
    return _tab(request, "concepts.html", f"{view.detail.display_name}: concepts", {"view": view}, "concepts")


def suggestion_confirm(request):
    web = get_web(request)
    slug = request.path_params["slug"]
    if not graphs_view.exists(web.library(), slug):
        return not_found("no such graph")
    view = graphs_view.suggestion_confirm(web.library(), slug, request.query_params.get("paper") or "")
    if view is None:
        return not_found("that paper is not a suggestion for this graph")
    return _tab(request, "confirm.html", f"{view.detail.display_name}: add a paper", {"view": view}, "concepts")


def suggestion_add(request):
    web = get_web(request)
    slug = request.path_params["slug"]
    if not graphs_view.exists(web.library(), slug):
        return not_found("no such graph")
    paper_id = str(form(request).get("paper_id") or "").strip()
    view = graphs_view.suggestion_confirm(web.library(), slug, paper_id)
    if view is None:
        return _back(request, "concepts", f"not added: {paper_id} is not a suggestion for this graph", "error")
    status = view.paper.joins_as
    try:
        with graphs.open_graph(web.library(), slug) as g:
            graphs.set_membership(g, paper_id, status, reason=f"shares {view.paper.via}")  # type: ignore[arg-type]
    except SeedgraphError as exc:
        return _back(request, "concepts", f"not added: {exc}", "error")
    return _back(request, "concepts", f"added {paper_id} as {status}")


# --- the 2D graph -----------------------------------------------------------------------------------

def _svg_view(request) -> str:
    return "citation" if request.query_params.get("view") == "citation" else "concept"


def graph_tab(request):
    web = get_web(request)
    d = graphs_view.detail(web.library(), request.path_params["slug"])
    if d is None:
        return not_found("no such graph")
    g = graph_svg.graph(web.library(), d.slug, _svg_view(request))
    return _tab(request, "graph.html", f"{d.display_name}: graph", {"detail": d, "g": g}, "graph")


def graph_partial(request):
    web = get_web(request)
    slug = request.path_params["slug"]
    if not graphs_view.exists(web.library(), slug):
        return not_found("no such graph")
    return web.render(request, "graphs/_svg.html", {"g": graph_svg.graph(web.library(), slug, _svg_view(request))})


# --- Grow: the guided growth wizard (jobs run on the JobRunner's general worker) ---------------------

def walk_job(lib, chain, slug: str, seeds: list[str], depth: int, cap: int) -> JobResult:
    """:func:`seedgraph.library.growth.walk` with the provider chain built on this thread."""
    with provider_chain(lib, chain) as ch:
        res = growth.walk(lib, slug, chain=ch, seeds=seeds or None, depth=depth, cap=cap)
    found = sum(g.found for g in res.generations)
    lines = [f"found {found} paper{'' if found == 1 else 's'} ({len(res.new_papers)} new to the library, "
             f"{res.links_added} new citation links); review them below"]
    lines += [f"generation {g.generation}: {g.provider_failures} provider failure(s)"
              for g in res.generations if g.provider_failures]
    if res.attention_item_ids:
        lines.append(f"{len(res.attention_item_ids)} reference(s) to check on the Review page")
    return JobResult("\n".join(lines), paper_ids=res.new_papers)


def parse_references_after(lib, paper_ids: tuple[str, ...], *, chain, only: frozenset[str]) -> str:
    """After-conversion hook: parse the references of the newly converted approved papers
    (:func:`seedgraph.library.growth.parse_references_for`)."""
    targets = [p for p in paper_ids if p in only]
    if not targets:
        return ""
    with provider_chain(lib, chain) as ch:
        results = [growth.parse_references_for(lib, p, chain=ch) for p in targets]
    return "; ".join(f"references of {r.paper_id}: {r.status.replace('_', ' ')}"
                     + (f" ({r.resolved} of {r.entries} matched)" if r.status == "parsed" else "") for r in results)


def approve_job(lib, runner, chain, slug: str, choices: dict[str, str]) -> JobResult:
    """:func:`seedgraph.library.growth.approve`, then convert the fetched open-access PDFs;
    each such paper's references are parsed once it converts."""
    with provider_chain(lib, chain) as ch:
        results = growth.approve(lib, slug, choices, chain=ch)  # type: ignore[arg-type]
    fetched = frozenset(a.paper_id for a in results if a.oa == "held")
    if fetched:
        runner.resume_queue(then=partial(parse_references_after, chain=chain, only=fetched))
    lines = [f"{a.paper_id}: {a.detail}" + (f" ({a.doi_url})" if a.needs_pdf and a.doi_url else "")
             for a in results]
    return JobResult("\n".join(lines) or "nothing chosen", paper_ids=tuple(a.paper_id for a in results))


def _int(value, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def grow_page(request):
    web = get_web(request)
    q = request.query_params
    view = graphs_view.grow_tab(web.library(), request.path_params["slug"], seeds=q.getlist("seed") or None,
                                depth=_int(q.get("depth"), 1), cap=_int(q.get("cap"), 50),
                                estimate=bool(q.get("estimate")), jobs=web.jobs.jobs())
    if view is None:
        return not_found("no such graph")
    return _tab(request, "grow.html", f"{view.detail.display_name}: grow", {"view": view}, "grow")


def grow_jobs(request):
    web = get_web(request)
    slug = request.path_params["slug"]
    if not graphs_view.exists(web.library(), slug):
        return not_found("no such graph")
    return web.render(request, "graphs/_growth_jobs.html",
                      {"slug": slug, "jobs": graphs_view.growth_jobs(web.jobs.jobs(), slug)})


def grow_walk(request):
    web = get_web(request)
    slug = request.path_params["slug"]
    if not graphs_view.exists(web.library(), slug):
        return not_found("no such graph")
    data = form(request)
    seeds = [str(s) for s in data.getlist("seed") if str(s).strip()]
    depth, cap = _int(data.get("depth"), 0), _int(data.get("cap"), 0)
    if not 1 <= depth <= growth.MAX_DEPTH or cap < 1:
        return _back(request, "grow", f"not started: depth must be 1 to {growth.MAX_DEPTH} and cap at least 1",
                     "error")
    web.jobs.submit(graphs_view.GROWTH_JOB, walk_job, web.chain, slug, seeds, depth, cap,
                    label=f"{slug}: growth walk")
    return _back(request, "grow", "growth walk started")


def grow_approve(request):
    web = get_web(request)
    slug = request.path_params["slug"]
    if not graphs_view.exists(web.library(), slug):
        return not_found("no such graph")
    data = form(request)
    choices = {k.removeprefix("choice:"): str(v) for k, v in data.multi_items()
               if k.startswith("choice:") and str(v) != "skip"}
    if not choices:
        return _back(request, "grow", "nothing chosen", "warning")
    errors = graphs_view.check_choices(web.library(), slug, choices)
    if errors:
        return _back(request, "grow", "not approved: " + "; ".join(errors), "error")
    web.jobs.submit(graphs_view.GROWTH_JOB, approve_job, web.jobs, web.chain, slug, choices,
                    label=f"{slug}: growth approval")
    return _back(request, "grow", f"approving {len(choices)} paper{'' if len(choices) == 1 else 's'}")


# --- Lenses -----------------------------------------------------------------------------------------

def lenses_page(request):
    web = get_web(request)
    view = graphs_view.lenses_tab(web.library(), request.path_params["slug"])
    if view is None:
        return not_found("no such graph")
    return _tab(request, "lenses.html", f"{view.detail.display_name}: lenses", {"view": view}, "lenses")


def _lens_post(request, chosen: bool):
    web = get_web(request)
    slug = request.path_params["slug"]
    view = graphs_view.lenses_tab(web.library(), slug)
    if view is None:
        return not_found("no such graph")
    lens_id = str(form(request).get("lens_id") or "").strip()
    if chosen and lens_id not in {lens.lens_id for lens in view.lenses if lens.in_library}:
        return _back(request, "lenses", f"not chosen: the library has no lens {lens_id!r}", "error")
    with graphs.open_graph(web.library(), slug) as g:
        if chosen:
            graphs.choose_lens(g, lens_id)
            return _back(request, "lenses", f"chose lens {lens_id}")
        changed = graphs.unchoose_lens(g, lens_id)
    return _back(request, "lenses", f"unchose lens {lens_id}" if changed else f"{lens_id} was not chosen",
                 "success" if changed else "warning")


def lens_choose(request):
    return _lens_post(request, True)


def lens_unchoose(request):
    return _lens_post(request, False)


ROUTES = [
    Route("/graphs", graphs_page, methods=["GET"]),
    Route("/graphs/{slug}/lenses", lenses_page, methods=["GET"]),
    Route("/graphs/{slug}/lenses/choose", lens_choose, methods=["POST"]),
    Route("/graphs/{slug}/lenses/unchoose", lens_unchoose, methods=["POST"]),
    Route("/graphs/{slug}/grow", grow_page, methods=["GET"]),
    Route("/graphs/{slug}/grow/jobs", grow_jobs, methods=["GET"]),
    Route("/graphs/{slug}/grow/walk", grow_walk, methods=["POST"]),
    Route("/graphs/{slug}/grow/approve", grow_approve, methods=["POST"]),
    Route("/graphs/{slug}", graph_page, methods=["GET"]),
    Route("/graphs/{slug}/graph", graph_tab, methods=["GET"]),
    Route("/graphs/{slug}/graph/svg", graph_partial, methods=["GET"]),
    Route("/graphs/{slug}/concepts", concepts_page, methods=["GET"]),
    Route("/graphs/{slug}/suggestions/confirm", suggestion_confirm, methods=["GET"]),
    Route("/graphs/{slug}/suggestions/add", suggestion_add, methods=["POST"]),
    Route("/graphs/{slug}/remove", remove, methods=["POST"]),
    Route("/graphs/{slug}/add", add, methods=["POST"]),
    Route("/graphs/{slug}/literatures/regroup", regroup, methods=["POST"]),
    Route("/graphs/{slug}/literatures/step", step, methods=["POST"]),
    Route("/graphs/{slug}/literatures/rename", rename_literature, methods=["POST"]),
]
