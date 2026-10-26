"""Context graphs (design §2.3): list, per-graph page, add to and remove from a graph."""

from __future__ import annotations

from starlette.routing import Route

from ...errors import SeedgraphError
from ...library import graphs, lifecycle
from ..app import Flash, get_web
from ..views import graphs as graphs_view
from ._pages import form, not_found, render

STATUSES = ("member", "frontier", "excluded")


def graphs_page(request):
    web = get_web(request)
    rows = graphs_view.list_page(web.library())
    return render(request, "graphs/list.html", {"page_title": "Context graphs", "rows": rows}, nav="graphs")


def graph_page(request):
    web = get_web(request)
    view = graphs_view.detail(web.library(), request.path_params["slug"])
    if view is None:
        return not_found("no such graph")
    return render(request, "graphs/detail.html", {"page_title": view.display_name, "view": view,
                                                  "statuses": STATUSES}, nav="graphs")


def remove(request):
    web = get_web(request)
    slug = request.path_params["slug"]
    if graphs_view.detail(web.library(), slug) is None:
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
    if graphs_view.detail(web.library(), slug) is None:
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


ROUTES = [
    Route("/graphs", graphs_page, methods=["GET"]),
    Route("/graphs/{slug}", graph_page, methods=["GET"]),
    Route("/graphs/{slug}/remove", remove, methods=["POST"]),
    Route("/graphs/{slug}/add", add, methods=["POST"]),
]
