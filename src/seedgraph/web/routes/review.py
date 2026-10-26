"""Review screens (design §2.6). GETs read views; the one POST resolves an attention item
with a fixed resolution from its queue (:func:`seedgraph.library.attention.resolve_item`)."""

from __future__ import annotations

from urllib.parse import urlencode

from starlette.responses import PlainTextResponse
from starlette.routing import Route

from ...library import attention
from ..app import Flash, get_web
from ..views import review as review_view
from ._pages import form, is_htmx, local_next, not_found, render


def attention_page(request):
    web = get_web(request)
    q = request.query_params
    view = review_view.attention_page(web.library(), kind=q.get("kind") or None, graph=q.get("graph") or None)
    back = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    return render(request, "review/attention.html", {"page_title": "Review", "view": view, "back": back},
                  nav="review")


def resolve(request):
    web = get_web(request)
    item_id = request.path_params["item_id"]
    data = form(request)
    lib = web.library()
    row = review_view.item_row(lib, item_id)
    if row is None:
        return not_found("no such attention item")
    resolution = str(data.get("resolution") or "")
    if resolution not in review_view.allowed_resolutions(row.kind):
        return PlainTextResponse(f"{resolution!r} is not an action for {row.kind} items", status_code=400)
    attention.resolve_item(lib, item_id, resolution)
    if is_htmx(request):
        return web.render(request, "review/_item_row.html", {"row": review_view.item_row(lib, item_id),
                                                             "back": ""})
    default = "/review/attention?" + urlencode({"kind": row.kind})
    return web.redirect(local_next(data.get("next"), default),
                        flash=[Flash(f"resolved ({resolution.replace('_', ' ')})", "success")])


def duplicates_page(request):
    web = get_web(request)
    view = review_view.duplicates_page(web.library())
    return render(request, "review/duplicates.html", {"page_title": "Duplicates", "view": view,
                                                      "back": "/review/duplicates"}, nav="review")


def citations_page(request):
    web = get_web(request)
    rows = review_view.citations_page(web.library())
    return render(request, "review/citations.html", {"page_title": "Citation review", "rows": rows},
                  nav="review")


ROUTES = [
    Route("/review/attention", attention_page, methods=["GET"]),
    Route("/review/attention/{item_id}/resolve", resolve, methods=["POST"]),
    Route("/review/duplicates", duplicates_page, methods=["GET"]),
    Route("/review/citations", citations_page, methods=["GET"]),
]
