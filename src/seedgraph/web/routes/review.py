"""Review screens (design §2.6). GETs read views. POSTs: resolve an attention item with a
fixed resolution from its queue (:func:`seedgraph.library.attention.resolve_item`; an
identity item's "different" also records the verdict), and confirm a previewed paper merge
(:func:`seedgraph.library.merge.merge_papers`, refused with 409 when the papers changed
since the preview)."""

from __future__ import annotations

from urllib.parse import urlencode

from starlette.responses import PlainTextResponse
from starlette.routing import Route

from ...library import attention, merge
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
    if row.queue.key == "identity" and resolution == "different":
        review_view.record_different(lib, row)
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


def _merge_page(request, preview, *, status_code: int = 200, stale: bool = False):
    return render(request, "review/merge.html", {"page_title": "Merge papers", "p": preview, "stale": stale},
                  nav="review", status_code=status_code)


def merge_preview(request):
    web = get_web(request)
    q = request.query_params
    try:
        preview = review_view.merge_preview(web.library(), q.get("item") or "", q.get("with") or "",
                                            keep=q.get("keep") or None, primary=q.get("primary") or None)
    except merge.MergeError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    if preview is None:
        return not_found("no such identity item and candidate")
    return _merge_page(request, preview)


def merge_confirm(request):
    web = get_web(request)
    data = form(request)
    lib = web.library()
    item_id, with_id, keep, other = (str(data.get(k) or "") for k in ("item", "with", "keep", "other"))
    try:
        preview = review_view.merge_preview(lib, item_id, with_id, keep=keep,
                                            primary=str(data.get("primary") or "") or None)
    except merge.MergeError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    if preview is None:
        return not_found("no such identity item and candidate")
    plan = preview.plan
    if (plan.keep_id, plan.other_id) != (keep, other) or plan.fingerprint != str(data.get("fingerprint") or ""):
        return _merge_page(request, preview, status_code=409, stale=True)
    if plan.blockers:
        return _merge_page(request, preview, status_code=409)
    try:
        merge.merge_papers(lib, plan, confirm=True)
    except merge.StalePlanError:
        return _merge_page(request, preview, status_code=409, stale=True)
    item = attention.get_item(lib, item_id)
    if item is not None and item.status == "open":
        attention.resolve_item(lib, item_id, "same")
    return web.redirect("/review/duplicates", flash=[Flash(f"merged {other} into {keep}", "success")])


ROUTES = [
    Route("/review/attention", attention_page, methods=["GET"]),
    Route("/review/attention/{item_id}/resolve", resolve, methods=["POST"]),
    Route("/review/duplicates", duplicates_page, methods=["GET"]),
    Route("/review/merge", merge_preview, methods=["GET"]),
    Route("/review/merge", merge_confirm, methods=["POST"]),
    Route("/review/citations", citations_page, methods=["GET"]),
]
