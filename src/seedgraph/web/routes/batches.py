"""Batches (design §2.5): list, detail, the items partial htmx polls every 5 s until the
batch finishes, and Stop (finish / now) and Continue, each one deep call."""

from __future__ import annotations

from starlette.responses import PlainTextResponse
from starlette.routing import Route

from ...errors import SeedgraphError
from ...work import routine
from ..app import Flash, get_web
from ..views import batch as batch_view
from ._pages import form, not_found, now, render

STOP_MODES = {"finish": "Stopping: current papers will finish", "now": "Stopped now"}


def batches_page(request):
    web = get_web(request)
    rows = batch_view.list_page(web.library(), now=now(web))
    return render(request, "batch/list.html", {"page_title": "Batches", "rows": rows}, nav="library")


def batch_page(request):
    web = get_web(request)
    view = batch_view.detail(web.library(), request.path_params["batch_id"], now=now(web))
    if view is None:
        return not_found("no such batch")
    return render(request, "batch/detail.html", {"page_title": f"Batch {view.batch_id}", "view": view},
                  nav="library")


def batch_items(request):
    web = get_web(request)
    view = batch_view.detail(web.library(), request.path_params["batch_id"], now=now(web))
    if view is None:
        return not_found("no such batch")
    return web.render(request, "batch/_items.html", {"view": view})


def stop(request):
    web = get_web(request)
    batch_id = request.path_params["batch_id"]
    mode = str(form(request).get("mode") or "")
    lib = web.library()
    if not batch_view.exists(lib, batch_id):
        return not_found("no such batch")
    if mode not in STOP_MODES:
        return PlainTextResponse(f"unknown stop mode {mode!r}", status_code=400)
    try:
        routine.stop(lib, batch_id, mode=mode, now=now(web))
    except SeedgraphError as exc:
        return web.redirect(f"/batches/{batch_id}", flash=[Flash(f"not stopped: {exc}", "error")])
    return web.redirect(f"/batches/{batch_id}", flash=[Flash(STOP_MODES[mode], "success")])


def continue_(request):
    web = get_web(request)
    batch_id = request.path_params["batch_id"]
    form(request)
    lib = web.library()
    if not batch_view.exists(lib, batch_id):
        return not_found("no such batch")
    try:
        routine.continue_batch(lib, batch_id, now=now(web))
    except SeedgraphError as exc:
        return web.redirect(f"/batches/{batch_id}", flash=[Flash(f"not continued: {exc}", "error")])
    return web.redirect(f"/batches/{batch_id}", flash=[Flash("batch continued", "success")])


ROUTES = [
    Route("/batches", batches_page, methods=["GET"]),
    Route("/batches/{batch_id}", batch_page, methods=["GET"]),
    Route("/batches/{batch_id}/items", batch_items, methods=["GET"]),
    Route("/batches/{batch_id}/stop", stop, methods=["POST"]),
    Route("/batches/{batch_id}/continue", continue_, methods=["POST"]),
]
