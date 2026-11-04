"""Records (stage 5 design §6.1-6.3; slice R5): the Records list, the record page, its
markdown export, and a graph's Records tab (``/graphs/{slug}/records``; the tab itself is
listed in :data:`seedgraph.web.views.graphs.TABS`).

GETs read :mod:`seedgraph.web.views.records` (or :func:`seedgraph.records.export.export_markdown`
for the download) and never write. Each POST calls one deep function of
:mod:`seedgraph.records.manage`: ``star``, ``annotate`` or ``delete`` (after a confirm
page). A refused action redirects back with the reason as an error flash; an unknown
record is a 404.
"""

from __future__ import annotations

from datetime import datetime

from starlette.responses import Response
from starlette.routing import Route

from ...records import export, manage
from ..app import Flash, get_web
from ..views import graphs as graphs_view
from ..views import records as records_view
from ._pages import form, local_next, not_found, now, render

NAV = "records"


def _record_id(request) -> str:
    return request.path_params["record_id"]


def records_page(request):
    web = get_web(request)
    view = records_view.list_page(web.library(), request.query_params, now=now(web))
    return render(request, "records/list.html", {"page_title": "Records", "view": view}, nav=NAV)


def graph_records_page(request):
    web = get_web(request)
    lib = web.library()
    slug = request.path_params["slug"]
    d = graphs_view.detail(lib, slug)
    if d is None:
        return not_found("no such graph")
    view = records_view.list_page(lib, request.query_params, graph=slug, now=now(web))
    return render(request, "records/graph_tab.html",
                  {"page_title": f"{d.display_name}: records", "view": view, "d": d, "tabs": graphs_view.TABS,
                   "tab": "records"}, nav="graphs")


def record_page(request):
    web = get_web(request)
    view = records_view.record_page(web.library(), _record_id(request), now=now(web))
    if view is None:
        return not_found("no such record")
    return render(request, "records/record.html", {"page_title": view.heading, "view": view}, nav=NAV)


def delete_confirm(request):
    web = get_web(request)
    view = records_view.record_page(web.library(), _record_id(request), now=now(web))
    if view is None:
        return not_found("no such record")
    return render(request, "records/delete.html", {"page_title": "Delete this record?", "view": view}, nav=NAV)


def export_page(request):
    web = get_web(request)
    rid = _record_id(request)
    try:
        text = export.export_markdown(web.library(), rid, now=now(web), tz=datetime.now().astimezone().tzinfo)
    except manage.RecordNotFound:
        return not_found("no such record")
    return Response(text, media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{rid}.md"'})


def _back(request, data, text: str, level: str = "success"):
    default = f"/records/{_record_id(request)}"
    return get_web(request).redirect(local_next(data.get("next"), default), flash=[Flash(text, level)])


def star(request):
    web = get_web(request)
    data = form(request)
    starred = str(data.get("starred") or "1") == "1"
    try:
        manage.star(web.library(), _record_id(request), starred)
    except manage.RecordNotFound:
        return not_found("no such record")
    return _back(request, data, "starred" if starred else "star removed")


def annotate(request):
    web = get_web(request)
    data = form(request)
    try:
        manage.annotate(web.library(), _record_id(request), str(data.get("text") or ""), now=now(web))
    except manage.RecordNotFound:
        return not_found("no such record")
    except manage.RecordError as exc:
        return _back(request, data, f"annotation not saved: {exc}", "error")
    return _back(request, data, "annotation saved")


def delete(request):
    web = get_web(request)
    form(request)
    try:
        manage.delete(web.library(), _record_id(request))
    except manage.RecordNotFound:
        return not_found("no such record")
    return web.redirect("/records", flash=[Flash("record deleted", "success")])


ROUTES = [
    Route("/records", records_page, methods=["GET"]),
    Route("/records/{record_id}", record_page, methods=["GET"]),
    Route("/records/{record_id}/export", export_page, methods=["GET"]),
    Route("/records/{record_id}/delete", delete_confirm, methods=["GET"]),
    Route("/records/{record_id}/delete", delete, methods=["POST"]),
    Route("/records/{record_id}/star", star, methods=["POST"]),
    Route("/records/{record_id}/annotate", annotate, methods=["POST"]),
    Route("/graphs/{slug}/records", graph_records_page, methods=["GET"]),
]
