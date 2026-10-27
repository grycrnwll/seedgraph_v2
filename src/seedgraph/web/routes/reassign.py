""""This isn't the paper's PDF" (paper page Actions and Review -> PDF may not be this paper).
Thin adapters: the GETs read :mod:`seedgraph.web.views.reassign`; the POST confirms a previewed
plan with :func:`seedgraph.library.reassign.reassign_pdf` (409 when the papers changed since
the preview, or the plan is blocked).

    GET  /papers/{id}/reassign-pdf?file=&q=&next=                 choose the paper (or a new one)
    GET  /papers/{id}/reassign-pdf/preview?file=&to=|new=1&next=  what would move
    POST /papers/{id}/reassign-pdf   file, to | new, fingerprint, next (CSRF)
    POST /papers/{id}/reassign-pdf/merge   to (CSRF): the target holds another PDF and the two are
         one paper -> library.reassign.offer_merge -> the merge preview
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from ...library import reassign
from ..app import Flash, get_web
from ..views import reassign as view
from ._pages import form, not_found, render


def chooser(request: Request) -> Response:
    web = get_web(request)
    q = request.query_params
    pid = request.path_params["paper_id"]
    try:
        page = view.chooser(web.library(), pid, q.get("file") or None, query=q.get("q") or "",
                            next_=q.get("next") or "")
    except reassign.ReassignError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    if page is None:
        return not_found("no such paper")
    return render(request, "paper/reassign.html", {"page_title": "Whose PDF is this?", "c": page}, nav="library")


def _preview_page(request: Request, p, *, status_code: int = 200, stale: bool = False) -> Response:
    return render(request, "paper/reassign_preview.html", {"page_title": "Move the PDF", "p": p, "stale": stale},
                  nav="library", status_code=status_code)


def preview(request: Request) -> Response:
    web = get_web(request)
    q = request.query_params
    try:
        p = view.preview(web.library(), request.path_params["paper_id"], q.get("file") or None,
                         to=q.get("to") or None, new_paper=q.get("new") == "1", next_=q.get("next") or "")
    except reassign.ReassignError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    if p is None:
        return not_found("no such paper")
    return _preview_page(request, p)


def confirm(request: Request) -> Response:
    web = get_web(request)
    data = form(request)
    lib = web.library()
    pid = request.path_params["paper_id"]
    to = str(data.get("to") or "") or None
    new_paper = str(data.get("new") or "") == "1"
    try:
        p = view.preview(lib, pid, str(data.get("file") or "") or None, to=to, new_paper=new_paper,
                         next_=str(data.get("next") or ""))
    except reassign.ReassignError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    if p is None:
        return not_found("no such paper")
    if p.plan.blockers:
        return _preview_page(request, p, status_code=409)
    if p.plan.fingerprint != str(data.get("fingerprint") or ""):
        return _preview_page(request, p, status_code=409, stale=True)
    try:
        result = reassign.reassign_pdf(lib, p.plan, confirm=True)
    except reassign.StalePlanError:
        return _preview_page(request, p, status_code=409, stale=True)
    where = "a new paper" if result.new_paper else result.receiving_paper_id
    flash = [Flash(f"moved the PDF to {where}; {pid} now needs its own PDF"
                   if p.plan.source_becomes_cited_only else f"moved the PDF to {where}", "success")]
    if result.new_item_id:
        flash.append(Flash("its title is not on the PDF either: check it in Review", "warning"))
    return web.redirect(p.next or f"/papers/{result.receiving_paper_id}", flash=flash)


def merge_offer(request: Request) -> Response:
    """The blocked preview's "Merge these papers": raise the identity check, open the merge preview."""
    from urllib.parse import urlencode

    web = get_web(request)
    data = form(request)
    pid, to = request.path_params["paper_id"], str(data.get("to") or "")
    try:
        item_id = reassign.offer_merge(web.library(), pid, to)
    except reassign.ReassignError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    if item_id is None:
        return web.redirect(f"/papers/{pid}", flash=[Flash(
            f"{pid} and {to} were recorded as different papers; nothing to merge", "warning")])
    return web.redirect("/review/merge?" + urlencode({"item": item_id, "with": to}))


ROUTES = [
    Route("/papers/{paper_id}/reassign-pdf", chooser, methods=["GET"]),
    Route("/papers/{paper_id}/reassign-pdf/preview", preview, methods=["GET"]),
    Route("/papers/{paper_id}/reassign-pdf", confirm, methods=["POST"]),
    Route("/papers/{paper_id}/reassign-pdf/merge", merge_offer, methods=["POST"]),
]
