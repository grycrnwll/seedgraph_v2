"""Spot-check (stage 5 design §5, §6.5; slice R6): the accuracy page with its start form and
earlier samples, and one sample's items with their marks.

    GET  /spot-check                              accuracy, start form, earlier samples
    POST /spot-check/start                        spotcheck.sample.create_sample -> the sample page
    GET  /spot-check/{sample_id}                  the sample's items
    POST /spot-check/{sample_id}/mark/{claim_id}  spotcheck.marks.mark (htmx: the refreshed item)

GETs read :mod:`seedgraph.web.views.spotcheck` and never write. Each POST calls one deep
function; a refused action shows the reason (an error flash, or a message on the item for
htmx). An unknown sample or claim is a 404. "Fix it now" is a link to the paper page
(:func:`seedgraph.web.views.spotcheck.item_view`); it needs no route here.
"""

from __future__ import annotations

from starlette.routing import Route

from ...spotcheck import SpotCheckError, marks
from ...spotcheck import sample as sample_mod
from ...work.store import fmt_ts
from ..app import Flash, get_web
from ..views import spotcheck as view
from ._pages import form, is_htmx, not_found, now, render

NAV = "spotcheck"
CHOICES = {"mark_choices": view.MARK_CHOICES, "reason_choices": view.REASON_CHOICES}


def _stamp(web) -> str | None:
    at = now(web)
    return fmt_ts(at) if at is not None else None


def index_page(request):
    web = get_web(request)
    return render(request, "spotcheck/index.html",
                  {"page_title": "Spot-check", "view": view.index(web.library())}, nav=NAV)


def sample_page(request):
    web = get_web(request)
    page = view.sample_page(web.library(), request.path_params["sample_id"])
    if page is None:
        return not_found("no such spot-check sample")
    return render(request, "spotcheck/sample.html", {"page_title": "Spot-check sample", "view": page, **CHOICES}, nav=NAV)


def _size(value) -> int | None:
    try:
        n = int(str(value).strip())
    except ValueError:
        return None
    return n if 1 <= n <= view.MAX_SIZE else None


def start(request):
    web = get_web(request)
    data = form(request)
    raw = data.get("size")
    size = view.DEFAULT_SIZE if raw is None or str(raw).strip() == "" else _size(raw)
    if size is None:
        return web.redirect("/spot-check", flash=[Flash(f"the size must be a whole number from 1 to "
                                                        f"{view.MAX_SIZE}", "error")])
    filters = {k: str(data.get(k) or "") for k in ("graph", "claim_type", "model")}
    try:
        made = sample_mod.create_sample(web.library(), n=size, filters=filters, now=_stamp(web))
    except SpotCheckError as exc:
        return web.redirect("/spot-check", flash=[Flash(f"no sample started: {exc}", "error")])
    return web.redirect(f"/spot-check/{made.sample_id}",
                        flash=[Flash(f"started a sample of {len(made.items)} claims", "success")])


def mark(request):
    web = get_web(request)
    data = form(request)
    sid, cid = request.path_params["sample_id"], request.path_params["claim_id"]
    lib = web.library()
    message, level = "mark saved", "success"
    try:
        marks.mark(lib, sid, cid, str(data.get("value") or ""), reasons=[str(r) for r in data.getlist("reasons")],
                   note=str(data.get("note") or "") or None, now=_stamp(web))
    except SpotCheckError as exc:
        message, level = f"not saved: {exc}", "error"
    swap = view.item_swap(lib, sid, cid, message=message, message_level=level)
    if swap is None:
        return not_found("no such claim in that spot-check sample")
    if is_htmx(request):
        return render(request, "spotcheck/_swap.html", {"swap": swap, **CHOICES})
    return web.redirect(f"/spot-check/{sid}#item-{cid}", flash=[Flash(message, level)])


ROUTES = [
    Route("/spot-check", index_page, methods=["GET"]),
    Route("/spot-check/start", start, methods=["POST"]),
    Route("/spot-check/{sample_id}", sample_page, methods=["GET"]),
    Route("/spot-check/{sample_id}/mark/{claim_id}", mark, methods=["POST"]),
]
