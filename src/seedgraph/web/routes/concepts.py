"""Concept pages (stage 4 design §7, §2.4, §11.2). GETs read views
(:mod:`seedgraph.web.views.concepts`) and never write. Each POST calls one deep function:
:func:`seedgraph.concepts.refresh.refresh`, a writer of
:mod:`seedgraph.concepts.researcher`, or :func:`seedgraph.work.routine.create_batch`.
A refused researcher action redirects back with the reason as an error flash.

Fixed paths are listed before ``/concepts/{family}/{label}``; a path whose first part is
not a concept family is a 404 there.
"""

from __future__ import annotations

from urllib.parse import quote, urlencode

from starlette.routing import Route

from ...concepts import refresh as rf
from ...concepts import researcher as rs
from ...concepts.families import CONCEPT_FAMILIES
from ...concepts.model import DEFAULT_SENSE, SenseKey, label_key, parse_handle
from ...errors import SeedgraphError
from ...work import routine
from ..app import Flash, get_web
from ..views import concepts as cv
from ._pages import form, local_next, not_found, render

NAV = "concepts"


def _page(request, template: str, title: str, ctx: dict, *, status_code: int = 200):
    return render(request, f"concepts/{template}", {"page_title": title, "here": quote(request.url.path), **ctx},
                  nav=NAV, status_code=status_code)


def _done(request, data, default: str, text: str, level: str = "success"):
    return get_web(request).redirect(local_next(data.get("next"), default), flash=[Flash(text, level)])


def _refused(request, data, default: str, exc: Exception):
    return _done(request, data, default, f"not done: {exc}", "error")


def _other(data, family: str) -> SenseKey:
    text = str(data.get("other") or "").strip()
    if not text:
        raise rs.ResearcherError("name the other concept as family/label (or label~s2 for a sense)")
    if "/" not in text:
        text = f"{family}/{text}"
    fam, key, sense = parse_handle(text)
    return SenseKey(fam, key, sense or DEFAULT_SENSE)


# --- search, refresh ----------------------------------------------------------------------------

def search_page(request):
    web = get_web(request)
    q = request.query_params
    view = cv.search_page(web.library(), q.get("q") or "", family=q.get("family") or None,
                          graph=q.get("graph") or None)
    return _page(request, "search.html", "Concepts", {"view": view})


def refresh(request):
    web = get_web(request)
    data = form(request)
    report = rf.refresh(web.library())
    return _done(request, data, "/concepts", f"concepts refreshed ({report.supported_concepts} supported concepts)")


# --- concept page and its actions ---------------------------------------------------------------

def concept_page(request):
    web = get_web(request)
    family = request.path_params["family"]
    if family not in CONCEPT_FAMILIES:
        return not_found("no such concept family")
    q = request.query_params
    view = cv.concept_page(web.library(), family, request.path_params["label"], q.get("sense") or None,
                           offset=cv.as_int(q.get("offset")) or 0)
    if view is None:
        return not_found("no such concept, or it has no live support")
    if isinstance(view, cv.Chooser):
        return _page(request, "chooser.html", f"{view.label_key}: choose a sense", {"view": view})
    return _page(request, "concept.html", view.name, {"view": view})


def _concept_post(request, action):
    family = request.path_params["family"]
    if family not in CONCEPT_FAMILIES:
        return not_found("no such concept family")
    data = form(request)
    key = SenseKey(family, label_key(request.path_params["label"]), str(data.get("sense") or DEFAULT_SENSE))
    back = cv.key_url(key)
    try:
        text = action(get_web(request).library(), key, data)
    except (rs.ResearcherError, ValueError) as exc:
        return _refused(request, data, back, exc)
    return _done(request, data, back, text)


def merge(request):
    return _concept_post(request, lambda lib, key, data: rs.merge(
        lib, key, _other(data, key.family), name=str(data.get("name") or "") or None).message)


def link(request):
    return _concept_post(request, lambda lib, key, data: rs.link(
        lib, key, _other(data, key.family), note=str(data.get("note") or ""),
        claim_a=str(data.get("claim_a") or "") or None, claim_b=str(data.get("claim_b") or "") or None).message)


def never_connect(request):
    return _concept_post(request, lambda lib, key, data: rs.never_connect(
        lib, key, _other(data, key.family)).message)


def split_senses(request):
    def act(lib, key, data):
        senses = []
        for i in range(1, cv.MAX_SENSES + 1):
            gloss = str(data.get(f"gloss_{i}") or "").strip()
            claims = [k[len("claim_"):] for k, v in data.multi_items() if k.startswith("claim_") and str(v) == str(i)]
            if gloss or claims:
                senses.append((gloss, claims))
        return rs.split_label(lib, key.family, key.label_key, senses).message
    return _concept_post(request, act)


def undo_split(request):
    return _concept_post(request, lambda lib, key, data: rs.undo_label_split(lib, key.family, key.label_key).message)


# --- decisions: split / undo --------------------------------------------------------------------

def split_decision(request):
    data = form(request)
    try:
        out = rs.split_decision(get_web(request).library(), str(data.get("decision") or ""))
    except rs.ResearcherError as exc:
        return _refused(request, data, "/concepts/recent", exc)
    return _done(request, data, "/concepts/recent", out.message)


def undo_decision(request):
    data = form(request)
    try:
        out = rs.undo(get_web(request).library(), str(data.get("decision") or ""))
    except rs.ResearcherError as exc:
        return _refused(request, data, "/concepts/recent", exc)
    return _done(request, data, "/concepts/recent", out.message)


# --- recent merges, queue -----------------------------------------------------------------------

def recent_page(request):
    web = get_web(request)
    view = cv.recent_page(web.library(), batch=request.query_params.get("batch") or None)
    return _page(request, "recent.html", "Recent merges", {"view": view})


def queue_page(request):
    web = get_web(request)
    return _page(request, "queue.html", "Concept queue", {"view": cv.queue_page(web.library())})


def settle(request):
    data = form(request)
    try:
        out = rs.settle(get_web(request).library(), str(data.get("item") or ""), str(data.get("action") or ""),
                        note=str(data.get("note") or "") or None, name=str(data.get("name") or "") or None)
    except rs.ResearcherError as exc:
        return _refused(request, data, "/concepts/queue", exc)
    return _done(request, data, "/concepts/queue", out.message)


# --- review sample ------------------------------------------------------------------------------

def review_index(request):
    web = get_web(request)
    return _page(request, "review_index.html", "Concept review samples", {"view": cv.review_index(web.library())})


def create_sample(request):
    data = form(request)
    try:
        sample = rs.create_review_sample(get_web(request).library(), n=cv.as_int(data.get("size")) or rs.SAMPLE_SIZE,
                                         seed=int(str(data.get("seed") or "0").strip() or 0))
    except (rs.ResearcherError, ValueError) as exc:
        return _refused(request, data, "/concepts/review", exc)
    return get_web(request).redirect(f"/concepts/review/{sample.sample_id}",
                                     flash=[Flash(f"sample of {len(sample.decision_ids)} drawn", "success")])


def review_page(request):
    web = get_web(request)
    view = cv.review_page(web.library(), request.path_params["sample_id"])
    if view is None:
        return not_found("no such review sample")
    return _page(request, "review.html", "Concept review", {"view": view})


def mark(request):
    data = form(request)
    sample_id = request.path_params["sample_id"]
    back = f"/concepts/review/{sample_id}"
    did = str(data.get("decision") or "")
    try:
        value = rs.mark(get_web(request).library(), sample_id, did, str(data.get("mark") or ""),
                        note=str(data.get("note") or "") or None)
    except rs.ResearcherError as exc:
        return _refused(request, data, back, exc)
    return _done(request, data, f"{back}#item-{did}", f"marked {value.replace('_', ' ')}")


# --- settings lists -----------------------------------------------------------------------------

def lists_page(request):
    web = get_web(request)
    return _page(request, "lists.html", "Concept lists", {"view": cv.lists_page(web.library())})


def edit_ambiguous(request):
    data = form(request)
    try:
        out = rs.edit_ambiguous(get_web(request).library(), str(data.get("action") or ""),
                                str(data.get("pattern") or ""), match=str(data.get("match") or "label"),
                                graph=str(data.get("graph") or "") or None, note=str(data.get("note") or "") or None)
    except rs.ResearcherError as exc:
        return _refused(request, data, "/concepts/lists", exc)
    return _done(request, data, "/concepts/lists", out.message)


def add_never_merge(request):
    data = form(request)
    try:
        out = rs.add_never_merge(get_web(request).library(), str(data.get("label_a") or ""),
                                 str(data.get("label_b") or ""), family=str(data.get("family") or "") or None,
                                 note=str(data.get("note") or "") or None)
    except rs.ResearcherError as exc:
        return _refused(request, data, "/concepts/lists", exc)
    return _done(request, data, "/concepts/lists", out.message)


def remove_never_merge(request):
    data = form(request)
    try:
        out = rs.remove_never_merge(get_web(request).library(), str(data.get("decision") or ""))
    except rs.ResearcherError as exc:
        return _refused(request, data, "/concepts/lists", exc)
    return _done(request, data, "/concepts/lists", out.message)


# --- concept batches ----------------------------------------------------------------------------

def batches_page(request):
    web = get_web(request)
    q = request.query_params
    view = cv.batch_panel(web.library(), q.get("task") or "bridge", q.getlist("family"), cv.as_int(q.get("k")),
                          cv.as_int(q.get("budget_tokens")), q.get("graph") or None)
    return _page(request, "batches.html", "Concept batches", {"view": view})


def create_batch(request):
    web = get_web(request)
    data = form(request)
    task = str(data.get("task") or "")
    back = "/concepts/batches?" + urlencode({"task": task})
    if task not in cv.BATCH_TASKS:
        return web.redirect(back, flash=[Flash("choose bridge or merge", "error")])
    opts = cv.batch_options(task, [str(f) for f in data.getlist("family")], cv.as_int(data.get("k")),
                            cv.as_int(data.get("budget_tokens")))
    try:
        batch = routine.create_batch(web.library(), f"concepts:{task}", options=opts,
                                     graph=str(data.get("graph") or "") or None, client="web")
    except (SeedgraphError, ValueError) as exc:
        return web.redirect(back, flash=[Flash(f"no batch created: {exc}", "error")])
    return web.redirect(f"/batches/{batch.batch_id}",
                        flash=[Flash("concept batch created; run it with one of the commands below", "success")])


ROUTES = [
    Route("/concepts", search_page, methods=["GET"]),
    Route("/concepts/refresh", refresh, methods=["POST"]),
    Route("/concepts/recent", recent_page, methods=["GET"]),
    Route("/concepts/queue", queue_page, methods=["GET"]),
    Route("/concepts/queue/settle", settle, methods=["POST"]),
    Route("/concepts/review", review_index, methods=["GET"]),
    Route("/concepts/review", create_sample, methods=["POST"]),
    Route("/concepts/review/{sample_id}", review_page, methods=["GET"]),
    Route("/concepts/review/{sample_id}/mark", mark, methods=["POST"]),
    Route("/concepts/lists", lists_page, methods=["GET"]),
    Route("/concepts/lists/ambiguous", edit_ambiguous, methods=["POST"]),
    Route("/concepts/lists/never-merge", add_never_merge, methods=["POST"]),
    Route("/concepts/lists/never-merge/remove", remove_never_merge, methods=["POST"]),
    Route("/concepts/batches", batches_page, methods=["GET"]),
    Route("/concepts/batches", create_batch, methods=["POST"]),
    Route("/concepts/decisions/split", split_decision, methods=["POST"]),
    Route("/concepts/decisions/undo", undo_decision, methods=["POST"]),
    Route("/concepts/{family}/{label}", concept_page, methods=["GET"]),
    Route("/concepts/{family}/{label}/merge", merge, methods=["POST"]),
    Route("/concepts/{family}/{label}/link", link, methods=["POST"]),
    Route("/concepts/{family}/{label}/never-connect", never_connect, methods=["POST"]),
    Route("/concepts/{family}/{label}/split-senses", split_senses, methods=["POST"]),
    Route("/concepts/{family}/{label}/undo-split", undo_split, methods=["POST"]),
]
