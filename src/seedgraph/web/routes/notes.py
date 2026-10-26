"""Note editing on the paper page (docs/V3_UI_DESIGN.md §4, §5). Thin adapters over
:mod:`notes.overlay`, :mod:`notes.replay` and :mod:`library.pages`.

Every edit form posts one op with the ``expected_version`` it was rendered from. An htmx
request gets the whole notes pane back (so every form on it carries the new version); a
plain form gets a redirect to the paper page. A stale version or an invalid op is refused
with a message and writes nothing.

    GET  /papers/{id}/notes                 the notes pane partial (?show_hidden=1)
    POST /papers/{id}/notes/edit            {claim_id, claim_text, normalized_label, claim_type}
    POST /papers/{id}/notes/hide|unhide     {claim_id}
    POST /papers/{id}/notes/remove          {claim_id}  (an added claim)
    POST /papers/{id}/notes/add             {kind=assertion|quote, claim_type, normalized_label,
                                             claim_text, [markdown_id, start_char, end_char]}
    POST /papers/{id}/notes/researcher-notes {text}
    POST /papers/{id}/notes/revert          {to_version}
    POST /papers/{id}/anchor                {page, text}: a PDF-pane selection -> the add form
    POST /papers/{id}/anchor-text           {start, end}: a text-view selection -> the add form
    GET  /papers/{id}/text?page=N           the text view partial
    GET  /papers/{id}/replay                the replay view
    POST /papers/{id}/replay/accept         {head_version_no, new_note_id, accept_new, dismiss,
                                             decision:<op key>=keep|discard|attach:<claim_id>}
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from ...notes import overlay, replay
from ..app import Flash, get_web
from ..views import paper as paper_view
from ._paperkit import done, form, int_field, is_htmx, not_found, paper_id_or_none

STALE_MESSAGE = overlay.STALE_MESSAGE


def _pane(request: Request, pid: str, *, show_hidden: bool, message: str | None = None,
          level: str = "info", status_code: int = 200) -> Response:
    web = get_web(request)
    pane = paper_view.notes(web.library(), pid, show_hidden=show_hidden, message=message, message_level=level)
    return web.render(request, "paper/_notes.html", {"pane": pane}, status_code=status_code)


def notes_partial(request: Request) -> Response:
    pid = paper_id_or_none(request)
    if pid is None:
        return not_found()
    return _pane(request, pid, show_hidden=request.query_params.get("show_hidden") == "1")


def _save(request: Request, build) -> Response:
    """Parse the form, build ``(kind, payload)`` and save it; answer with the pane."""
    web = get_web(request)
    pid = paper_id_or_none(request)
    if pid is None:
        return not_found()
    data = form(request)
    show_hidden = data.get("show_hidden") == "1"
    expected = int_field(data, "expected_version")
    lib = web.library()
    message, level = None, "success"
    if expected is None:
        message, level = "the form has no version; reload the page", "error"
    else:
        try:
            message = build(lib, pid, data, expected)
        except overlay.OverlayConflict:
            message, level = STALE_MESSAGE, "error"
        except overlay.OverlayError as exc:
            message, level = f"not saved: {exc}", "error"
    if is_htmx(request):
        return _pane(request, pid, show_hidden=show_hidden, message=message, level=level)
    target = f"/papers/{pid}" + ("?show_hidden=1" if show_hidden else "")
    return done(request, web, target, [Flash(message, level)] if message else [])


def _claim_id(data) -> str:
    cid = str(data.get("claim_id") or "")
    if not cid:
        raise overlay.OverlayError("no claim was named")
    return cid


def _edit(lib, pid, data, expected) -> str:
    cid = _claim_id(data)
    set_ = paper_view.edit_set(lib, pid, cid, data)
    if not set_:
        return "nothing changed"
    overlay.save_edit(lib, pid, {"op": "edit", "target": {"claim_id": cid}, "set": set_},
                      expected_version=expected)
    return "edit saved"


def _simple(kind: str, done_text: str):
    def build(lib, pid, data, expected) -> str:
        cid = _claim_id(data)
        op: dict[str, Any] = ({"op": "remove_added", "add_id": cid} if kind == "remove_added"
                              else {"op": kind, "target": {"claim_id": cid}})
        overlay.save_edit(lib, pid, op, expected_version=expected)
        return done_text
    return build


def _add(lib, pid, data, expected) -> str:
    kind = str(data.get("kind") or "assertion")
    op: dict[str, Any] = {"op": "add", "kind": kind, "claim_type": str(data.get("claim_type") or ""),
                          "normalized_label": str(data.get("normalized_label") or "").strip(),
                          "claim_text": str(data.get("claim_text") or "").strip()}
    if kind == "quote":
        op["quote"] = {"markdown_id": str(data.get("markdown_id") or ""),
                       "start_char": int_field(data, "start_char"), "end_char": int_field(data, "end_char")}
    overlay.save_edit(lib, pid, op, expected_version=expected)
    return "quote-backed claim added" if kind == "quote" else "assertion added"


def _notes(lib, pid, data, expected) -> str:
    overlay.save_notes(lib, pid, str(data.get("text") or ""), expected_version=expected)
    return "researcher notes saved"


def _revert(lib, pid, data, expected) -> str:
    to = int_field(data, "to_version")
    if to is None:
        raise overlay.OverlayError("no version to revert to")
    overlay.revert(lib, pid, to, expected_version=expected)
    return f"reverted to version {to}"


def edit(request: Request) -> Response:
    return _save(request, _edit)


def hide(request: Request) -> Response:
    return _save(request, _simple("hide", "claim hidden"))


def unhide(request: Request) -> Response:
    return _save(request, _simple("unhide", "claim shown again"))


def remove(request: Request) -> Response:
    return _save(request, _simple("remove_added", "added claim removed"))


def add(request: Request) -> Response:
    return _save(request, _add)


def researcher_notes(request: Request) -> Response:
    return _save(request, _notes)


def revert(request: Request) -> Response:
    return _save(request, _revert)


# --- quote-backed claims: selection -> anchor -> add form --------------------------------


def _anchor_response(request: Request, pid: str, result) -> Response:
    web = get_web(request)
    lib = web.library()
    return web.render(request, "paper/_anchor.html", {
        "paper_id": pid, "result": result, "expected_version": overlay.head_version_no(lib.conn, pid),
        "claim_types": paper_view.claim_type_choices()})


def anchor_pdf(request: Request) -> Response:
    from ...library import pages

    web = get_web(request)
    pid = paper_id_or_none(request)
    if pid is None:
        return not_found()
    data = form(request)
    page = int_field(data, "page")
    text = str(data.get("text") or "")
    if page is None or page < 1:
        result = pages.AnchorResult("not_found", message="the selection has no page; use the text view")
    else:
        result = pages.anchor_selection(web.library(), pid, page, text)
    return _anchor_response(request, pid, result)


def anchor_text(request: Request) -> Response:
    from ...library import pages

    web = get_web(request)
    pid = paper_id_or_none(request)
    if pid is None:
        return not_found()
    data = form(request)
    start, end = int_field(data, "start"), int_field(data, "end")
    if start is None or end is None:
        result = pages.AnchorResult("empty", message="the selection is empty")
    else:
        result = paper_view.text_selection(web.library(), pid, start, end)
    return _anchor_response(request, pid, result)


def text_view(request: Request) -> Response:
    web = get_web(request)
    pid = paper_id_or_none(request)
    if pid is None:
        return not_found()
    page = int_field(request.query_params, "page")
    view = paper_view.text_view(web.library(), pid, page)
    return web.render(request, "paper/_text_view.html", {"tv": view})


# --- replay ------------------------------------------------------------------------------


def replay_page(request: Request) -> Response:
    web = get_web(request)
    pid = paper_id_or_none(request)
    if pid is None:
        return not_found()
    lib = web.library()
    view = paper_view.replay(lib, pid)
    header = paper_view.header(lib, pid)
    return web.render(request, "paper/replay.html", {"rv": view, "h": header,
                                                     "page_title": "Compare: newer extraction"},
                      nav="library", status_code=200 if view is not None else 404)


def _decisions(data, plan: replay.ReplayPlan) -> tuple[set[str], dict[str, str]]:
    discard: set[str] = set()
    attach: dict[str, str] = {}
    for op in plan.ops:
        choice = str(data.get(f"decision:{op.key}") or "keep")
        if choice == "discard":
            discard.add(op.key)
        elif choice.startswith("attach:") and op.old_claim_id:
            attach[op.old_claim_id] = choice.split(":", 1)[1]
    return discard, attach


def replay_accept(request: Request) -> Response:
    web = get_web(request)
    pid = paper_id_or_none(request)
    if pid is None:
        return not_found()
    data = form(request)
    lib = web.library()
    back = f"/papers/{pid}/replay"
    try:
        plan = paper_view.replay_plan(lib, pid)
    except replay.ReplayError as exc:
        return done(request, web, f"/papers/{pid}", [Flash(f"nothing to accept: {exc}", "error")])
    if (int_field(data, "head_version_no") != plan.head_version_no
            or str(data.get("new_note_id") or "") != plan.new_note_id):
        return done(request, web, back, [Flash(f"{STALE_MESSAGE} (compare again)", "error")])
    mode = str(data.get("mode") or "selected")
    new_ids = [c.claim_id for c in plan.new_claims]
    if mode == "all":
        accept_new: Any = "all"
    elif mode == "dismiss_all":
        accept_new = []
    else:
        chosen = set(data.getlist("accept_new"))
        accept_new = [c for c in new_ids if c in chosen]
    discard, attach = _decisions(data, plan)
    try:
        v = replay.accept(lib, pid, plan, accept_new=accept_new, discard=discard, attach=attach)
    except replay.ReplayStale:
        return done(request, web, back, [Flash(f"{STALE_MESSAGE} (compare again)", "error")])
    except overlay.OverlayError as exc:
        return done(request, web, back, [Flash(f"not accepted: {exc}", "error")])
    return done(request, web, f"/papers/{pid}", [Flash(f"newer extraction accepted (version {v.version_no})",
                                                       "success")])


ROUTES = [
    Route("/papers/{paper_id}/notes", notes_partial, methods=["GET"]),
    Route("/papers/{paper_id}/notes/edit", edit, methods=["POST"]),
    Route("/papers/{paper_id}/notes/hide", hide, methods=["POST"]),
    Route("/papers/{paper_id}/notes/unhide", unhide, methods=["POST"]),
    Route("/papers/{paper_id}/notes/remove", remove, methods=["POST"]),
    Route("/papers/{paper_id}/notes/add", add, methods=["POST"]),
    Route("/papers/{paper_id}/notes/researcher-notes", researcher_notes, methods=["POST"]),
    Route("/papers/{paper_id}/notes/revert", revert, methods=["POST"]),
    Route("/papers/{paper_id}/anchor", anchor_pdf, methods=["POST"]),
    Route("/papers/{paper_id}/anchor-text", anchor_text, methods=["POST"]),
    Route("/papers/{paper_id}/text", text_view, methods=["GET"]),
    Route("/papers/{paper_id}/replay", replay_page, methods=["GET"]),
    Route("/papers/{paper_id}/replay/accept", replay_accept, methods=["POST"]),
]
