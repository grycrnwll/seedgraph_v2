"""Small helpers shared by the paper-page route modules (papers, notes, files). The
leading underscore keeps this module out of route discovery."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anyio.from_thread
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from ..app import Flash, Web, get_web
from ..views import paper as paper_view


def form(request: Request, **kw: Any) -> FormData:
    """The request's form, read from a sync endpoint (Starlette runs those in its pool)."""
    return anyio.from_thread.run(lambda: request.form(**kw))


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def not_found(text: str = "no such paper") -> Response:
    return PlainTextResponse(text, status_code=404)


def paper_id_or_none(request: Request) -> str | None:
    """The path's paper id when it names a library paper (this thread's Library)."""
    pid = request.path_params.get("paper_id", "")
    web = get_web(request)
    return pid if paper_view.exists(web.library(), pid) else None


def int_field(data: FormData | dict, name: str, default: int | None = None) -> int | None:
    raw = data.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw))
    except ValueError:
        return default


def done(request: Request, web: Web, url: str, flash: Sequence[Flash] = ()) -> Response:
    """Redirect after a POST: 303 for a plain form, ``HX-Redirect`` for an htmx request."""
    resp = web.redirect(url, flash=flash)
    if is_htmx(request):
        return Response(status_code=200, headers={"HX-Redirect": resp.headers["location"]})
    return resp
