"""Small helpers shared by the U7 route modules (home, library, batches, review, graphs,
settings, records, spot-check). The leading underscore keeps this module out of route
discovery. No SQL here: routes call one view or one deep function."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import anyio.from_thread
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from ..app import Flash, Web, get_web


def form(request: Request, **kw: Any) -> FormData:
    """The form of a sync endpoint's request (Starlette runs those in its pool)."""
    return anyio.from_thread.run(lambda: request.form(**kw))


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def not_found(text: str = "not found") -> Response:
    return PlainTextResponse(text, status_code=404)


def now(web: Web) -> datetime | None:
    """The injected clock's time (tests), else ``None`` (the deep modules' own clock)."""
    return web.clock() if web.clock is not None else None


def local_next(value: Any, default: str) -> str:
    """A ``next`` redirect target, kept only when it is a local path."""
    text = str(value or "")
    return text if text.startswith("/") and not text.startswith("//") else default


def review_open(web: Web) -> int:
    """Open attention items, for the sidebar's Review badge (app.py does not set it)."""
    from ..views import review

    return review.open_count(web.library())


def render(request: Request, template: str, context: dict[str, Any] | None = None, *,
           nav: str | None = None, flash: Sequence[Flash] = (), status_code: int = 200) -> Response:
    web = get_web(request)
    ctx = dict(context or {})
    ctx.setdefault("review_open", review_open(web))
    return web.render(request, template, ctx, nav=nav, flash=flash, status_code=status_code)
