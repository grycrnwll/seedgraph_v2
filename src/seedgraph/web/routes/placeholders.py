"""Spot-check: a static placeholder until slice R6 (stage 5 design §11). Records became a
real page in slice R5 (:mod:`.records`)."""

from __future__ import annotations

from starlette.routing import Route

from ._pages import render


def spot_check(request):
    return render(request, "placeholders/placeholder.html",
                  {"page_title": "Spot-check", "what": "Spot-check"}, nav="spotcheck")


ROUTES = [
    Route("/spot-check", spot_check, methods=["GET"]),
]
