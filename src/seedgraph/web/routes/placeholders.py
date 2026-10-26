"""Records and Spot-check: static placeholders until stage 5 (design §2)."""

from __future__ import annotations

from starlette.routing import Route

from ._pages import render


def records(request):
    return render(request, "placeholders/placeholder.html",
                  {"page_title": "Records", "what": "Records"}, nav="records")


def spot_check(request):
    return render(request, "placeholders/placeholder.html",
                  {"page_title": "Spot-check", "what": "Spot-check"}, nav="spotcheck")


ROUTES = [
    Route("/records", records, methods=["GET"]),
    Route("/spot-check", spot_check, methods=["GET"]),
]
