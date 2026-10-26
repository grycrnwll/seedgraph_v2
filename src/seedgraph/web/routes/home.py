"""Home (design §2.1): the per-graph checklist and the activity feed, which htmx polls
every 10 s while a batch is working."""

from __future__ import annotations

from starlette.routing import Route

from ..app import get_web
from ..views import home as home_view
from ._pages import now, render


def home_page(request):
    web = get_web(request)
    view = home_view.page(web.library(), jobs=web.jobs.jobs(), now=now(web))
    return render(request, "home/home.html", {"page_title": "Home", "view": view}, nav="home")


def activity_partial(request):
    web = get_web(request)
    feed = home_view.activity(web.library(), jobs=web.jobs.jobs(), now=now(web))
    return web.render(request, "home/_activity.html", {"activity": feed})


ROUTES = [
    Route("/", home_page, methods=["GET"], name="home"),
    Route("/home/activity", activity_partial, methods=["GET"]),
]
