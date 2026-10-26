"""Settings page (design §2.7). GET reads; the one POST re-checks Ollama and keeps the
result in memory for the page (it writes nothing)."""

from __future__ import annotations

from starlette.routing import Route

from ...llm import ollama
from ..app import Flash, get_web
from ..views import settings as settings_view
from ._pages import render

_STATE_KEY = "sg_ollama_check"


def _existing_pool(web):
    # Never resolve the pool on a GET (that would probe GPUs): only one already built.
    return web.pool if web.pool is not None else web.jobs.pool_if_started()


def settings_page(request):
    web = get_web(request)
    view = settings_view.page(web.library(), web.home, pool=_existing_pool(web),
                              ollama_check=getattr(request.app.state, _STATE_KEY, None))
    return render(request, "settings/settings.html", {"page_title": "Settings", "view": view}, nav="settings")


def check_ollama(request):
    web = get_web(request)
    result = ollama.check_reachable()
    setattr(request.app.state, _STATE_KEY, result)
    text = "Ollama is reachable" if result.ok else "Ollama is not reachable"
    return web.redirect("/settings", flash=[Flash(text, "success" if result.ok else "warning")])


ROUTES = [
    Route("/settings", settings_page, methods=["GET"]),
    Route("/settings/ollama/check", check_ollama, methods=["POST"]),
]
