"""The one wiring point of the web UI (docs/V3_UI_DESIGN.md §1.2): middleware, routes,
templates and jobs. Tests build it with fakes.

Route registration (for the page slices): every module in :mod:`seedgraph.web.routes`
that defines ``ROUTES`` (a list of Starlette routes) is mounted, modules in sorted name
order, so a new page module needs no edit here. Endpoints should be plain ``def``
(Starlette runs them in its thread pool) and reach the app's services through
:func:`get_web`::

    from starlette.routing import Route
    from ..app import get_web

    def library_page(request):
        web = get_web(request)
        view = views.library.page(web.library(), ...)   # this thread's own Library
        return web.render(request, "pages/library.html", {"view": view}, nav="library")

    ROUTES = [Route("/library", library_page)]

Rules for templates: put ``{{ csrf_input }}`` first in every ``<form method="post">``;
no inline ``<script>``, ``style=`` or ``hx-on`` (the CSP forbids them). Flash
messages: pass ``flash=[Flash(...)]`` to :meth:`Web.render`, or redirect after a POST
with :meth:`Web.redirect` (a one-shot server-side message keyed by ``?flash=``; there
are no cookies or sessions).

Rules chosen where the design is silent: ``create_app`` takes the ``port`` (the Host
check needs it) and extra ``routes`` (tests); ``/`` shows a shell page until a routes
module claims it.
"""

from __future__ import annotations

import importlib
import mimetypes
import pkgutil
import secrets
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl

import jinja2
from markupsafe import Markup, escape
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import BaseRoute, Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from ..library.store import Library, library_dir
from . import security
from .jobs import JobRunner, JobsSummary, ThreadLibraries

DEFAULT_PORT = 8765
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

# Windows may map .js to text/plain from the registry; with nosniff a module would not run.
for _type, _ext in (("text/javascript", ".js"), ("text/javascript", ".mjs"),
                    ("text/css", ".css"), ("application/wasm", ".wasm"),
                    ("application/octet-stream", ".bcmap"), ("application/octet-stream", ".pfb")):
    mimetypes.add_type(_type, _ext)


@dataclass(frozen=True)
class NavItem:
    key: str
    label: str
    href: str


NAV: tuple[NavItem, ...] = (
    NavItem("home", "Home", "/"),
    NavItem("library", "Library", "/library"),
    NavItem("graphs", "Context graphs", "/graphs"),
    NavItem("records", "Records", "/records"),
    NavItem("spotcheck", "Spot-check", "/spot-check"),
    NavItem("settings", "Settings", "/settings"),
)


@dataclass(frozen=True)
class Flash:
    text: str
    level: str = "info"  # info | success | warning | error


class FlashStore:
    """One-shot flash messages for redirect-after-POST, keyed by an unguessable id.

    In memory and bounded; a key is consumed by its first read.
    """

    def __init__(self, limit: int = 256) -> None:
        self._items: OrderedDict[str, tuple[Flash, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self._limit = limit

    def put(self, messages: Sequence[Flash]) -> str:
        key = secrets.token_urlsafe(12)
        with self._lock:
            self._items[key] = tuple(messages)
            while len(self._items) > self._limit:
                self._items.popitem(last=False)
        return key

    def pop(self, key: str | None) -> tuple[Flash, ...]:
        if not key:
            return ()
        with self._lock:
            return self._items.pop(key, ())


@dataclass
class Web:
    """The app's services, reachable from any endpoint via :func:`get_web`."""

    home: Path
    port: int
    csrf_token: str
    templates: Jinja2Templates
    jobs: JobRunner
    libraries: ThreadLibraries
    flashes: FlashStore = field(default_factory=FlashStore)
    pool: Any = None
    chain: Any = None
    clock: Callable[[], datetime] | None = None

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def library(self) -> Library:
        """This thread's own read/write ``Library`` (opened on first use)."""
        return self.libraries.get()

    def render(self, request: Request, template: str, context: dict[str, Any] | None = None, *,
               status_code: int = 200, nav: str | None = None, flash: Sequence[Flash] = (),
               headers: dict[str, str] | None = None) -> Response:
        """Render ``template`` with the shell context (CSRF, sidebar, flashes, jobs).

        ``review_open`` (the sidebar's Review badge) is filled in here for every page; a
        page that passes its own value keeps it. (Starlette context processors override
        the page's context, so this one shell value is set here, not in ``_shell_context``.)
        """
        ctx = dict(context or {})
        ctx.setdefault("nav_active", nav)
        if "review_open" not in ctx:
            ctx["review_open"] = self.review_open()
        ctx["flashes"] = tuple(flash) + self.flashes.pop(request.query_params.get("flash"))
        return self.templates.TemplateResponse(request, template, ctx, status_code=status_code,
                                               headers=headers)

    def review_open(self) -> int:
        """Open attention items (the sidebar's Review badge); 0 if the library can't be read,
        so a broken library never breaks the page (or the error page) that shows the badge."""
        from .views.review import open_count  # local: views import this module

        try:
            return open_count(self.library())
        except Exception:  # noqa: BLE001 - a badge is never worth a failed render
            return 0

    def redirect(self, url: str, *, flash: Sequence[Flash] = ()) -> RedirectResponse:
        """303 to ``url`` (a local path), carrying one-shot flash messages."""
        if not url.startswith("/") or url.startswith("//"):
            raise ValueError(f"redirect target must be a local path: {url!r}")
        if flash:
            parts = urlsplit(url)
            query = parse_qsl(parts.query) + [("flash", self.flashes.put(flash))]
            url = urlunsplit(("", "", parts.path, urlencode(query), parts.fragment))
        return RedirectResponse(url, status_code=303)


def get_web(request: Request) -> Web:
    return request.app.state.web


def _shell_context(request: Request) -> dict[str, Any]:
    web = get_web(request)
    summary: JobsSummary = web.jobs.summary()
    return {
        "csrf_token": web.csrf_token,
        "csrf_input": Markup('<input type="hidden" name="csrf_token" value="%s">'
                             % escape(web.csrf_token)),
        "nav": NAV,
        "jobs_summary": summary,
    }


def discovered_routes() -> list[BaseRoute]:
    """``ROUTES`` of every module in :mod:`seedgraph.web.routes`, in module-name order."""
    try:
        package = importlib.import_module("seedgraph.web.routes")
    except ModuleNotFoundError:
        return []
    found: list[BaseRoute] = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"seedgraph.web.routes.{info.name}")
        found.extend(getattr(module, "ROUTES", ()))
    return found


def _shell_home(request: Request) -> Response:
    return get_web(request).render(request, "base.html", {
        "page_title": "Home",
        "placeholder": "The Home page arrives with the library pages.",
    }, nav="home")


class SeedgraphApp(Starlette):
    """Starlette with the security wrappers outside everything, error pages included."""

    def __init__(self, *args: Any, port: int, csrf_token: str, **kwargs: Any) -> None:
        self._sg_port = port
        self._sg_token = csrf_token
        super().__init__(*args, **kwargs)

    def build_middleware_stack(self):  # type: ignore[override]
        return security.secure(super().build_middleware_stack(), port=self._sg_port,
                               token=self._sg_token)


def create_app(home: Path | str, *, pool: Any = None, chain: Any = None,
               clock: Callable[[], datetime] | None = None, port: int = DEFAULT_PORT,
               routes: Sequence[BaseRoute] = ()) -> Starlette:
    """Build the web app for the v3 home ``home`` (its library must exist).

    The lifespan starts the :class:`JobRunner` (which resumes ``conversion_queue``) and
    stops it on shutdown; a ``TestClient`` used without ``with`` never starts it.
    """
    home = Path(home)
    lib_dir = library_dir(home)
    token = security.new_token()
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    templates = Jinja2Templates(env=env, context_processors=[_shell_context])
    web = Web(home=home, port=port, csrf_token=token, templates=templates,
              jobs=JobRunner(lib_dir, pool=pool, clock=clock), libraries=ThreadLibraries(lib_dir),
              pool=pool, chain=chain, clock=clock)

    all_routes: list[BaseRoute] = [*discovered_routes(), *routes]
    if not any(isinstance(r, Route) and r.path == "/" for r in all_routes):
        all_routes.insert(0, Route("/", _shell_home, name="home"))
    all_routes.append(Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"))

    @asynccontextmanager
    async def lifespan(app: Starlette):
        web.jobs.start()
        try:
            yield
        finally:
            web.jobs.stop(shutdown_pool=True)
            web.libraries.close_all()

    app = SeedgraphApp(routes=all_routes, lifespan=lifespan, port=port, csrf_token=token)
    app.state.web = web
    return app
