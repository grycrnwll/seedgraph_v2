"""FastAPI app factory + module-global instance.

``create_app(root=None)`` builds the fully-wired app: the ``/health`` probe, the
existing localhost JSON read-routers (cache + citations), the one-time ``/auth``
bootstrap router, and the Jinja2 ``/ui`` surface (via ``attach_ui``). It mints the
per-process serve secrets onto ``app.state`` (``session_token`` / ``csrf_token``)
and defaults ``remote_bind`` off. The module-global ``app = create_app()`` keeps the
existing ``TestClient`` / JSON read-route callers green.

The CLI ``serve`` command runs this app under uvicorn behind the loopback +
session-cookie gate in :mod:`seedgraph.web.serve` (localhost-first; content-access
fail-closed). There is still no HTTP write/answer surface over private artifacts —
mutation flows through the service layer (decision 81).
"""

from __future__ import annotations

import secrets
import sqlite3
from pathlib import Path

from fastapi import FastAPI

from .. import __version__
from ..cache.routes import cache_router
from ..db.connection import cache_db_path
from ..web.routes import router as citations_router
from ..web.serve import auth_router
from ..web.ui import attach_ui


def create_app(root: Path | str | None = None) -> FastAPI:
    """Build a fully-wired seedgraph app (JSON routers + ``/auth`` + ``/ui``).

    ``root`` is recorded on ``app.state.root`` for later home resolution; the live
    home is pinned by ``serve`` via ``$SEEDGRAPH_HOME`` before any request arrives.
    """
    app = FastAPI(title="seedgraph", version=__version__)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "sqlite": sqlite3.sqlite_version,
            "cache_db": cache_db_path().exists(),
            "version": __version__,
        }

    # Localhost JSON read-routers (decision 81): cache health + citation-graph reads.
    app.include_router(cache_router)
    app.include_router(citations_router)
    # One-time /auth bootstrap → HttpOnly, SameSite=Strict session cookie.
    app.include_router(auth_router)
    # Jinja2 /ui surface + /static asset tree.
    attach_ui(app)
    # Track 3 — local-session-only 3D graph workspace (graph view dispatch, the
    # in-memory graph3d.json, provenance drawers, gated source-file serving). Every
    # route is Depends(require_local_session); imported here (after attach_ui) so the
    # shared Jinja env is already configured.
    from ..web.graph3d_routes import graph3d_router

    app.include_router(graph3d_router)

    app.state.root = root
    # Per-process serve secrets (cross-cutting #1): one-time bootstrap token + CSRF.
    app.state.session_token = secrets.token_urlsafe(32)
    app.state.csrf_token = secrets.token_urlsafe(32)
    app.state.remote_bind = False
    return app


# Module-global instance — existing TestClient/read-route callers import this.
app = create_app()
