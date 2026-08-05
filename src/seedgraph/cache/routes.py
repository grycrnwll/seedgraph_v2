"""Minimal FastAPI cache read router (decision 81 — demand-driven serve surface).

Only ``GET /cache/health`` ships this phase; read-list endpoints
(``/cache/source_files``, ``/cache/markdown/{id}``) are NOT built until a real UI
consumes them. This router is NOT yet mounted — the wiring stage adds
``app.include_router(cache_router)`` to ``seedgraph/api/app.py`` (a shared file
this phase must not edit).
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import paths

cache_router = APIRouter(tags=["cache"])


@cache_router.get("/cache/health")
def cache_health() -> dict:
    """Report cache liveness: resolved cache root + whether ``cache.db`` exists."""
    root = paths.cache_root()
    return {
        "ok": True,
        "cache_root": str(root),
        "db_initialized": (root / "cache.db").exists(),
    }
