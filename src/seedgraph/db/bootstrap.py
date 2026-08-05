"""Create + migrate the two scopes."""

from __future__ import annotations

from pathlib import Path

from .. import paths
from .connection import cache_db_path, open_cache_db, open_project_db, project_db_path
from .migrations import run_migrations


def ensure_cache_db(root: Path | str | None = None) -> Path:
    """Create + migrate ``cache/cache.db``; return its path."""
    conn = open_cache_db(root)
    try:
        run_migrations(conn, "cache")
    finally:
        conn.close()
    return cache_db_path(root)


def ensure_project_db(slug: str, root: Path | str | None = None) -> Path:
    """Create + migrate ``projects/{slug}/project.db``; return its path.

    ``slug`` is validated (raises before any filesystem use).
    """
    paths.validate_slug(slug)
    conn = open_project_db(slug, root)
    try:
        run_migrations(conn, "project")
    finally:
        conn.close()
    return project_db_path(slug, root)
