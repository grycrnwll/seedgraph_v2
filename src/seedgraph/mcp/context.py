"""Server-scoped state for the MCP server (design 00, decisions M6/M7/M8).

Holds the home ``root`` resolved once at startup, the ``--redact-private`` flag,
and a per-slug :class:`ProjectHandle` cache. ``open_project`` idempotently re-runs
migrations, so caching the handle per slug (not per call) is the only state worth
keeping (M8).

SDK-free at module load: the sole SDK touch is the *deferred* ``_tool_error``
import inside :meth:`ServerContext.get_handle` (mapping an invalid/unknown slug to
the M9 ``project_not_found`` tool error). Deferring it breaks the
``server`` <-> ``context`` import cycle — ``server.py`` imports
:class:`ServerContext` at module load, so this module must not import from
``server.py`` at the top level.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .. import paths
from ..db.connection import open_project_db
from ..errors import ValidationError
from ..project.service import ProjectHandle, open_project


@dataclass
class ServerContext:
    """Resolved home, the redaction flag, and the per-slug handle cache (M8).

    ``root`` is already resolved (via ``paths.resolve_home``) by the time it
    reaches here — the server resolves home once at startup (M6) and threads the
    concrete path through every service call.
    """

    root: Path
    redact_private: bool = False
    _handles: dict[str, ProjectHandle] = field(default_factory=dict, repr=False)

    def get_handle(self, slug: str) -> ProjectHandle:
        """Return a cached :class:`ProjectHandle` for ``slug``, opening it once (M8).

        Validates the slug and opens the project on a cache miss. An invalid or
        unknown slug surfaces as the M9 structured ``project_not_found`` tool
        error (``open_project`` raises :class:`ValidationError` for both a
        malformed slug and a missing project directory).
        """
        cached = self._handles.get(slug)
        if cached is not None:
            return cached
        try:
            paths.validate_slug(slug)
            handle = open_project(slug, root=self.root)
        except ValidationError as exc:
            # Deferred import (see module docstring): avoids the server<->context
            # load-time cycle. get_handle is never called at import time, so this
            # resolves cleanly once both modules are loaded. The raised ToolError
            # is implicitly chained to `exc` (except-block __context__).
            from .server import _tool_error

            _tool_error("project_not_found", str(exc), slug=slug)
        self._handles[slug] = handle
        return handle

    @contextmanager
    def project_conn(self, handle: ProjectHandle) -> Iterator[sqlite3.Connection]:
        """Yield a fresh ``project.db`` connection (M7), closed on exit.

        A new connection per call, opened through ``open_project_db`` (WAL +
        ``busy_timeout=5000`` + FK) — sqlite3 connections are thread-affine and
        FastMCP schedules sync tool bodies on worker threads, so a long-lived
        cached connection would be unsafe. Opens are cheap; ``busy_timeout``
        absorbs contention with a concurrent ``serve`` process.
        """
        conn = open_project_db(handle.slug, root=self.root)
        try:
            yield conn
        finally:
            conn.close()
