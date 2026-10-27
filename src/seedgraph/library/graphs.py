"""Context graphs: one ``graph.db`` per graph under ``<graphs_root>/<slug>/`` (§1.2, §3).

A graph holds only its own rows (meta, membership, lens choice) and refers to
library papers softly. Joins go through a read-only ``ATTACH`` of the library
(:func:`attached_library`); the in-graph citation view (:func:`graph_citations`)
is derived that way and never stored.

Graphs travel with the library (the migration swaps ``graphs/`` with
``library/``), so every write here also flags the library dirty unless the
library was opened in migration mode.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .. import paths
from .store import Library, LibraryError, is_held, open_migrated, paper_exists, transaction

GRAPH_SCOPE = "graph"
GRAPH_DB = "graph.db"
LIB_ALIAS = "lib"

Status = Literal["member", "frontier", "excluded"]
STATUSES: tuple[str, ...] = ("member", "frontier", "excluded")


class GraphError(LibraryError):
    """A graph operation was refused."""


class NotHeldError(GraphError):
    """A cited-only paper cannot be a member; it belongs on the frontier."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Graph:
    """An open context graph bound to its library."""

    slug: str
    path: Path
    conn: sqlite3.Connection
    lib: Library

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Graph:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class GraphInfo:
    slug: str
    display_name: str
    created_at: str | None
    origin: str | None


@dataclass(frozen=True)
class GraphPaper:
    paper_id: str
    status: str
    is_seed: bool
    reason: str | None
    added_at: str
    origin: str


@dataclass(frozen=True)
class GraphCitation:
    source_paper_id: str
    target_paper_id: str
    provenance: str
    confidence: float
    reference_id: str | None


def graph_db_path(lib: Library, slug: str) -> Path:
    paths.validate_slug(slug)
    return lib.graphs_root / slug / GRAPH_DB


def _open(lib: Library, slug: str) -> Graph:
    path = graph_db_path(lib, slug)
    conn = open_migrated(path, GRAPH_SCOPE, uri=True)
    return Graph(slug=slug, path=path, conn=conn, lib=lib)


@contextmanager
def graph_write(g: Graph) -> Iterator[sqlite3.Connection]:
    g.lib.mark_dirty()  # flag first: a crash after this leaves dirty set, never an unflagged write
    with transaction(g.conn) as conn:
        yield conn


def set_graph_meta(conn: sqlite3.Connection, key: str, value: str | None) -> None:
    conn.execute(
        "INSERT INTO graph_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def create_graph(
    lib: Library,
    slug: str,
    *,
    display_name: str | None = None,
    origin: str = "library",
    created_at: str | None = None,
) -> Graph:
    """Create ``<graphs_root>/<slug>/graph.db``. Refuses a slug that already exists."""
    path = graph_db_path(lib, slug)
    if path.exists():
        raise GraphError(f"graph {slug!r} already exists")
    g = _open(lib, slug)
    try:
        with graph_write(g) as conn:
            set_graph_meta(conn, "slug", slug)
            set_graph_meta(conn, "display_name", display_name or slug)
            set_graph_meta(conn, "created_at", created_at or _now())
            set_graph_meta(conn, "origin", origin)
    except BaseException:
        g.close()
        raise
    return g


def open_graph(lib: Library, slug: str) -> Graph:
    """Open an existing graph (applying any newer graph migrations)."""
    if not graph_db_path(lib, slug).is_file():
        raise GraphError(f"no graph {slug!r} in {lib.graphs_root}")
    return _open(lib, slug)


def graph_info(g: Graph) -> GraphInfo:
    meta = {r[0]: r[1] for r in g.conn.execute("SELECT key, value FROM graph_meta")}
    return GraphInfo(
        slug=meta.get("slug") or g.slug,
        display_name=meta.get("display_name") or g.slug,
        created_at=meta.get("created_at"),
        origin=meta.get("origin"),
    )


def list_graphs(lib: Library) -> list[GraphInfo]:
    """Every graph under the library's graphs root, sorted by slug."""
    root = lib.graphs_root
    if not root.is_dir():
        return []
    out: list[GraphInfo] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not (child / GRAPH_DB).is_file() or not paths.SLUG_RE.match(child.name):
            continue
        with open_graph(lib, child.name) as g:
            out.append(graph_info(g))
    return out


def rename_graph(lib: Library, slug: str, display_name: str) -> None:
    """Change the display name. The slug is fixed."""
    if not display_name or not display_name.strip():
        raise GraphError("display name must not be empty")
    with open_graph(lib, slug) as g, graph_write(g) as conn:
        set_graph_meta(conn, "display_name", display_name)


def set_membership(
    g: Graph,
    paper_id: str,
    status: Status,
    *,
    is_seed: bool = False,
    reason: str | None = None,
    origin: str = "library",
    added_at: str | None = None,
) -> None:
    """Add or update a paper's row in the graph.

    The paper must exist in the library. A ``member`` must be held (has a
    ``paper_files`` row); a cited-only paper raises :class:`NotHeldError`.
    Updating an existing row keeps its ``added_at``.
    """
    if status not in STATUSES:
        raise GraphError(f"invalid graph status {status!r}; expected one of {STATUSES}")
    if not paper_exists(g.lib.conn, paper_id):
        raise GraphError(f"paper {paper_id!r} is not in the library")
    if status == "member" and not is_held(g.lib.conn, paper_id):
        raise NotHeldError(
            f"paper {paper_id!r} is cited-only (no PDF); add it to the frontier until its PDF is added"
        )
    with graph_write(g) as conn:
        conn.execute(
            "INSERT INTO graph_papers (paper_id, status, is_seed, reason, added_at, origin) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(paper_id) DO UPDATE SET status = excluded.status, "
            "is_seed = excluded.is_seed, reason = excluded.reason, origin = excluded.origin",
            (paper_id, status, int(is_seed), reason, added_at or _now(), origin),
        )


def remove_membership(g: Graph, paper_id: str) -> GraphPaper | None:
    """Delete the paper's row from this graph; return the removed row (so callers
    can say whether it was a seed), or None when the paper was not in the graph.
    Nothing in the library changes."""
    with graph_write(g) as conn:
        row = conn.execute(
            "SELECT paper_id, status, is_seed, reason, added_at, origin FROM graph_papers "
            "WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM graph_papers WHERE paper_id = ?", (paper_id,))
    return GraphPaper(row[0], row[1], bool(row[2]), row[3], row[4], row[5])


def membership(g: Graph, *, status: Status | None = None) -> list[GraphPaper]:
    """The graph's paper rows, ordered by ``paper_id``."""
    sql = "SELECT paper_id, status, is_seed, reason, added_at, origin FROM graph_papers"
    args: tuple[str, ...] = ()
    if status is not None:
        sql += " WHERE status = ?"
        args = (status,)
    sql += " ORDER BY paper_id"
    return [
        GraphPaper(r[0], r[1], bool(r[2]), r[3], r[4], r[5]) for r in g.conn.execute(sql, args)
    ]


def choose_lens(g: Graph, lens_id: str, *, origin: str = "library", chosen_at: str | None = None) -> None:
    """Apply a library lens to this graph (idempotent)."""
    with graph_write(g) as conn:
        conn.execute(
            "INSERT INTO graph_lenses (lens_id, chosen_at, origin) VALUES (?, ?, ?) "
            "ON CONFLICT(lens_id) DO NOTHING",
            (lens_id, chosen_at or _now(), origin),
        )


def unchoose_lens(g: Graph, lens_id: str) -> bool:
    """Remove a lens from this graph; ``False`` when it was not chosen. The lens and its
    results stay in the library."""
    if g.conn.execute("SELECT 1 FROM graph_lenses WHERE lens_id = ?", (lens_id,)).fetchone() is None:
        return False
    with graph_write(g) as conn:
        conn.execute("DELETE FROM graph_lenses WHERE lens_id = ?", (lens_id,))
    return True


def chosen_lenses(g: Graph) -> list[str]:
    return [r[0] for r in g.conn.execute("SELECT lens_id FROM graph_lenses ORDER BY lens_id")]


@contextmanager
def attached_library(g: Graph, alias: str = LIB_ALIAS) -> Iterator[sqlite3.Connection]:
    """The graph connection with the library attached READ-ONLY as ``alias``.

    Library tables are then ``<alias>.<table>`` and graph tables ``main.<table>``.
    Detached on exit; fetch results inside the block.
    """
    uri = g.lib.db_path.resolve().as_uri() + "?mode=ro"
    g.conn.execute("ATTACH DATABASE ? AS " + alias, (uri,))
    try:
        yield g.conn
    finally:
        g.conn.execute("DETACH DATABASE " + alias)


def graph_citation_sql(*, include_frontier: bool = False, alias: str = LIB_ALIAS) -> str:
    """SQL for the in-graph citation view, run on a graph connection with the library
    attached as ``alias``: ``citation_links`` restricted to pairs whose both ends are
    members (or members/frontier with ``include_frontier``)."""
    statuses = "('member', 'frontier')" if include_frontier else "('member')"
    return (
        "SELECT cl.source_paper_id, cl.target_paper_id, cl.provenance, cl.confidence, "
        "cl.reference_id "
        f"FROM {alias}.citation_links cl "
        f"JOIN main.graph_papers s ON s.paper_id = cl.source_paper_id AND s.status IN {statuses} "
        f"JOIN main.graph_papers t ON t.paper_id = cl.target_paper_id AND t.status IN {statuses} "
        "ORDER BY cl.source_paper_id, cl.target_paper_id, cl.provenance"
    )


def graph_citations(g: Graph, *, include_frontier: bool = False) -> list[GraphCitation]:
    """The in-graph citation view (derived through the read-only ATTACH)."""
    with attached_library(g) as conn:
        rows = conn.execute(graph_citation_sql(include_frontier=include_frontier)).fetchall()
    return [GraphCitation(*tuple(r)) for r in rows]
