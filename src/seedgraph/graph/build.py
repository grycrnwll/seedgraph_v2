"""Build a NetworkX view of the citation graph from durable edge rows.

The relational ``citation_edges`` rows are the source of truth; the NetworkX
``DiGraph`` is a *computed view* (doc 07 §1-2). :func:`build_citation_graph`
assembles that view from :func:`citation.edges.authoritative_edges` for a given
``run_id`` — one directed ``cites`` edge per authoritative ``(source, target)``
pair.

Closed-world (default) restricts nodes to the included corpus
(``project_documents.inclusion_status='included'``), dropping out-of-corpus
metadata-only stub targets; ``closed_world=False`` (the ``--open-world`` CLI
option) keeps those metadata-only targets in the graph (decision 13).

``networkx`` is imported lazily inside the function so the package stays
import-clean when networkx is not installed (it is a phase_2 runtime dependency
that the wiring stage adds to pyproject — see needs_wiring).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx


def _work_attrs(conn: sqlite3.Connection) -> dict[str, dict]:
    """``work_id -> {title, year, inclusion_status}`` for every work (LEFT JOIN so a
    work with no membership row still resolves, with ``inclusion_status=None``)."""
    cursor = conn.execute(
        "SELECT w.work_id, w.canonical_title, w.year, d.inclusion_status "
        "FROM works w "
        "LEFT JOIN project_documents d ON d.work_id = w.work_id"
    )
    attrs: dict[str, dict] = {}
    for work_id, title, year, inclusion_status in cursor.fetchall():
        attrs[work_id] = {
            "title": title,
            "year": year,
            "inclusion_status": inclusion_status,
        }
    return attrs


def _included_work_ids(conn: sqlite3.Connection) -> set[str]:
    """The closed-world node set: works whose membership is ``included``."""
    cursor = conn.execute(
        "SELECT work_id FROM project_documents WHERE inclusion_status = 'included'"
    )
    return {row[0] for row in cursor.fetchall()}


def build_citation_graph(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    closed_world: bool = True,
) -> "networkx.DiGraph":
    """Assemble an ``nx.DiGraph`` from the authoritative edges of ``run_id``.

    Nodes are ``works`` (work_id), edges are directed ``cites`` relations taken
    from :func:`citation.edges.authoritative_edges` (highest-authority provenance
    per source->target). With ``closed_world=True`` (default) the node set is the
    included corpus only (``project_documents.inclusion_status='included'``) and
    edges to out-of-corpus metadata-only stub targets are dropped; with
    ``closed_world=False`` those metadata-only targets are retained
    (``--open-world``; decision 13). Edge/node attributes carry at least
    ``provenance`` and ``confidence`` so :func:`graph.export.export_graph` can
    apply the ``is_shareable_edge`` filter on the way out.
    """
    import networkx as nx

    from ..citation.edges import authoritative_edges

    edges = authoritative_edges(conn, run_id)
    attrs = _work_attrs(conn)
    included = _included_work_ids(conn)

    graph = nx.DiGraph()

    def _add_node(work_id: str) -> None:
        if work_id not in graph:
            graph.add_node(work_id, **attrs.get(work_id, {}))

    # Closed-world (default): node set is the included corpus only; an edge survives
    # only when BOTH endpoints are included (out-of-corpus metadata-only stub targets
    # are dropped). Open-world (--open-world): every authoritative endpoint is kept as
    # a node, retaining metadata-only stub targets (decision 13).
    if closed_world:
        for work_id in included:
            _add_node(work_id)
    else:
        for work_id in attrs:
            if attrs[work_id].get("inclusion_status") == "included":
                _add_node(work_id)

    for edge in edges:
        source = edge["source_work_id"]
        target = edge["target_work_id"]
        if closed_world:
            if source not in included or target not in included:
                continue
        else:
            _add_node(source)
            _add_node(target)
        graph.add_edge(
            source,
            target,
            edge_type=edge["edge_type"],
            provenance=edge["provenance"],
            confidence=edge["confidence"],
        )

    return graph
