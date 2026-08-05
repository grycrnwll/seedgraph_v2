"""Materialize + export a run's citation-graph view (the deterministic projection).

:func:`build_and_export` is the single entry point both adapters — the Typer CLI
(``cite project`` / ``cite parse``) and the Track-2 web run-planner
(``web.planner._job_cite``) — go through to turn a run's ``citation_edges`` into
the on-disk ``graph.json`` + ``manifest.json`` under ``runs/{run_id}/``. It
orchestrates the ``graph`` package (``build`` + ``export``), the deterministic
edge projection (``citation.edges``), and the disjoint ``citation`` manifest
section; it reads ``project.db`` and writes NOTHING back to it.

This lives beside the modules it drives (``graph.build`` / ``graph.export``) so
neither adapter has to reach into the other's private helpers to run it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from ..db.connection import connect_project_raw, project_db_path


def _citation_manifest(
    slug: str, run_id: str, g, edges: list[dict], *, root: Optional[Path] = None
) -> dict:
    """Build the disjoint ``citation`` manifest section (decision 35) — no secrets.

    ``config_snapshot`` is the secret-stripped effective project config and
    ``config_fingerprint`` its sha256 over canonical key-sorted JSON, so the
    section round-trips: ``fingerprint == config_fingerprint(config_snapshot)``
    (Build F ch5; replaces the old slug-hash that was identical for every run).
    """
    import networkx

    from .. import __version__
    from ..cache.provider_cache import DEFAULT_TTL_SECONDS
    from ..config.loader import (
        config_fingerprint,
        effective_config_snapshot,
        load_project_config,
    )

    provenance_counts: dict[str, int] = {}
    for edge in edges:
        provenance_counts[edge["provenance"]] = provenance_counts.get(edge["provenance"], 0) + 1
    snapshot = effective_config_snapshot(load_project_config(slug, root))
    return {
        "run_id": run_id,
        "config_snapshot": snapshot,
        "config_fingerprint": config_fingerprint(snapshot),
        "tool_versions": {
            "seedgraph": __version__,
            "networkx": networkx.__version__,
            "sqlite": sqlite3.sqlite_version,
        },
        "source_works": sorted({edge["source_work_id"] for edge in edges}),
        "provenance_counts": provenance_counts,
        "edge_count": g.number_of_edges(),
        "node_count": g.number_of_nodes(),
        "provider_cache_ttl_days": DEFAULT_TTL_SECONDS // 86400,
    }


def build_and_export(
    slug: str, root: Optional[Path], project_root: Path, run_id: str, *, open_world: bool
):
    """Build the NetworkX view for ``run_id`` and export graph.json + manifest.json."""
    from ..citation.edges import authoritative_edges
    from .build import build_citation_graph
    from .export import export_graph

    conn = connect_project_raw(project_db_path(slug, project_root))
    try:
        graph = build_citation_graph(conn, run_id=run_id, closed_world=not open_world)
        edges = authoritative_edges(conn, run_id)
    finally:
        conn.close()
    out_dir = project_db_path(slug, project_root).parent / "runs" / run_id
    manifest = _citation_manifest(slug, run_id, graph, edges, root=project_root)
    export_graph(graph, out_dir, manifest={"citation": manifest})
    return graph, out_dir
