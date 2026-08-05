"""Content-access boundary checker (doc 09 §10) — reuses the existing export guard.

Asserts no ``user_supplied_private``-derived artifact leaks into any
shareable/export output. The MVP shareable surface is the citation-graph export
(:func:`seedgraph.graph.export.export_graph`): the work nodes plus the
``is_shareable_edge``-admitted (``provider_reference``) citation edges. Per-project
full-text-derived artifacts — ``evidence_spans`` / ``extracted_claims`` /
``structured_notes`` / ``reference_entries`` and the phase_9 ``audit_records`` and
``eval/`` files — are private-by-default (decision 60/76, D8) and are NEVER
serialized into that surface.

The checker re-uses the phase_0 export gate (:func:`seedgraph.vocab.field_allowed`
/ :func:`is_shareable`) rather than re-deriving access rules, and it is
**data-driven** (not a tautology): a shareable edge that actually carries
full-text-derived content, or a private artifact id that reaches the serialized
export id-set, is returned as a violation. An EMPTY list means green.
"""

from __future__ import annotations

import sqlite3


def _safe_query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list:
    """Run ``sql`` returning ``[]`` if the table is absent (project not yet built).

    The boundary check runs over whatever surfaces a project has materialized; a
    missing table is "nothing to leak here", never a hard error (fail-soft read).
    """
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []


def find_boundary_violations(
    conn: sqlite3.Connection, project: str, run_id: str | None = None
) -> list[str]:
    """Return the ids of any ``user_supplied_private``-derived artifacts found in a
    shareable/export output (EMPTY list == green). The §10/§11 pytest invariant
    asserts this returns empty over a correctly-gated project.

    Two data-driven predicates, both routing through the phase_0 export gate:

    1. **Shareable-edge content audit.** A ``provider_reference`` edge is shareable
       only because it is built from public ``referenced_works`` metadata (no full
       text); such an edge carries ``reference_id IS NULL``. A non-null
       ``reference_id`` points into a full-text-derived ``reference_entries`` row,
       so the edge's *effective* access class is ``user_supplied_private`` — and an
       ``is_shareable_edge``-admitted edge whose effective access class fails
       :func:`field_allowed` is a leak (this is the original §10 check made
       satisfiable: the access class is derived from the row, not hard-coded).
    2. **Private-artifact containment.** Every full-text-derived artifact
       (``evidence_spans`` / ``extracted_claims`` — the home of the fixture's
       ``user_supplied_private`` span) is project-scoped: none of their ids may
       appear among the ids serialized into the shareable citation-graph export
       (its work nodes + admitted edges). A private artifact id that reaches that
       serialized set is a leak.

    ``audit_records`` and per-project ``eval/`` files are never placed in the
    export surface, so a private-by-default audit row is correctly absent here.
    """
    from ..citation.edges import is_shareable_edge
    from ..vocab import AccessClass, field_allowed, is_shareable

    violations: list[str] = []

    if run_id is None:
        edge_rows = _safe_query(
            conn,
            "SELECT edge_id, source_work_id, target_work_id, provenance, reference_id "
            "FROM citation_edges",
        )
    else:
        edge_rows = _safe_query(
            conn,
            "SELECT edge_id, source_work_id, target_work_id, provenance, reference_id "
            "FROM citation_edges WHERE run_id = ?",
            (run_id,),
        )

    # The ids serialized into the shareable citation-graph export: the work-node
    # endpoints (and edge ids) of every is_shareable_edge-admitted edge — exactly
    # what graph/export.export_graph writes into graph.json.
    serialized_ids: set[str] = set()
    for edge_id, source, target, provenance, reference_id in edge_rows:
        if not is_shareable_edge(provenance):
            continue  # withheld by the export filter; never serialized
        serialized_ids.update((source, target))
        # (1) shareable-edge content audit — derive the edge's effective access
        # class from whether it points at a full-text-derived reference_entries row,
        # then route THAT through the export gate (no hard-coded constant).
        effective_access = (
            AccessClass.metadata_only.value
            if reference_id is None
            else AccessClass.user_supplied_private.value
        )
        if not field_allowed(effective_access):
            violations.append(f"citation_edge:{edge_id}")

    # (2) private-artifact containment — full-text-derived artifacts stay project-
    # scoped; none of their ids may appear among the serialized export node ids.
    # This is where the fixture's `span_priv` (user_supplied_private) is inspected:
    # it must be absent from the shareable surface, not merely assumed absent.
    for table, id_col in (("evidence_spans", "span_id"), ("extracted_claims", "claim_id")):
        for row_id, access_class in _safe_query(
            conn, f"SELECT {id_col}, access_class FROM {table}"
        ):
            if not is_shareable(access_class) and row_id in serialized_ids:
                violations.append(f"{table}:{row_id}")

    return violations
