"""Citation-edge persistence: write/dedup, provenance authority, share gate.

This module owns the *write side* of ``citation_edges`` (project.db). It is the
forward-compatible seam the deferred parsed-bibliography phase (phase_3b) reuses:

* :func:`write_edge` inserts one ``citation_edges`` row, collapsing duplicates via
  the table's ``UNIQUE(source_work_id, target_work_id, edge_type, provenance,
  run_id)`` constraint (union storage + dedup — decisions 5/13/26/45/66).
* :func:`authoritative_edges` projects, per ``run_id``, the single highest-authority
  edge per ``(source, target)`` using :data:`PROVENANCE_AUTHORITY`
  (``manual_override > parsed_bibliography > provider_reference``). With phase_2's
  single provenance the authority resolution is trivially satisfied, but the seam
  is exercised so phase_3b's second provenance slots in without change.
* :func:`is_shareable_edge` is the export gate for edges: a ``provider_reference``
  edge is built from public ``referenced_works`` metadata only (no full text), so
  it is metadata-class / shareable; everything else fails closed (decision 5, D8).

Schema authority is ``db/schema/project/0004_citation_graph.sql`` (D6); nothing
here authors schema. All connections run ``PRAGMA foreign_keys=ON`` (decision 8),
so a ``source``/``target``/``reference_id`` that is not a real row fails closed.

NOTE (naming collision flagged to the wiring stage): the phase_2 plan names this
edge-provenance vocabulary ``Provenance`` / ``PROVENANCE_AUTHORITY`` in vocab.py,
but vocab.py already binds both names to the D2 *epistemic_type* 6-tier concept
(deterministic/metadata_resolved/llm_extracted/...). Those are a DIFFERENT concept
from citation-edge provenance (provider_reference/parsed_bibliography/
manual_override). To avoid clobbering the D2 vocabulary, the citation-edge
authority map is defined locally here (per the section 6.3 signature, which shows
it as an ``edges.py`` module constant). See needs_wiring / blocking_decisions.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

# Citation-edge provenance authority (decision 13): higher integer outranks lower
# when collapsing to the authoritative edge per (source, target) per run. This is
# the citation-edge concept — distinct from vocab.PROVENANCE_AUTHORITY, which maps
# the D2 epistemic_type tiers. See module docstring.
PROVENANCE_AUTHORITY: dict[str, int] = {
    "manual_override": 2,
    "parsed_bibliography": 1,
    "provider_reference": 0,
}

# The only provenance phase_2 writes; the rest are forward seams (phase_3b).
PROVIDER_REFERENCE = "provider_reference"

# Columns a write/read of a citation edge selects (positional, row_factory-agnostic).
_EDGE_COLS = (
    "source_work_id",
    "target_work_id",
    "edge_type",
    "provenance",
    "confidence",
    "reference_id",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_edge(
    conn: sqlite3.Connection,
    *,
    source: str,
    target: str,
    provenance: str,
    confidence: float,
    run_id: str,
    reference_id: str | None = None,
) -> None:
    """Insert one ``citation_edges`` row, deduped by the table UNIQUE constraint.

    Writes a single ``edge_type='cites'`` edge from ``source`` work to ``target``
    work under ``run_id``. A duplicate ``(source, target, 'cites', provenance,
    run_id)`` collapses silently via ``INSERT ... ON CONFLICT DO NOTHING`` (union
    storage + dedup; decisions 5/13/26/45/66). ``reference_id`` is NULL for
    provider edges and set by phase_3b for parsed edges. ``confidence`` is 1.0 for
    ``provider_reference``. The caller is responsible for having allocated
    ``run_id`` at the start of the invocation (must-fix #3) so the NOT-NULL
    ``run_id`` column is always satisfied — including standalone ``cite project``.

    Both endpoints must already exist as ``works`` rows (decision D3 / 59); with
    ``foreign_keys=ON`` a missing endpoint raises ``IntegrityError`` (fail-closed,
    never auto-vivified — phase_5b is the sole target materializer).
    """
    conn.execute(
        "INSERT INTO citation_edges "
        "(source_work_id, target_work_id, edge_type, provenance, confidence, "
        " reference_id, run_id, created_at) "
        "VALUES (?, ?, 'cites', ?, ?, ?, ?, ?) "
        "ON CONFLICT(source_work_id, target_work_id, edge_type, provenance, run_id) "
        "DO NOTHING",
        (source, target, provenance, float(confidence), reference_id, run_id, _now_iso()),
    )


def authoritative_edges(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """Return the highest-authority edge per ``(source, target)`` for ``run_id``.

    Reads every ``citation_edges`` row scoped to ``run_id`` and, where several
    provenances assert the same ``(source_work_id, target_work_id)`` pair, keeps
    only the one with the greatest :data:`PROVENANCE_AUTHORITY` rank
    (``manual_override > parsed_bibliography > provider_reference``). Each result
    dict carries at least ``source_work_id``, ``target_work_id``, ``edge_type``,
    ``provenance``, ``confidence``, ``reference_id``. This is the durable-rows ->
    computed-view step (doc 07 §2) consumed by ``graph/build.py``. With phase_2's
    single provenance there is no disagreement to resolve, but the seam is live so
    phase_3b's second provenance composes without change (``edge_disagreements`` is
    intentionally NOT defined until that second provenance exists — §2).
    """
    cursor = conn.execute(
        f"SELECT {', '.join(_EDGE_COLS)} FROM citation_edges WHERE run_id = ?",
        (run_id,),
    )
    best: dict[tuple[str, str], tuple[int, dict]] = {}
    for raw in cursor.fetchall():
        row = dict(zip(_EDGE_COLS, raw))
        key = (row["source_work_id"], row["target_work_id"])
        rank = PROVENANCE_AUTHORITY.get(row["provenance"], -1)
        current = best.get(key)
        if current is None or rank > current[0]:
            best[key] = (rank, row)
    return [row for _rank, row in best.values()]


def is_shareable_edge(provenance: str) -> bool:
    """True iff ``provenance`` denotes a metadata-class (shareable) edge.

    A ``provider_reference`` edge derives only from public ``referenced_works``
    metadata (no full text), so it is shareable by construction (decision 5, D8);
    every other provenance fails closed. This gate filters edges on the way into
    the shareable ``graph.json`` export.
    """
    return provenance == PROVIDER_REFERENCE


# SQL group-by surfacing every (source, target) pair under a run that carries more
# than one distinct provenance (§6.3). The winner is computed in PYTHON from the
# single :data:`PROVENANCE_AUTHORITY` dict — the ladder is NOT duplicated in SQL.
_DISAGREEMENT_SQL = (
    "SELECT source_work_id, target_work_id, "
    "       GROUP_CONCAT(DISTINCT provenance) AS provenances, "
    "       COUNT(DISTINCT provenance)        AS n_prov "
    "FROM citation_edges "
    "WHERE run_id = ? AND edge_type = 'cites' "
    "GROUP BY source_work_id, target_work_id "
    "HAVING n_prov > 1"
)


def edge_disagreements(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """Surface (source, target) pairs under ``run_id`` carrying ≥2 provenances.

    The *confirm / overlap* set (decision 26 / AGENTS.md §2.3 "surface, never
    hide"): one SQL group-by returns each pair's distinct provenance set; the
    authoritative winner is mapped **in Python from the single
    :data:`PROVENANCE_AUTHORITY` dict** (no second SQL ladder — §6.3). A pair with a
    single provenance (an adds-coverage edge) is NOT a disagreement and is absent.

    Returns ``[{source, target, provenances:[sorted...], authoritative_provenance}]``.
    """
    out: list[dict] = []
    for src, tgt, prov_csv, _n in conn.execute(_DISAGREEMENT_SQL, (run_id,)).fetchall():
        provs = (prov_csv or "").split(",")
        winner = max(provs, key=lambda p: PROVENANCE_AUTHORITY.get(p, -1))
        out.append(
            {
                "source": src,
                "target": tgt,
                "provenances": sorted(provs),
                "authoritative_provenance": winner,
            }
        )
    return out
