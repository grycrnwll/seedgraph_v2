"""Assemble the computed NetworkX view from relational rows (plan §5, step 10).

The relational rows are the source of truth; the NetworkX ``DiGraph`` is a
*computed view* (doc 07 §2, §3, §15) reconstructable per ``run_id``. Nodes:
``works``, ``concepts``, and ``concept_aliases`` (as ``ConceptAlias`` nodes).
Edges: ``citation_edges`` (folded in, marked shareable/deterministic),
``claim_concepts`` (``mentions_concept``), ``project_graph_edges`` (``discusses``
/ interpretive / ``co_occurs_with``), and ``has_alias`` edges projected from
``concept_aliases``. Every edge carries ``epistemic_type`` + ``access_class`` as
attributes so the export layer (and downstream answer layer) can hedge.

NetworkX is imported lazily inside the builder so this module stays import-clean
even where the dependency is absent.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from ..display import derive_label

if TYPE_CHECKING:
    import networkx as nx


def _parse_authors(authors_json) -> list | None:
    """``works.authors`` is a JSON array (string) in raw SQL; parse defensively."""
    if authors_json is None:
        return None
    if isinstance(authors_json, list):
        return authors_json
    try:
        parsed = json.loads(authors_json)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def _alias_node_id(concept_id: str, alias_label: str) -> str:
    """Stable ConceptAlias node id (namespaced so it never collides with a work/concept)."""
    return f"alias::{concept_id}::{alias_label}"


def unread_work_map(conn: sqlite3.Connection) -> dict[str, bool]:
    """``{work_id: unread}`` for the read-next analysis (Build B chunk 2).

    A work is **unread** when the project has no extracted content to read from
    it: zero ``evidence_spans`` rows, or membership ``metadata_only``. An
    ``excluded`` work is NOT unread (it will never be recommended). One SQL
    pass; the shared derivation behind export / graph3d / answer read-next.
    """
    rows = conn.execute(
        "SELECT w.work_id, d.inclusion_status, "
        "(SELECT COUNT(*) FROM evidence_spans s WHERE s.work_id = w.work_id) "
        "FROM works w LEFT JOIN project_documents d ON d.work_id = w.work_id"
    ).fetchall()
    out: dict[str, bool] = {}
    for work_id, status, span_count in rows:
        if status == "excluded":
            out[work_id] = False
        else:
            out[work_id] = status == "metadata_only" or not span_count
    return out


def build_graph(conn: sqlite3.Connection, *, run_id: str) -> "nx.DiGraph":
    """Assemble and return a NetworkX ``DiGraph`` for the run.

    Loads works/concepts/concept_aliases as nodes and
    citation_edges + project_graph_edges (the ``discusses`` aggregation of
    ``claim_concepts`` + interpretive + ``co_occurs_with``) + ``has_alias`` as
    edges, attaching ``epistemic_type`` and ``access_class`` to EVERY edge and
    marking citation edges shareable/deterministic (``is_citation``). Deterministic
    given the same rows (``test_graph_export`` asserts the rebuild is stable).
    Lazy-imports ``networkx``.
    """
    import networkx as nx

    graph = nx.DiGraph()

    # --- nodes: works -------------------------------------------------------
    # authors/doi/arxiv_id/openalex_id are D8-shareable metadata; they feed the
    # never-blank display `label` (Build B chunk 9) and the BibTeX/RIS writers.
    for (
        work_id,
        title,
        year,
        authors_json,
        doi,
        arxiv_id,
        openalex_id,
    ) in conn.execute(
        "SELECT work_id, canonical_title, year, authors, doi, arxiv_id, openalex_id "
        "FROM works"
    ).fetchall():
        authors = _parse_authors(authors_json)
        graph.add_node(
            work_id,
            node_type="Work",
            title=title,
            year=year,
            authors=authors,
            doi=doi,
            arxiv_id=arxiv_id,
            openalex_id=openalex_id,
            label=derive_label(
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                openalex_id=openalex_id,
                arxiv_id=arxiv_id,
                work_id=work_id,
            ),
        )

    # --- nodes: concepts ----------------------------------------------------
    for (
        concept_id,
        canonical_label,
        concept_type,
        access_class,
        epistemic_type,
        definition,
        paper_frequency,
        weight,
        status,
    ) in conn.execute(
        "SELECT concept_id, canonical_label, concept_type, access_class, epistemic_type, "
        "definition, paper_frequency, weight, status FROM concepts"
    ).fetchall():
        graph.add_node(
            concept_id,
            node_type="Concept",
            canonical_label=canonical_label,
            concept_type=concept_type,
            access_class=access_class,
            epistemic_type=epistemic_type,
            definition=definition,
            paper_frequency=paper_frequency,
            # IDF discriminativeness (decision 71) — NOT importance.
            weight=weight,
            status=status,
        )

    # --- nodes + edges: concept_aliases (ConceptAlias nodes + has_alias) -----
    for concept_id, alias_label, fold_reason, epistemic_type in conn.execute(
        "SELECT c.concept_id, a.alias_label, a.fold_reason, a.epistemic_type "
        "FROM concept_aliases a JOIN concepts c ON c.concept_id = a.concept_id"
    ).fetchall():
        node_id = _alias_node_id(concept_id, alias_label)
        graph.add_node(
            node_id,
            node_type="ConceptAlias",
            alias_label=alias_label,
            fold_reason=fold_reason,
        )
        concept_access = graph.nodes.get(concept_id, {}).get(
            "access_class", "user_supplied_private"
        )
        graph.add_edge(
            concept_id,
            node_id,
            edge_type="has_alias",
            epistemic_type=epistemic_type,
            access_class=concept_access,
            is_citation=False,
        )

    # --- edges: citation_edges (folded in; deterministic + shareable) --------
    # citation_edges carry no access_class; they are marked deterministic so
    # is_shareable_citation_edge keeps the metadata-class graph shareable. The
    # interpretive llm_inferred provenance NEVER lands on a citation edge.
    for source_work_id, target_work_id, edge_type in conn.execute(
        "SELECT source_work_id, target_work_id, edge_type FROM citation_edges"
    ).fetchall():
        # defensive: an endpoint absent from `works` still gets a never-blank label.
        if source_work_id not in graph:
            graph.add_node(source_work_id, node_type="Work",
                           label=derive_label(work_id=source_work_id))
        if target_work_id not in graph:
            graph.add_node(target_work_id, node_type="Work",
                           label=derive_label(work_id=target_work_id))
        graph.add_edge(
            source_work_id,
            target_work_id,
            edge_type=edge_type or "cites",
            epistemic_type="deterministic",
            access_class="metadata_only",
            is_citation=True,
        )

    # --- edges: project_graph_edges (discusses / interpretive / co_occurs) ---
    for (
        source_node_id,
        target_node_id,
        edge_type,
        epistemic_type,
        access_class,
        confidence,
        shared_count,
    ) in conn.execute(
        "SELECT source_node_id, target_node_id, edge_type, epistemic_type, access_class, "
        "confidence, shared_count FROM project_graph_edges"
    ).fetchall():
        if source_node_id not in graph or target_node_id not in graph:
            # endpoint pruned (e.g. a concept removed by a rebuild) — skip the edge
            continue
        graph.add_edge(
            source_node_id,
            target_node_id,
            edge_type=edge_type,
            epistemic_type=epistemic_type,
            access_class=access_class,
            # co-occurrence strength provenance (Jaccard + shared works); NULL
            # on edge types that carry no strength signal.
            confidence=confidence,
            shared_count=shared_count,
            is_citation=False,
        )

    return graph
