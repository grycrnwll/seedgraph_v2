"""Emit the computed graph through the default-deny access guard (step 11).

Writes ``graph.json`` / ``graph.graphml`` / ``nodes.csv`` + ``edges.csv`` under
``projects/{slug}/runs/{run_id}/`` and ``.../exports/`` (doc 07 §4.5). Every node
and edge is projected through the export gate:

  * concepts/edges → phase_0 :func:`vocab.field_allowed` on their stamped
    ``access_class`` (D8 default-deny; private withheld unless ``allow_private``);
  * citation edges → phase_7-local :func:`access.is_shareable_citation_edge`
    (they carry no ``access_class``);
  * ``definition`` is dropped for any concept that fails the gate even when its
    label is emitted (definition is full-text-derived; decision 21).

Every emitted edge carries ``epistemic_type`` (the graph is never a truth
source; doc 07 §13). Also updates ``manifest.json`` with concept/edge counts, an
access-class summary, and ``concept_mode`` (shallow section-keyed write, D5).
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .. import paths
from ..display import extract_surname
from ..vocab import field_allowed, is_shareable
from .access import is_shareable_citation_edge
from .graph_build import build_graph

if TYPE_CHECKING:  # pragma: no cover - typing only
    from . import SemanticBuildReport

ExportFormat = Literal["json", "graphml", "csv", "bibtex", "ris"]

# Node-link edges key (pinned so writer/reader agree across networkx versions).
_EDGES_KEY = "links"


def _run_dir(slug: str, run_id: str, root: Path | str | None) -> Path:
    return paths.project_dir(slug, root) / "runs" / run_id


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _node_allowed(attrs: dict, allow_private: bool) -> bool:
    """Default-deny node gate. Works (citation metadata) always pass; concepts and
    their alias nodes pass only when shareable (or ``allow_private``)."""
    node_type = attrs.get("node_type")
    if node_type == "Work":
        return True
    if allow_private:
        return True
    if node_type == "Concept":
        return field_allowed(attrs.get("access_class"))
    # ConceptAlias visibility is decided by its parent concept (handled by caller).
    return False


def _filtered_graph(graph, allow_private: bool):
    """Return a new DiGraph with every node/edge projected through the export gate."""
    import networkx as nx

    out = nx.DiGraph()
    # graph-level attrs (the deterministic ``analysis`` block) ride through the
    # gate: they reference Work ids only, which always pass (Build B chunk 2).
    out.graph.update(graph.graph)
    kept: set[str] = set()
    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type")
        if node_type == "ConceptAlias":
            continue  # decided alongside its parent concept below
        if not _node_allowed(attrs, allow_private):
            continue
        clean = dict(attrs)
        # definition is full-text-derived: drop it for any non-shareable concept,
        # even when the node itself is emitted under --allow-private (decision 21).
        if node_type == "Concept" and not is_shareable(attrs.get("access_class")):
            clean["definition"] = None
        out.add_node(node, **clean)
        kept.add(node)

    # ConceptAlias nodes + has_alias edges ride along with a kept concept.
    for source, target, attrs in graph.edges(data=True):
        if attrs.get("edge_type") == "has_alias" and source in kept:
            out.add_node(target, **graph.nodes[target])
            kept.add(target)

    for source, target, attrs in graph.edges(data=True):
        if attrs.get("is_citation"):
            shareable = is_shareable_citation_edge(attrs.get("epistemic_type", ""))
            if not (shareable or allow_private):
                continue
        else:
            if not allow_private and not field_allowed(attrs.get("access_class")):
                continue
        if source in kept and target in kept:
            out.add_edge(source, target, **attrs)
    return out


def _node_link(graph) -> dict:
    import networkx as nx

    try:
        return nx.node_link_data(graph, edges=_EDGES_KEY)
    except TypeError:  # pragma: no cover - older networkx
        return nx.node_link_data(graph)


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def export_graph(
    conn: sqlite3.Connection,
    *,
    slug: str,
    run_id: str,
    fmt: ExportFormat = "json",
    allow_private: bool = False,
    root: Path | str | None = None,
) -> list[Path]:
    """Build the computed view and write the requested export(s); return the
    written paths.

    Routes every node/edge through the default-deny guard (see module docstring).
    ``runs/{run_id}/graph.json`` is **always public-safe** (cross-cutting #2): it is
    written with the default-deny filter regardless of ``allow_private``, so the
    localhost ``graph.json`` read route can never leak private definitions/edges.
    ``allow_private=True`` (CLI ``--allow-private``) instead writes the private
    full-text-derived view to a **separate** ``exports/graph.private.json`` (local
    only; never the public run-dir artifact); when ``fmt`` is GraphML/CSV that
    explicit export also honors the flag. Pinned by ``test_graph_export`` /
    ``test_access_export_guard``.
    """
    import networkx as nx

    from ..graph.analyze import annotate_citation_communities
    from .graph_build import unread_work_map

    graph = build_graph(conn, run_id=run_id)
    # Build B chunk 2: deterministic citation-community analysis annotated onto
    # the computed view before filtering (additive node attrs + graph-level
    # ``analysis`` block; Concept nodes get community=None, never clustered).
    # conn may be None when build_graph is stubbed/injected (e.g. the public-safe
    # regression test); the unread derivation is the only direct conn use here.
    unread = unread_work_map(conn) if conn is not None else {}
    analysis = annotate_citation_communities(graph, unread=unread)
    # graph.json is ALWAYS the public-safe view — never leaks private content.
    public = _filtered_graph(graph, allow_private=False)
    # Explicit export formats (graph.private.json / graphml / csv) live under
    # exports/ and honor --allow-private; they are never the run-dir graph.json.
    view = public if not allow_private else _filtered_graph(graph, allow_private=True)

    run_dir = _run_dir(slug, run_id, root)
    exports_dir = run_dir / "exports"
    written: list[Path] = []

    # graph.json is always (re)written with the PUBLIC filter so the run dir (and the
    # HTTP read route that serves it) reflects the latest public-safe view.
    graph_json = run_dir / "graph.json"
    _atomic_write(graph_json, json.dumps(_node_link(public), indent=2, sort_keys=True))
    written.append(graph_json)

    # --allow-private writes the private node-link view to a SEPARATE exports file
    # (cross-cutting #2): private viewing is local-only, never the public graph.json.
    if allow_private:
        private_json = exports_dir / "graph.private.json"
        _atomic_write(
            private_json, json.dumps(_node_link(view), indent=2, sort_keys=True)
        )
        written.append(private_json)

    if fmt == "graphml":
        # GraphML rejects None attribute values; stringify every attr defensively.
        gml = nx.DiGraph()
        for node, attrs in view.nodes(data=True):
            gml.add_node(node, **{k: _stringify(v) for k, v in attrs.items()})
        for source, target, attrs in view.edges(data=True):
            gml.add_edge(source, target, **{k: _stringify(v) for k, v in attrs.items()})
        path = exports_dir / "graph.graphml"
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(gml, str(path))
        written.append(path)
    elif fmt == "csv":
        nodes_csv = exports_dir / "nodes.csv"
        edges_csv = exports_dir / "edges.csv"
        nodes_csv.parent.mkdir(parents=True, exist_ok=True)
        node_buf = _nodes_csv(view)
        edge_buf = _edges_csv(view)
        _atomic_write(nodes_csv, node_buf)
        _atomic_write(edges_csv, edge_buf)
        written.extend([nodes_csv, edges_csv])
    elif fmt == "bibtex":
        path = exports_dir / "corpus.bib"
        _atomic_write(path, _bibtex_text(_biblio_records(view)))
        written.append(path)
    elif fmt == "ris":
        path = exports_dir / "corpus.ris"
        _atomic_write(path, _ris_text(_biblio_records(view)))
        written.append(path)

    # Idempotent section-keyed manifest write (D5): the analysis summary counts.
    stats = analysis.get("stats", {})
    _write_manifest_section(
        slug=slug,
        run_id=run_id,
        section="analysis",
        payload={
            "run_id": run_id,
            "community_count": stats.get("community_count", 0),
            "god_node_count": stats.get("god_node_count", 0),
            "bridge_edge_count": stats.get("bridge_edge_count", 0),
            "unread_count": stats.get("unread_count", 0),
            "epistemic_type": "deterministic",
        },
        root=root,
    )

    return written


def _nodes_csv(graph) -> str:
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["node_id", "node_type", "label", "access_class", "epistemic_type"])
    for node, attrs in sorted(graph.nodes(data=True)):
        # Work nodes fall through to the never-blank derived `label` (chunk 9)
        # so a title-less stub shows a visible placeholder, never a blank cell.
        label = (attrs.get("canonical_label") or attrs.get("alias_label")
                 or attrs.get("title") or attrs.get("label") or "")
        writer.writerow(
            [
                node,
                attrs.get("node_type", ""),
                label,
                attrs.get("access_class", ""),
                attrs.get("epistemic_type", ""),
            ]
        )
    return buf.getvalue()


def _edges_csv(graph) -> str:
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["source", "target", "edge_type", "epistemic_type", "access_class",
         "confidence", "shared_count"]
    )
    for source, target, attrs in sorted(graph.edges(data=True), key=lambda e: (e[0], e[1], e[2].get("edge_type", ""))):
        writer.writerow(
            [
                source,
                target,
                attrs.get("edge_type", ""),
                attrs.get("epistemic_type", ""),
                attrs.get("access_class", ""),
                _stringify(attrs.get("confidence")),
                _stringify(attrs.get("shared_count")),
            ]
        )
    return buf.getvalue()


def _write_manifest_section(
    *,
    slug: str,
    run_id: str,
    section: str,
    payload: dict,
    root: Path | str | None = None,
) -> None:
    """Idempotent section-keyed manifest write (D5): overwrite only ``section``,
    never clobbering other stages' sections."""
    manifest_path = _run_dir(slug, run_id, root) / "manifest.json"
    if manifest_path.exists():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        document = {"run_id": run_id, "sections": {}}
    sections = document.setdefault("sections", {})
    sections[section] = payload
    _atomic_write(manifest_path, json.dumps(document, indent=2, sort_keys=True))


# =========================================================================== #
# BibTeX + RIS bibliographic exports (Build B chunk 10; ported from v1
# export.py). Writers read ONLY the _BIBLIO_FIELDS allowlist projection — no
# gated field (definition, quotes, claim text) can appear by construction.
# =========================================================================== #

#: The only node attrs the bibliographic writers may read (ids/title/authors/
#: year — all metadata-shareable, so D8 is trivially satisfied).
_BIBLIO_FIELDS = ("title", "authors", "year", "doi")


def _biblio_records(graph) -> list[dict]:
    """Allowlist-projected Work records from the gate-filtered view, sorted by
    ``work_id``. Title-less cited-only stubs are included, never dropped."""
    records: list[dict] = []
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Work":
            continue
        rec = {"work_id": node}
        for key in _BIBLIO_FIELDS:
            rec[key] = attrs.get(key)
        records.append(rec)
    records.sort(key=lambda r: str(r["work_id"]))
    return records


def _author_list(authors) -> list[str]:
    if not authors:
        return []
    if isinstance(authors, str):
        return [a.strip() for a in authors.replace(";", " and ").split(" and ") if a.strip()]
    return [str(a).strip() for a in authors if str(a).strip()]


def _cite_key(rec: dict) -> str:
    """Stable, unique-ish BibTeX cite key from a record's metadata."""
    authors = _author_list(rec.get("authors"))
    surname = ""
    if authors:
        # Shared comma-head/last-token extraction (display.extract_surname);
        # cite-key mode keeps the whole comma-head ("Van Der Berg, G" -> "Van Der Berg").
        surname = extract_surname(authors[0], whole_comma_head=True)
    surname = "".join(ch for ch in surname if ch.isalnum()) or "anon"
    year = rec.get("year") or "nd"
    uid = str(rec.get("work_id", ""))[:13]
    return f"{surname}{year}_{uid}".replace(" ", "")


def _bibtex_escape(value: str) -> str:
    return str(value).replace("{", "").replace("}", "")


def _bibtex_text(records: list[dict]) -> str:
    """``corpus.bib`` — one ``@article`` per Work (stubs included; a citation
    export must not silently drop cited works). ``note = {work_id=…}`` is the
    round-trip tag back to the truth layer."""
    chunks: list[str] = []
    for rec in records:
        key = _cite_key(rec)
        fields: list[tuple[str, str]] = []
        if rec.get("title"):
            fields.append(("title", _bibtex_escape(rec["title"])))
        authors = _author_list(rec.get("authors"))
        if authors:
            fields.append(("author", " and ".join(_bibtex_escape(a) for a in authors)))
        if rec.get("year") not in (None, ""):
            fields.append(("year", str(rec["year"])))
        if rec.get("doi"):
            fields.append(("doi", str(rec["doi"])))
        fields.append(("note", f"work_id={rec.get('work_id', '')}"))
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
        chunks.append(f"@article{{{key},\n{body}\n}}")
    return "\n\n".join(chunks) + ("\n" if chunks else "")


def _ris_text(records: list[dict]) -> str:
    """``corpus.ris`` — one ``TY  - JOUR`` … ``ER  -`` record per Work; the
    ``C1`` custom field carries the ``work_id`` round-trip tag."""
    lines: list[str] = []
    for rec in records:
        lines.append("TY  - JOUR")
        if rec.get("title"):
            lines.append(f"TI  - {rec['title']}")
        for a in _author_list(rec.get("authors")):
            lines.append(f"AU  - {a}")
        if rec.get("year") not in (None, ""):
            lines.append(f"PY  - {rec['year']}")
        if rec.get("doi"):
            lines.append(f"DO  - {rec['doi']}")
        lines.append(f"C1  - work_id={rec.get('work_id', '')}")
        lines.append("ER  - ")
        lines.append("")
    return "\n".join(lines)


def update_graph_manifest(
    *,
    slug: str,
    run_id: str,
    report: "SemanticBuildReport",
    root: Path | str | None = None,
) -> None:
    """Write the ``concepts`` manifest section: concept/edge counts +
    access-class summary + ``concept_mode``.

    Idempotent section-keyed write (D5) of the run manifest's ``concepts`` section
    — re-running the build overwrites only this stage's own section without
    clobbering other stages' sections.
    """
    _write_manifest_section(
        slug=slug,
        run_id=run_id,
        section="concepts",
        payload={
            "run_id": run_id,
            "concept_mode": report.concept_mode,
            "concepts": report.concepts_written,
            "aliases": report.aliases_written,
            "claim_concepts": report.claim_concepts_written,
            "discusses_edges": report.discusses_edges,
            "interpretive_edges": report.interpretive_edges,
            "co_occurs_edges": report.co_occurs_edges,
            "review_items_enqueued": report.review_items_enqueued,
            "access_class_summary": dict(Counter(report.access_class_summary)),
        },
        root=root,
    )
