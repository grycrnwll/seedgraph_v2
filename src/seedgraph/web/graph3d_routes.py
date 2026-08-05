"""Track 3 Stage B — local-session-only 3D graph workspace routes.

A single :class:`APIRouter` (``graph3d_router``) wired into ``create_app``. **Every**
route is gated by :func:`require_local_session` — these surfaces expose PRIVATE,
full-text-derived evidence (private concept definitions, ``llm_inferred`` edges,
claim/span provenance, and the source PDF/markdown bytes) and must never be
reachable except from a loopback client presenting the session cookie.

Routes:

* ``GET /ui/projects/{slug}/runs/{run_id}/graph?view=auto|3d|2d|table`` — the
  run-scoped graph view dispatch. ``auto`` resolves to ``3d`` iff the project has any
  concept rows (``SELECT EXISTS(SELECT 1 FROM concepts)``) else the table view; ``2d``
  is an alias for the server-rendered ``table`` view (``graph_table.html``); ``3d``
  renders the vendored two-plane WebGL canvas (``graph3d.html``) which fetches the
  JSON below client-side.
* ``GET /api/projects/{slug}/runs/{run_id}/graph3d.json`` — **local-session-only
  internal UI data (plan review #8)**: ``build_graph`` + ``nx.node_link_data`` +
  supplemental enrichment → :func:`build_3d_payload`. Computed in-memory, NEVER
  persisted to ``runs/``, NEVER exported, and distinct from the public ``graph.json``
  route — so private concept definitions and ``llm_inferred`` edges are present for
  local display only.
* ``GET /api/projects/{slug}/runs/{run_id}/nodes/{node_id}/provenance`` — the Work /
  Concept drawer. A Concept folds :func:`concept_provenance` by work, plus its
  ``concept_aliases`` and per-claim extraction run model/provider.
* ``GET /api/projects/{slug}/runs/{run_id}/edges/provenance`` — the edge drawer. A
  strut (paper↔concept) returns span-level claim provenance narrowed to that work; a
  concept↔concept edge is **structural only** (``evidence_mode="structural"``,
  ``shared_works`` self-join over ``claim_concepts``) because ``edge_spans`` has no
  producer — the response flags the absence so the UI never implies span evidence.
* ``GET /api/projects/{slug}/works/{work_id}/{pdf,markdown}`` — source-file serving.
  ``work_id`` is validated by ``work_source_files`` membership (never concatenated
  into a path); the blob is resolved through cache.db ``source_files`` /
  ``markdown_documents`` ``storage_uri`` (pdf additionally validates
  ``file_type == 'pdf'``) under the cache root; missing / non-member / traversal →
  bare ``404`` with NO path in the message.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sqlmodel import Session

from .. import paths
from ..acquisition.bridge import resolve_work_source
from ..cache import store as cache_store
from ..db.connection import open_cache_db
from ..errors import SeedgraphError
from ..graph.analyze import annotate_citation_communities
from ..project import service as project_service
from ..semantic.graph_build import build_graph, unread_work_map
from ..semantic.query import concept_provenance, concept_provenance_counts
from .graph3d import build_3d_payload
from .serve import require_local_session
from .ui import render

#: Path-free 404 detail — these routes NEVER leak a filesystem path / id into the body.
_NOT_FOUND = "not found"

graph3d_router = APIRouter(tags=["graph3d"])


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------

def _handle_or_404(slug: str) -> project_service.ProjectHandle:
    """Open ``slug`` or raise a path-free 404 (never echo the project dir path)."""
    try:
        paths.validate_slug(slug)
        return project_service.open_project(slug)
    except SeedgraphError as exc:  # ValidationError subclass — message carries the path
        raise HTTPException(status_code=404, detail=_NOT_FOUND) from exc


def _validate_run_id(run_id: str) -> None:
    """Reject a run id containing a path separator / ``..`` with a path-free 404."""
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)


def _project_conn(handle: project_service.ProjectHandle) -> sqlite3.Connection:
    conn = sqlite3.connect(str(handle.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _safe_cache_query(
    cache_conn: sqlite3.Connection, sql: str, params: tuple
) -> sqlite3.Row | None:
    """Run a cache.db read, degrading to ``None`` if the cache is empty/schemaless.

    A throwaway ``open_cache_db`` on a home with no cache.db yet yields a blank
    sqlite file (no ``source_files`` table); the lookup then raises
    ``OperationalError`` (a ``sqlite3.Error``) which we treat as "no such row".
    """
    try:
        return cache_conn.execute(sql, params).fetchone()
    except sqlite3.Error:
        return None


def document_link(
    session: Session,
    cache_conn: sqlite3.Connection,
    cache_root: Path,
    slug: str,
    work_id: str,
) -> dict:
    """Compose ``{has_pdf, has_markdown, pdf_url, markdown_url, access_class}`` for a work.

    Resolves the work's bridge row via :func:`resolve_work_source` (review #4), then
    reads cache.db ``source_files`` (``file_type`` / ``access_class``) and
    ``markdown_documents`` to decide what is servable. ``has_pdf`` requires
    ``file_type == 'pdf'``; the URLs point at the gated source-serving routes and are
    ``None`` when the corresponding blob isn't available. No bridge row → all-empty
    (never raises). ``cache_root`` is accepted for signature parity with the
    source-serving rule; resolution uses the cache connection directly.
    """
    res = resolve_work_source(session, work_id=work_id)
    if res is None:
        return {
            "has_pdf": False,
            "has_markdown": False,
            "pdf_url": None,
            "markdown_url": None,
            "access_class": None,
        }

    has_pdf = False
    access_class = None
    row = _safe_cache_query(
        cache_conn,
        "SELECT file_type, access_class FROM source_files WHERE source_file_id = ?",
        (res.source_file_id,),
    )
    if row is not None:
        access_class = row["access_class"]
        has_pdf = row["file_type"] == "pdf"

    has_markdown = False
    if res.markdown_id is not None:
        md = _safe_cache_query(
            cache_conn,
            "SELECT 1 FROM markdown_documents WHERE markdown_id = ?",
            (res.markdown_id,),
        )
        has_markdown = md is not None

    return {
        "has_pdf": has_pdf,
        "has_markdown": has_markdown,
        "pdf_url": f"/api/projects/{slug}/works/{work_id}/pdf" if has_pdf else None,
        "markdown_url": (
            f"/api/projects/{slug}/works/{work_id}/markdown" if has_markdown else None
        ),
        "access_class": access_class,
    }


def _node_link(graph) -> dict:
    """``nx.node_link_data`` with the ``edges='links'`` key (matches export.py)."""
    import networkx as nx

    try:
        return nx.node_link_data(graph, edges="links")
    except TypeError:  # pragma: no cover - older networkx
        return nx.node_link_data(graph)


# --------------------------------------------------------------------------
# graph3d.json payload (shared by the JSON route and the server-rendered table)
# --------------------------------------------------------------------------

def _payload_from_conn(
    conn: sqlite3.Connection,
    handle: project_service.ProjectHandle,
    slug: str,
    run_id: str,
) -> dict:
    """Build the in-memory 3D payload from a project connection.

    Owns the single ``build_graph`` call (NOT ``export_graph``/``_filtered_graph``),
    so private definitions + ``llm_inferred`` edges survive into the local-only view.
    Enriches links with ``confidence`` (from ``project_graph_edges`` /
    ``citation_edges``) and Work nodes with ``document_link`` + inclusion/seed before
    handing the already-materialized dicts to the pure :func:`build_3d_payload`. The
    result is returned to the caller and NEVER written to ``runs/``.
    """
    graph = build_graph(conn, run_id=run_id)
    # Build B chunk 3: citation-community analysis (same helper + unread SQL as
    # the export path) so the 3D view can color papers by community and mark
    # god-nodes / bridges. Deterministic, additive node/edge attrs.
    annotate_citation_communities(graph, unread=unread_work_map(conn))
    nodelink = _node_link(graph)
    nodes = nodelink.get("nodes", []) or []
    links = nodelink.get("links", []) or []

    concept_ids = [n["id"] for n in nodes if n.get("node_type") == "Concept"]
    prov_counts = concept_provenance_counts(conn, concept_ids)
    definitions = {
        cid: definition
        for cid, definition in conn.execute("SELECT concept_id, definition FROM concepts")
    }

    # --- link confidence enrichment (not carried on the NetworkX view) ----------
    cite_conf: dict[tuple, float] = {}
    for s, t, c in conn.execute(
        "SELECT source_work_id, target_work_id, MAX(confidence) "
        "FROM citation_edges GROUP BY source_work_id, target_work_id"
    ):
        cite_conf[(s, t)] = c
    pge_conf: dict[tuple, float] = {}
    for s, t, et, c in conn.execute(
        "SELECT source_node_id, target_node_id, edge_type, confidence FROM project_graph_edges"
    ):
        pge_conf[(s, t, et)] = c
    for link in links:
        s, t, et = link.get("source"), link.get("target"), link.get("edge_type")
        conf = pge_conf.get((s, t, et))
        if conf is None:
            conf = cite_conf.get((s, t))
        link["confidence"] = conf

    # --- Work-node document links (pdf/markdown affordances + access_class) ------
    work_ids = [n["id"] for n in nodes if n.get("node_type") == "Work"]
    doc_links: dict[str, dict] = {}
    if work_ids:
        meta_by_id = {d["work_id"]: d for d in project_service.list_documents(handle)}
        cache_conn = open_cache_db(root=handle.root)
        cache_root = cache_store.cache_root(handle.root)
        try:
            with Session(handle.engine) as session:
                for wid in work_ids:
                    dl = document_link(session, cache_conn, cache_root, slug, wid)
                    meta = meta_by_id.get(wid, {})
                    dl["inclusion_status"] = meta.get("inclusion_status")
                    dl["seed"] = bool(meta.get("is_seed", False))
                    doc_links[wid] = dl
        finally:
            cache_conn.close()

    payload = build_3d_payload(nodelink, prov_counts, doc_links)

    # Local-only enrichment (plan review #8): private concept definitions are present
    # for in-browser display — gate-exempt CONTENT behind an auth-gated ROUTE, never
    # in the public graph.json / export path.
    for node in payload["nodes"]:
        if node.get("type") == "concept":
            node["definition"] = definitions.get(node["id"])

    return payload


# --------------------------------------------------------------------------
# HTML view dispatch
# --------------------------------------------------------------------------

@graph3d_router.get(
    "/ui/projects/{slug}/runs/{run_id}/graph",
    response_class=HTMLResponse,
    dependencies=[Depends(require_local_session)],
)
def graph_view(request: Request, slug: str, run_id: str, view: str = "auto") -> HTMLResponse:
    """Run-scoped graph view: ``auto|3d|2d|table`` (2d aliases the table view)."""
    _validate_run_id(run_id)
    handle = _handle_or_404(slug)
    conn = _project_conn(handle)
    try:
        has_concepts = bool(
            conn.execute("SELECT EXISTS(SELECT 1 FROM concepts)").fetchone()[0]
        )
        if view == "auto":
            resolved = "3d" if has_concepts else "table"
        elif view == "2d":
            resolved = "table"
        elif view in ("3d", "table"):
            resolved = view
        else:
            raise HTTPException(status_code=400, detail="view must be auto|3d|2d|table")

        if resolved == "3d":
            return render(request, "graph3d.html", slug=slug, run_id=run_id)

        payload = _payload_from_conn(conn, handle, slug, run_id)
        return render(
            request,
            "graph_table.html",
            slug=slug,
            run_id=run_id,
            nodes=payload["nodes"],
            links=payload["links"],
            counts=payload["counts"],
            warnings=payload["warnings"],
        )
    finally:
        conn.close()


# --------------------------------------------------------------------------
# graph3d.json — local-session-only internal UI data (never persisted/exported)
# --------------------------------------------------------------------------

@graph3d_router.get(
    "/api/projects/{slug}/runs/{run_id}/graph3d.json",
    dependencies=[Depends(require_local_session)],
)
def get_graph3d_json(slug: str, run_id: str) -> JSONResponse:
    """In-memory 3D render payload for the run (computed, never written to disk)."""
    _validate_run_id(run_id)
    handle = _handle_or_404(slug)
    conn = _project_conn(handle)
    try:
        payload = _payload_from_conn(conn, handle, slug, run_id)
    finally:
        conn.close()
    return JSONResponse(payload)


# --------------------------------------------------------------------------
# node provenance drawer (Work or Concept)
# --------------------------------------------------------------------------

def _concept_drawer(conn: sqlite3.Connection, concept_id: str, crow: tuple) -> dict:
    concept = {
        "concept_id": crow[0],
        "normalized_label": crow[1],
        "canonical_label": crow[2],
        "concept_type": crow[3],
        "definition": crow[4],
        "paper_frequency": crow[5],
        "status": crow[6],
        "epistemic_type": crow[7],
        "access_class": crow[8],
    }
    rows = concept_provenance(conn, concept_id)

    works: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        wid = r["work_id"]
        if wid not in works:
            works[wid] = {
                "work_id": wid,
                "title": r["title"],
                "year": r["year"],
                "claims": {},
            }
            order.append(wid)
        claims = works[wid]["claims"]
        cid = r["claim_id"]
        if cid not in claims:
            claims[cid] = {
                "claim_id": cid,
                "claim_type": r["claim_type"],
                "claim_subtype": r["claim_subtype"],
                "claim_text": r["claim_text"],
                "extraction_run_id": r["extraction_run_id"],
                "access_class": r["access_class"],
                "spans": [],
            }
        if r["span_id"] is not None:
            claims[cid]["spans"].append(
                {
                    "span_id": r["span_id"],
                    "exact_quote": r["exact_quote"],
                    "start_char": r["start_char"],
                    "end_char": r["end_char"],
                    "page_start": r["page_start"],
                    "page_end": r["page_end"],
                    "heading_path": r["heading_path"],
                    "heading_text": r["heading_text"],
                    "access_class": r["span_access_class"],
                }
            )

    # per-claim extraction run model/provider/access_mode.
    run_ids = sorted({r["extraction_run_id"] for r in rows if r["extraction_run_id"]})
    runs_meta: dict[str, dict] = {}
    if run_ids:
        placeholders = ",".join("?" for _ in run_ids)
        for rid, model_name, provider, access_mode in conn.execute(
            "SELECT extraction_run_id, model_name, provider, access_mode "
            f"FROM extraction_runs WHERE extraction_run_id IN ({placeholders})",
            run_ids,
        ):
            runs_meta[rid] = {
                "model_name": model_name,
                "provider": provider,
                "access_mode": access_mode,
            }

    aliases = [
        {"alias_label": a, "fold_reason": f, "epistemic_type": e}
        for a, f, e in conn.execute(
            "SELECT alias_label, fold_reason, epistemic_type FROM concept_aliases "
            "WHERE concept_id = ? ORDER BY alias_label",
            (concept_id,),
        )
    ]

    works_list = []
    for wid in order:
        w = works[wid]
        w["claims"] = list(w["claims"].values())
        works_list.append(w)

    return {
        "node_type": "Concept",
        "concept": concept,
        "works": works_list,
        "aliases": aliases,
        "extraction_runs": runs_meta,
    }


def _work_drawer(
    conn: sqlite3.Connection,
    handle: project_service.ProjectHandle,
    slug: str,
    work_id: str,
    wrow: tuple,
) -> dict:
    identifiers = {
        "doi": wrow[3],
        "arxiv_id": wrow[4],
        "openalex_id": wrow[5],
        "semantic_scholar_id": wrow[6],
        "ssrn_id": wrow[7],
    }
    membership = conn.execute(
        "SELECT inclusion_status, is_seed FROM project_documents WHERE work_id = ?",
        (work_id,),
    ).fetchone()
    cache_conn = open_cache_db(root=handle.root)
    cache_root = cache_store.cache_root(handle.root)
    try:
        with Session(handle.engine) as session:
            link = document_link(session, cache_conn, cache_root, slug, work_id)
    finally:
        cache_conn.close()
    return {
        "node_type": "Work",
        "work": {
            "work_id": work_id,
            "title": wrow[1],
            "year": wrow[2],
            "identifiers": identifiers,
            "inclusion_status": membership[0] if membership else None,
            "is_seed": bool(membership[1]) if membership else False,
        },
        "document": link,
    }


@graph3d_router.get(
    "/api/projects/{slug}/runs/{run_id}/nodes/{node_id:path}/provenance",
    dependencies=[Depends(require_local_session)],
)
def get_node_provenance(slug: str, run_id: str, node_id: str) -> dict:
    """Provenance drawer for one node — a Concept or a Work (404 otherwise)."""
    _validate_run_id(run_id)
    handle = _handle_or_404(slug)
    conn = _project_conn(handle)
    try:
        crow = conn.execute(
            "SELECT concept_id, normalized_label, canonical_label, concept_type, "
            "definition, paper_frequency, status, epistemic_type, access_class "
            "FROM concepts WHERE concept_id = ?",
            (node_id,),
        ).fetchone()
        if crow is not None:
            return _concept_drawer(conn, node_id, crow)
        wrow = conn.execute(
            "SELECT work_id, canonical_title, year, doi, arxiv_id, openalex_id, "
            "semantic_scholar_id, ssrn_id FROM works WHERE work_id = ?",
            (node_id,),
        ).fetchone()
        if wrow is not None:
            return _work_drawer(conn, handle, slug, node_id, wrow)
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# edge provenance drawer (strut / concept-concept structural / citation)
# --------------------------------------------------------------------------

def _group_claims(rows: list[dict]) -> list[dict]:
    claims: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        cid = r["claim_id"]
        if cid not in claims:
            claims[cid] = {
                "claim_id": cid,
                "claim_type": r["claim_type"],
                "claim_subtype": r["claim_subtype"],
                "claim_text": r["claim_text"],
                "extraction_run_id": r["extraction_run_id"],
                "access_class": r["access_class"],
                "spans": [],
            }
            order.append(cid)
        if r["span_id"] is not None:
            claims[cid]["spans"].append(
                {
                    "span_id": r["span_id"],
                    "exact_quote": r["exact_quote"],
                    "start_char": r["start_char"],
                    "end_char": r["end_char"],
                    "page_start": r["page_start"],
                    "page_end": r["page_end"],
                    "heading_path": r["heading_path"],
                    "heading_text": r["heading_text"],
                    "access_class": r["span_access_class"],
                }
            )
    return [claims[cid] for cid in order]


@graph3d_router.get(
    "/api/projects/{slug}/runs/{run_id}/edges/provenance",
    dependencies=[Depends(require_local_session)],
)
def get_edge_provenance(
    slug: str,
    run_id: str,
    source: str,
    target: str,
    edge_type: str | None = None,
) -> dict:
    """Edge drawer keyed by endpoint type.

    * **strut** (one concept + one work): span-level claim provenance via
      :func:`concept_provenance` narrowed to that work (``evidence_mode='spans'``).
    * **concept↔concept**: STRUCTURAL ONLY (``evidence_mode='structural'``) — a
      ``shared_works`` self-join over ``claim_concepts``. ``edge_spans`` has no
      producer, so the response explicitly carries no span evidence and says so.
    * **paper↔paper**: the citation edge's provenance + confidence
      (``evidence_mode='citation'``).
    """
    _validate_run_id(run_id)
    handle = _handle_or_404(slug)
    conn = _project_conn(handle)
    try:
        s_concept = (
            conn.execute(
                "SELECT 1 FROM concepts WHERE concept_id = ?", (source,)
            ).fetchone()
            is not None
        )
        t_concept = (
            conn.execute(
                "SELECT 1 FROM concepts WHERE concept_id = ?", (target,)
            ).fetchone()
            is not None
        )

        if s_concept and t_concept:
            shared = [
                {"work_id": w, "title": title, "year": year}
                for w, title, year in conn.execute(
                    "SELECT DISTINCT a.work_id, w.canonical_title, w.year "
                    "FROM claim_concepts a "
                    "JOIN claim_concepts b ON a.work_id = b.work_id "
                    "JOIN works w ON w.work_id = a.work_id "
                    "WHERE a.concept_id = ? AND b.concept_id = ? "
                    "ORDER BY a.work_id",
                    (source, target),
                )
            ]
            return {
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "evidence_mode": "structural",
                "shared_works": shared,
                "has_span_evidence": False,
                "note": (
                    "concept-concept edges carry no span-level evidence (edge_spans has "
                    "no producer); shared_works is a structural co-occurrence over "
                    "claim_concepts"
                ),
            }

        if s_concept != t_concept:
            concept_id = source if s_concept else target
            work_id = target if s_concept else source
            rows = concept_provenance(conn, concept_id, work_id=work_id)
            return {
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "evidence_mode": "spans",
                "concept_id": concept_id,
                "work_id": work_id,
                "claims": _group_claims(rows),
            }

        crow = conn.execute(
            "SELECT provenance, confidence, edge_type FROM citation_edges "
            "WHERE source_work_id = ? AND target_work_id = ? "
            "ORDER BY confidence DESC LIMIT 1",
            (source, target),
        ).fetchone()
        if crow is None:
            raise HTTPException(status_code=404, detail=_NOT_FOUND)
        return {
            "source": source,
            "target": target,
            "edge_type": crow[2],
            "evidence_mode": "citation",
            "provenance": crow[0],
            "confidence": crow[1],
        }
    finally:
        conn.close()


# --------------------------------------------------------------------------
# source-file serving (pdf / markdown) — membership + cache metadata, no path-guess
# --------------------------------------------------------------------------

def _serve_source(slug: str, work_id: str, *, kind: str) -> FileResponse:
    """Serve a work's source PDF / markdown blob via cache metadata (review #9).

    ``work_id`` is validated by ``work_source_files`` membership; the blob is located
    through cache.db ``storage_uri`` (pdf additionally requires ``file_type='pdf'``),
    resolved under the cache root, and existence-checked before a ``FileResponse``.
    Non-member / non-pdf / missing blob / traversal all collapse to a bare path-free
    404 — the message never reveals a filesystem path.
    """
    handle = _handle_or_404(slug)
    with Session(handle.engine) as session:
        res = resolve_work_source(session, work_id=work_id)
    if res is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)

    cache_conn = open_cache_db(root=handle.root)
    cache_root = cache_store.cache_root(handle.root)
    try:
        if kind == "pdf":
            row = _safe_cache_query(
                cache_conn,
                "SELECT storage_uri, file_type FROM source_files WHERE source_file_id = ?",
                (res.source_file_id,),
            )
            if row is None or row["file_type"] != "pdf" or not row["storage_uri"]:
                raise HTTPException(status_code=404, detail=_NOT_FOUND)
            storage_uri = row["storage_uri"]
            media_type = "application/pdf"
        else:  # markdown
            if res.markdown_id is None:
                raise HTTPException(status_code=404, detail=_NOT_FOUND)
            row = _safe_cache_query(
                cache_conn,
                "SELECT storage_uri FROM markdown_documents WHERE markdown_id = ?",
                (res.markdown_id,),
            )
            if row is None or not row["storage_uri"]:
                raise HTTPException(status_code=404, detail=_NOT_FOUND)
            storage_uri = row["storage_uri"]
            media_type = "text/markdown"
    finally:
        cache_conn.close()

    resolved = cache_store.resolve_uri(storage_uri, handle.root).resolve()
    root = cache_root.resolve()
    # Containment guard (defense in depth): a stored uri must stay under the cache root.
    if root != resolved and root not in resolved.parents:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return FileResponse(str(resolved), media_type=media_type)


@graph3d_router.get(
    "/api/projects/{slug}/works/{work_id}/pdf",
    dependencies=[Depends(require_local_session)],
)
def get_work_pdf(slug: str, work_id: str) -> FileResponse:
    """Serve the work's source PDF blob (membership + ``file_type='pdf'`` validated)."""
    return _serve_source(slug, work_id, kind="pdf")


@graph3d_router.get(
    "/api/projects/{slug}/works/{work_id}/markdown",
    dependencies=[Depends(require_local_session)],
)
def get_work_markdown(slug: str, work_id: str) -> FileResponse:
    """Serve the work's converted markdown blob (membership validated via the bridge)."""
    return _serve_source(slug, work_id, kind="markdown")
