"""Offline provider-edge projection (decision D3 — link-only, no vivification).

This is the heart of phase_2 and is **pure, offline, deterministic**: it issues
NO network calls and constructs NO provider transport (phase_2 is a projection,
not a walk). Corpus growth — the outbound ``referenced_works`` network walk, the
per-generation frontier, the upsert of metadata-only target works, and the
``project_documents(metadata_only, citation_walk)`` membership rows — is owned
SOLELY by phase_5b (binding r2-1 §4). This phase only READS what phase_5b already
materialized.

:func:`project_provider_edges` walks the existing ``works`` rows in ``project.db``
and, for each, reads that work's cached ``referenced_works`` list from the
read-only-ATTACHed ``provider_cache`` (cache.db, written by phase_5b) under
phase_5b's pinned request-key grammar ``referenced_works:<source canonical id>``
(phase_5b §6.5). For every referenced target that ALREADY EXISTS as a ``works``
row it writes one ``provider_reference`` edge (via :func:`citation.edges.write_edge`,
confidence ``1.0``) under the supplied ``run_id``.

Decision D3 (link-only): a referenced target with NO ``works`` row is NEVER
created here (phase_5b is the sole materializer). It is instead appended to the
returned ``unresolved_target`` diagnostic list — surfaced, never silently created
and never silently dropped (proven by the unresolved_target test). Because edges
are emitted only between existing works, ``citation_edges.target_work_id`` is a
real FK by construction and edges never dangle.
"""

from __future__ import annotations

import json
import sqlite3

from ..cache.provider_cache import (
    cache_is_fresh,
    canonical_request_id,
    key_referenced_works,
)
from ..project.identity import normalize_id
from .edges import PROVIDER_REFERENCE, write_edge

# Columns read off a ``works`` row to (a) rebuild its §6.5 cache key and (b) seed
# the strong-id -> work_id resolver. ``(id_type, works-column)`` pairs mirror
# project.identity.ID_TYPE_TO_WORK_COLUMN.
_WORK_COLS = (
    "work_id",
    "canonical_title",
    "doi",
    "arxiv_id",
    "openalex_id",
    "semantic_scholar_id",
    "ssrn_id",
)
_ID_TYPE_TO_COLUMN: tuple[tuple[str, str], ...] = (
    ("doi", "doi"),
    ("openalex", "openalex_id"),
    ("arxiv", "arxiv_id"),
    ("s2", "semantic_scholar_id"),
    ("ssrn", "ssrn_id"),
)
# Provider-reference target dicts may carry an id under any of these aliases.
_ID_TYPE_TO_REF_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("doi", ("doi",)),
    ("openalex", ("openalex_id", "openalex")),
    ("arxiv", ("arxiv_id", "arxiv")),
    ("s2", ("semantic_scholar_id", "s2_id", "s2")),
    ("ssrn", ("ssrn_id", "ssrn")),
)


def _work_record(work: dict) -> dict:
    """Rebuild the minimal id+title record ``canonical_request_id`` reads, so the
    §6.5 ``referenced_works:src=<canonical id>`` key matches the one phase_5b wrote."""
    record: dict = {}
    for column in ("openalex_id", "doi", "arxiv_id", "semantic_scholar_id", "ssrn_id"):
        value = work.get(column)
        if value:
            record[column] = value
    if work.get("canonical_title"):
        record["title"] = work["canonical_title"]
    return record


def _build_resolver(conn: sqlite3.Connection, works: list[dict]) -> dict[tuple[str, str], str]:
    """Map every strong ``(id_type, normalized value)`` an existing work carries to
    its ``work_id`` — from both the denormalized ``works`` id columns and the
    ``identifiers`` join table (so a target resolves by any id it shares)."""
    resolver: dict[tuple[str, str], str] = {}
    for work in works:
        for id_type, column in _ID_TYPE_TO_COLUMN:
            value = work.get(column)
            if not value:
                continue
            norm = normalize_id(id_type, value)
            if norm:
                resolver.setdefault((id_type, norm), work["work_id"])
    cursor = conn.execute("SELECT id_type, id_value, work_id FROM identifiers")
    for id_type, id_value, work_id in cursor.fetchall():
        norm = normalize_id(id_type, id_value)
        if norm:
            resolver.setdefault((id_type, norm), work_id)
    return resolver


def _resolve_target(ref: dict, resolver: dict[tuple[str, str], str]) -> str | None:
    """Resolve a provider-reference target dict to an EXISTING ``work_id`` by any
    strong id it carries, or ``None`` (no works row -> unresolved_target; D3)."""
    for id_type, keys in _ID_TYPE_TO_REF_KEYS:
        for key in keys:
            value = ref.get(key)
            if not value:
                continue
            norm = normalize_id(id_type, value)
            if norm is not None and (id_type, norm) in resolver:
                return resolver[(id_type, norm)]
    return None


def _cached_referenced_works(cache_conn: sqlite3.Connection, request_key: str) -> list:
    """Read the freshest FRESH cached ``referenced_works`` list for ``request_key``
    across providers (the §6.5 key is provider-independent). Read-only; no network."""
    try:
        rows = cache_conn.execute(
            "SELECT response, fetched_at, ttl_seconds FROM provider_cache "
            "WHERE request_key = ? ORDER BY fetched_at DESC",
            (request_key,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    for response, fetched_at, ttl_seconds in rows:
        if response is None:
            continue
        if not cache_is_fresh(fetched_at=fetched_at, ttl_seconds=int(ttl_seconds)):
            continue
        try:
            data = json.loads(response)
        except (ValueError, TypeError):
            continue
        if isinstance(data, list):
            return data
    return []


def project_provider_edges(
    conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection,
    *,
    run_id: str,
) -> dict:
    """Project ``provider_reference`` edges offline from ``provider_cache``.

    For each existing ``works`` row in ``conn`` (project.db), read its cached
    ``referenced_works:<source canonical id>`` list from ``cache_conn`` / the
    ATTACHed ``provider_cache`` (phase_5b's pinned key grammar) and write a
    ``provider_reference`` edge (confidence ``1.0``) to every target that already
    exists as a ``works`` row. A referenced target with NO ``works`` row is
    appended to ``unresolved_targets`` and never created (decision D3 — phase_5b
    is the sole materializer); it produces no edge (fail-closed).

    No network, no provider transport, no frontier ranking, no upsert — purely a
    read of ``provider_cache`` + writes to ``citation_edges`` (offline by
    construction; criteria 2/3/6). ``run_id`` MUST be supplied by the caller,
    allocated at the start of the invocation (must-fix #3), so every written edge
    satisfies the NOT-NULL ``citation_edges.run_id`` — including standalone
    ``cite project``. ``provider_cache`` is read-only here; this phase writes
    nothing to cache.db.

    Returns a diagnostic dict::

        {"edges": <int>, "unresolved_targets": [<target canonical id>, ...]}

    where ``edges`` is the number of edges written and ``unresolved_targets`` lists
    the referenced targets absent as ``works`` rows (the D3 diagnostic).
    """
    cursor = conn.execute(f"SELECT {', '.join(_WORK_COLS)} FROM works")
    works = [dict(zip(_WORK_COLS, raw)) for raw in cursor.fetchall()]
    resolver = _build_resolver(conn, works)

    edges_written = 0
    unresolved_targets: list[str] = []
    seen_unresolved: set[str] = set()

    for work in works:
        request_key = key_referenced_works(_work_record(work))
        refs = _cached_referenced_works(cache_conn, request_key)
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            target_work_id = _resolve_target(ref, resolver)
            if target_work_id is None:
                # D3: a referenced target with NO works row is NEVER created here
                # (phase_5b is the sole materializer) — surfaced, never dropped.
                target_id = canonical_request_id(ref)
                if target_id not in seen_unresolved:
                    seen_unresolved.add(target_id)
                    unresolved_targets.append(target_id)
                continue
            before = conn.total_changes
            write_edge(
                conn,
                source=work["work_id"],
                target=target_work_id,
                provenance=PROVIDER_REFERENCE,
                confidence=1.0,
                run_id=run_id,
            )
            if conn.total_changes > before:
                edges_written += 1

    conn.commit()
    return {"edges": edges_written, "unresolved_targets": unresolved_targets}
