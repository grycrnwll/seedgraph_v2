"""Apply a migration plan into a staging folder (design §6.1 steps 6-7).

Creates a fresh ``staging/<run_id>/library/library.db`` (migration mode, so writes
leave ``dirty`` alone) through the numbered library migrations, inserts every plan
table in primary-key order inside one transaction, rebuilds the FTS tables, raises
the attention items through :mod:`seedgraph.library.attention` (``created_at`` = the
snapshot time), creates each ``graphs/<slug>/graph.db`` with v2-derived timestamps,
copies v2's provider cache rows into ``provider_cache.db``, and runs
``PRAGMA integrity_check`` and ``foreign_key_check`` on every store.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from seedgraph.library import attention, graphs
from seedgraph.library.store import Library, open_library_at, open_provider_cache, record_migration_origin

from .plan import COLS, MigrationPlan

#: Insert order (foreign keys point backwards in this list).
INSERT_ORDER: tuple[str, ...] = (
    "papers", "identifiers", "source_files", "conversion_runs", "markdown_documents", "cache_events",
    "paper_files", "document_sections", "evidence_spans", "extraction_runs", "structured_notes",
    "extracted_claims", "claim_spans", "reference_entries", "citation_links", "citation_review_decisions",
    "duplicate_decisions", "paper_merge_members", "lens_definitions", "lens_outputs", "concept_decisions",
    "conversion_queue", "v2_id_map",
)
#: FTS table -> (content table, FTS columns, the content columns in that order).
FTS_REBUILD: dict[str, tuple[str, str, str]] = {
    "span_fts": ("evidence_spans", "quote_text, span_id, markdown_id, section_id",
                 "exact_quote, span_id, markdown_id, section_id"),
    "claim_fts": ("extracted_claims", "claim_id, normalized_label, claim_text",
                  "claim_id, normalized_label, claim_text"),
    "note_fts": ("structured_notes", "note_id, note_text", "note_id, note_text"),
}


class ApplyError(RuntimeError):
    """The staging build failed its integrity checks."""


@dataclass
class ApplyResult:
    library_dir: Path
    graphs_dir: Path
    counts: dict[str, int] = field(default_factory=dict)
    graph_counts: dict[str, dict[str, int]] = field(default_factory=dict)


def _code_version() -> str:
    try:
        from importlib.metadata import version

        return version("seedgraph")
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def _check(conn: sqlite3.Connection, what: str) -> None:
    ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if ok != "ok":
        raise ApplyError(f"{what}: integrity_check failed: {ok}")
    bad = conn.execute("PRAGMA foreign_key_check").fetchall()
    if bad:
        raise ApplyError(f"{what}: foreign_key_check failed: {[tuple(r) for r in bad[:5]]}")


def _insert_all(lib: Library, plan: MigrationPlan) -> None:
    with lib.write() as conn:
        for table in INSERT_ORDER:
            cols = COLS[table]
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
            conn.executemany(sql, [tuple(r[c] for c in cols) for r in plan.rows(table)])
        for fts, (content, fts_cols, src_cols) in FTS_REBUILD.items():
            id_col = src_cols.split(",")[0] if fts != "span_fts" else "span_id"
            conn.execute(f"INSERT INTO {fts} ({fts_cols}) SELECT {src_cols} FROM {content} ORDER BY {id_col}")


def apply_plan(plan: MigrationPlan, staging_dir: Path, *, run_id: str, v2_home: Path,
               fingerprint: str | None = None) -> ApplyResult:
    """Build the staging library and graphs for ``plan`` under ``staging_dir``."""
    lib_dir = Path(staging_dir) / "library"
    if (lib_dir / "library.db").exists():
        raise ApplyError(f"staging library already exists: {lib_dir}")
    lib = open_library_at(lib_dir, migration=True)
    result = ApplyResult(library_dir=lib_dir, graphs_dir=lib.graphs_root)
    try:
        _insert_all(lib, plan)
        for spec in plan.attention:
            attention.raise_item(lib, spec.kind, dict(spec.subject), paper_id=spec.paper_id, detail=spec.detail,
                                 hides_notes=spec.hides_notes, created_at=plan.taken_at, origin="migration")
        record_migration_origin(lib, run_id=run_id, v2_home=str(Path(v2_home).resolve()))
        with lib.write() as conn:
            conn.execute(
                "INSERT INTO migration_runs (run_id, started_at, finished_at, v2_fingerprint, dump_sha256, "
                "code_version) VALUES (?, ?, ?, ?, NULL, ?)",
                (run_id, plan.taken_at, datetime.now(timezone.utc).isoformat(), fingerprint, _code_version()))
        for g in plan.graphs:
            with graphs.create_graph(lib, g.slug, display_name=g.display_name, origin=g.origin,
                                     created_at=g.created_at) as gr:
                for r in g.papers:
                    graphs.set_membership(gr, r["paper_id"], r["status"], is_seed=bool(r["is_seed"]),
                                          reason=r["reason"], origin=r["origin"], added_at=r["added_at"])
                for lens in g.lenses:
                    graphs.choose_lens(gr, lens["lens_id"], origin=lens["origin"], chosen_at=lens["chosen_at"])
                _check(gr.conn, f"graph {g.slug}")
                result.graph_counts[g.slug] = {
                    s: gr.conn.execute("SELECT COUNT(*) FROM graph_papers WHERE status = ?", (s,)).fetchone()[0]
                    for s in graphs.STATUSES}
        pc = open_provider_cache(lib_dir)
        try:
            with pc:
                pc.executemany(
                    "INSERT INTO provider_cache (provider_cache_id, provider, request_key, response, fetched_at, "
                    "ttl_seconds) VALUES (?, ?, ?, ?, ?, ?)",
                    [(r["provider_cache_id"], r["provider"], r["request_key"], r["response"], r["fetched_at"],
                      r["ttl_seconds"]) for r in plan.provider_cache])
            _check(pc, "provider_cache.db")
        finally:
            pc.close()
        _check(lib.conn, "library.db")
        for fts, (content, _, _) in FTS_REBUILD.items():
            a = lib.conn.execute(f"SELECT COUNT(*) FROM {fts}").fetchone()[0]
            b = lib.conn.execute(f"SELECT COUNT(*) FROM {content}").fetchone()[0]
            if a != b:
                raise ApplyError(f"{fts} has {a} rows, {content} has {b}")
        result.counts = {t: lib.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                         for t in (*INSERT_ORDER, "attention_items")}
        lib.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        lib.close()
    return result


def record_dump_sha(lib_dir: Path, run_id: str, sha: str) -> None:
    """Store the dump sha in ``migration_runs`` (a table the dump excludes)."""
    lib = open_library_at(lib_dir, migration=True)
    try:
        with lib.write() as conn:
            conn.execute("UPDATE migration_runs SET dump_sha256 = ? WHERE run_id = ?", (sha, run_id))
        lib.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        lib.close()
