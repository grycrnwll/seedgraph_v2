"""Phase 2 — Citation Graph MVP — acceptance tests (§11).

Every plan §11 test is implemented and runs FULLY OFFLINE: ``provider_cache`` is
pre-seeded from recorded JSON-shaped fixtures via phase_5b's ``cache_put`` under the
PINNED §6.5 ``referenced_works:src=<canonical id>`` key grammar; this phase issues
ZERO network calls and constructs NO provider transport (asserted directly). The
module-level imports double as a whole-phase import-cleanliness check.

Decisions exercised: D3 (link-only, unresolved_target diagnostic, no vivification),
D6 (migration == ORM parity), D8 / decision 5 (shareable provider_reference edges),
must-fix #3 (run_id allocated at invocation start).
"""

from __future__ import annotations

import sqlite3

from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import sqlite as sqlite_dialect
from typer.testing import CliRunner

from seedgraph.cache.db import init_cache_db
from seedgraph.cache.provider_cache import (
    cache_get,
    cache_put,
    canonical_request_id,
    key_referenced_works,
)
from seedgraph.citation.edges import (
    PROVENANCE_AUTHORITY,
    authoritative_edges,
    is_shareable_edge,
    write_edge,
)
from seedgraph.citation.project_edges import project_provider_edges
from seedgraph.cli import app
from seedgraph.db.connection import open_cache_db
from seedgraph.db.migrations import run_migrations
from seedgraph.db.project_models import CitationEdge, ReferenceEntry
from seedgraph.graph.build import build_citation_graph
from seedgraph.graph.export import export_graph, load_graph
from seedgraph.project import service
from seedgraph.run import ensure_run

runner = CliRunner()


# --------------------------------------------------------------------------
# Offline fixtures + helpers
# --------------------------------------------------------------------------

def _project_conn(handle) -> sqlite3.Connection:
    """Raw project.db connection with fail-closed FKs (foreign_keys=ON)."""
    conn = sqlite3.connect(str(handle.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _seed_refs(source_record: dict, ref_list: list) -> None:
    """Pre-seed ``provider_cache`` with one work's ``referenced_works`` list under
    phase_5b's PINNED §6.5 key — exactly the writer side phase_5b uses (no network)."""
    init_cache_db(None)
    conn = open_cache_db(None)
    try:
        cache_put(
            conn,
            provider="openalex",
            request_key=key_referenced_works(source_record),
            response=ref_list,
        )
    finally:
        conn.close()


def _wid_by_openalex(conn: sqlite3.Connection) -> dict[str, str]:
    """``openalex_id -> work_id`` for the existing works."""
    return {
        row[1]: row[0]
        for row in conn.execute("SELECT work_id, openalex_id FROM works").fetchall()
        if row[1]
    }


def _works_count(handle) -> int:
    conn = sqlite3.connect(str(handle.db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM works").fetchone()[0]
    finally:
        conn.close()


# --- projection (offline, pure) -------------------------------------------

def test_projection_builds_provider_reference_edges_between_existing_works():
    """A pre-seeded ``provider_cache`` ``referenced_works`` fixture plus the
    matching ``works`` rows yields exactly the expected ``provider_reference``
    edges (confidence 1.0, edge_type 'cites') between the existing works —
    projected offline from cache, never opening a PDF (criteria 1/2/6)."""
    h = service.create_project("proj1")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")
    service.add_work(h, ids={"openalex": "W_C"}, title="C")
    _seed_refs({"openalex_id": "W_A"}, [{"openalex_id": "W_B"}, {"openalex_id": "W_C"}])
    _seed_refs({"openalex_id": "W_B"}, [{"openalex_id": "W_C"}])

    conn = _project_conn(h)
    cache_conn = open_cache_db(None)
    try:
        result = project_provider_edges(conn, cache_conn, run_id="run-1")
        rows = conn.execute(
            "SELECT source_work_id, target_work_id, edge_type, provenance, confidence "
            "FROM citation_edges"
        ).fetchall()
        wid = _wid_by_openalex(conn)
    finally:
        conn.close()
        cache_conn.close()

    assert result["edges"] == 3
    assert result["unresolved_targets"] == []
    assert len(rows) == 3
    for _s, _t, edge_type, provenance, confidence in rows:
        assert edge_type == "cites"
        assert provenance == "provider_reference"
        assert confidence == 1.0
    pairs = {(s, t) for s, t, *_ in rows}
    # criterion 2: a known citing->cited pair appears (built without opening a PDF).
    assert (wid["W_A"], wid["W_B"]) in pairs
    assert (wid["W_A"], wid["W_C"]) in pairs
    assert (wid["W_B"], wid["W_C"]) in pairs


def test_projection_constructs_no_network_transport(monkeypatch):
    """``project_provider_edges`` issues NO network calls and constructs NO
    provider transport — phase_2 is a pure offline projection over
    ``provider_cache`` (binding r2-1 §4; criterion 6). Asserted by sabotaging the
    provider/transport layer and verifying it is never instantiated."""
    import httpx

    from seedgraph.providers import base as provider_base

    def _boom(*args, **kwargs):
        raise AssertionError("phase_2 must not construct any network transport")

    monkeypatch.setattr(httpx, "AsyncClient", _boom)
    monkeypatch.setattr(httpx, "Client", _boom)
    monkeypatch.setattr(provider_base, "ProviderChain", _boom)

    h = service.create_project("nonet")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")
    _seed_refs({"openalex_id": "W_A"}, [{"openalex_id": "W_B"}])

    conn = _project_conn(h)
    cache_conn = open_cache_db(None)
    try:
        result = project_provider_edges(conn, cache_conn, run_id="run-nonet")
    finally:
        conn.close()
        cache_conn.close()
    # Reached here -> no transport was constructed; the edge was projected offline.
    assert result["edges"] == 1


def test_projection_absent_target_yields_no_edge():
    """A ``referenced_works`` entry whose target has NO ``works`` row produces NO
    edge (fail-closed) — phase_2 never auto-vivifies a target (decision D3;
    phase_5b is the sole materializer)."""
    h = service.create_project("absent")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    _seed_refs({"openalex_id": "W_A"}, [{"openalex_id": "W_GHOST"}])

    conn = _project_conn(h)
    cache_conn = open_cache_db(None)
    try:
        before = conn.execute("SELECT COUNT(*) FROM works").fetchone()[0]
        result = project_provider_edges(conn, cache_conn, run_id="r")
        after = conn.execute("SELECT COUNT(*) FROM works").fetchone()[0]
        n_edges = conn.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0]
    finally:
        conn.close()
        cache_conn.close()

    assert result["edges"] == 0
    assert n_edges == 0
    assert before == after  # no vivification — the absent target is never created


# --- unresolved_target diagnostic (D3) ------------------------------------

def test_unresolved_target_diagnostic_surfaced():
    """A referenced target absent as a ``works`` row produces no edge AND is
    recorded in the ``unresolved_targets`` list returned by
    ``project_provider_edges`` — proving absent targets are neither silently
    created (decision D3) nor silently dropped (surfaced count/list)."""
    h = service.create_project("unres")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")
    _seed_refs(
        {"openalex_id": "W_A"},
        [{"openalex_id": "W_B"}, {"openalex_id": "W_GHOST"}],
    )

    conn = _project_conn(h)
    cache_conn = open_cache_db(None)
    try:
        result = project_provider_edges(conn, cache_conn, run_id="r")
    finally:
        conn.close()
        cache_conn.close()

    assert result["edges"] == 1  # only A->B (the resolvable target)
    assert result["unresolved_targets"] == ["openalex=w_ghost"]


# --- run_id allocation (must-fix #3) --------------------------------------

def test_standalone_cite_project_allocates_not_null_run_id():
    """Standalone ``cite project`` writes edges carrying a valid, NOT-NULL
    ``run_id`` allocated at the start of the invocation (must-fix #3) — the
    ``citation_edges.run_id`` NOT NULL constraint is always satisfied."""
    h = service.create_project("standalone")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")
    _seed_refs({"openalex_id": "W_A"}, [{"openalex_id": "W_B"}])

    result = runner.invoke(app, ["cite", "project", "standalone"])
    assert result.exit_code == 0, result.output

    conn = _project_conn(h)
    try:
        rows = conn.execute("SELECT run_id FROM citation_edges").fetchall()
        nulls = conn.execute(
            "SELECT COUNT(*) FROM citation_edges WHERE run_id IS NULL"
        ).fetchone()[0]
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0][0] and rows[0][0].startswith("run-")
    assert nulls == 0


def test_two_projections_distinct_run_ids_both_preserved():
    """Two successive projections mint two distinct ``run_id`` values, and both
    runs' edges are preserved (append-only; prior run never clobbered)."""
    h = service.create_project("tworuns")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")
    _seed_refs({"openalex_id": "W_A"}, [{"openalex_id": "W_B"}])

    rid1 = ensure_run("tworuns", root=h.root)
    rid2 = ensure_run("tworuns", root=h.root)
    assert rid1 != rid2

    conn = _project_conn(h)
    cache_conn = open_cache_db(None)
    try:
        project_provider_edges(conn, cache_conn, run_id=rid1)
        project_provider_edges(conn, cache_conn, run_id=rid2)
        runs = {r[0] for r in conn.execute("SELECT DISTINCT run_id FROM citation_edges")}
        n1 = conn.execute(
            "SELECT COUNT(*) FROM citation_edges WHERE run_id = ?", (rid1,)
        ).fetchone()[0]
        n2 = conn.execute(
            "SELECT COUNT(*) FROM citation_edges WHERE run_id = ?", (rid2,)
        ).fetchone()[0]
    finally:
        conn.close()
        cache_conn.close()

    assert runs == {rid1, rid2}
    assert n1 == 1 and n2 == 1  # both runs preserved (append-only)


# --- no dangling edges (decision 59 / D3) ---------------------------------

def test_no_dangling_edges_every_target_resolves():
    """Every ``citation_edges.target_work_id`` selects a real ``works`` row
    (no dangling edges) — guaranteed because edges are emitted only between
    existing works phase_5b materialized (criterion 3)."""
    h = service.create_project("nodangle")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")
    service.add_work(h, ids={"openalex": "W_C"}, title="C")
    _seed_refs(
        {"openalex_id": "W_A"},
        [{"openalex_id": "W_B"}, {"openalex_id": "W_C"}, {"openalex_id": "W_GHOST"}],
    )

    conn = _project_conn(h)
    cache_conn = open_cache_db(None)
    try:
        project_provider_edges(conn, cache_conn, run_id="r")
        total = conn.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0]
        dangling = conn.execute(
            "SELECT COUNT(*) FROM citation_edges e "
            "LEFT JOIN works w ON e.target_work_id = w.work_id "
            "WHERE w.work_id IS NULL"
        ).fetchone()[0]
        fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        conn.close()
        cache_conn.close()

    assert total == 2  # A->B, A->C; the ghost target is dropped (no edge)
    assert dangling == 0
    assert fk_violations == []


# --- key-grammar cross-phase contract -------------------------------------

def test_referenced_works_key_grammar_contract():
    """The projector recovers each citing work's reference list using phase_5b's
    pinned ``referenced_works:<source canonical id>`` request-key grammar: a
    fixture written by phase_5b's ``cache_put`` is read back unchanged
    (cross-phase contract; phase_5b §6.5)."""
    from sqlmodel import Session, select

    from seedgraph.db.project_models import Work

    h = service.create_project("keygrammar")
    service.add_work(h, ids={"doi": "10.1/x", "openalex": "W123"}, title="Paper")
    service.add_work(h, ids={"openalex": "W9"}, title="Target")

    with Session(h.engine) as session:
        src = session.exec(select(Work).where(Work.openalex_id == "W123")).one()
        # canonical_request_id REUSED for the contract (id-preference openalex>doi).
        assert canonical_request_id(src) == "openalex=w123"
        key = key_referenced_works(src)
    assert key == "referenced_works:src=openalex=w123"

    init_cache_db(None)
    cache_conn = open_cache_db(None)
    try:
        cache_put(cache_conn, provider="openalex", request_key=key, response=[{"openalex_id": "W9"}])
        # phase_2 reconstructs the SAME key and reads the SAME list back unchanged.
        assert cache_get(
            cache_conn, provider="openalex", request_key="referenced_works:src=openalex=w123"
        ) == [{"openalex_id": "W9"}]
    finally:
        cache_conn.close()

    # End-to-end: the projector recovers the ref list via the pinned key -> 1 edge.
    conn = _project_conn(h)
    cache_conn = open_cache_db(None)
    try:
        result = project_provider_edges(conn, cache_conn, run_id="r")
    finally:
        conn.close()
        cache_conn.close()
    assert result["edges"] == 1


# --- dedup & authority seam -----------------------------------------------

def test_duplicate_edge_collapses_via_unique():
    """A duplicate ``(source, target, edge_type, provenance, run_id)`` collapses
    via the table UNIQUE constraint (union storage + dedup) — one row persists."""
    h = service.create_project("dedup")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")

    conn = _project_conn(h)
    try:
        wid = _wid_by_openalex(conn)
        for _ in range(2):
            write_edge(
                conn, source=wid["W_A"], target=wid["W_B"],
                provenance="provider_reference", confidence=1.0, run_id="r",
            )
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0]
    finally:
        conn.close()
    assert n == 1


def test_authoritative_edges_one_per_source_target_per_run():
    """``authoritative_edges(conn, run_id)`` returns exactly one edge per
    ``(source, target)`` for the run (the highest-authority provenance)."""
    h = service.create_project("authone")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")
    service.add_work(h, ids={"openalex": "W_C"}, title="C")

    conn = _project_conn(h)
    try:
        wid = _wid_by_openalex(conn)
        for _ in range(2):  # duplicate (A,B) collapses via UNIQUE
            write_edge(conn, source=wid["W_A"], target=wid["W_B"],
                       provenance="provider_reference", confidence=1.0, run_id="r")
        write_edge(conn, source=wid["W_A"], target=wid["W_C"],
                   provenance="provider_reference", confidence=1.0, run_id="r")
        conn.commit()
        auth = authoritative_edges(conn, "r")
    finally:
        conn.close()

    pairs = {(e["source_work_id"], e["target_work_id"]) for e in auth}
    assert len(auth) == 2
    assert pairs == {(wid["W_A"], wid["W_B"]), (wid["W_A"], wid["W_C"])}


def test_higher_provenance_outranks_provider_reference():
    """An injected ``manual_override`` row outranks a ``provider_reference`` row for
    the same ``(source, target)`` per ``PROVENANCE_AUTHORITY`` (forward-seam check
    for phase_3b's second provenance)."""
    h = service.create_project("authrank")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")

    conn = _project_conn(h)
    try:
        wid = _wid_by_openalex(conn)
        write_edge(conn, source=wid["W_A"], target=wid["W_B"],
                   provenance="provider_reference", confidence=1.0, run_id="r")
        write_edge(conn, source=wid["W_A"], target=wid["W_B"],
                   provenance="manual_override", confidence=1.0, run_id="r")
        conn.commit()
        auth = authoritative_edges(conn, "r")
    finally:
        conn.close()

    assert len(auth) == 1
    assert auth[0]["provenance"] == "manual_override"
    assert (
        PROVENANCE_AUTHORITY["manual_override"]
        > PROVENANCE_AUTHORITY["parsed_bibliography"]
        > PROVENANCE_AUTHORITY["provider_reference"]
    )


# --- shareable seam (decision 5 / D8) -------------------------------------

def test_is_shareable_edge_provider_reference_true():
    """``is_shareable_edge('provider_reference') is True`` and any other
    provenance is False (default-deny export gate for edges)."""
    assert is_shareable_edge("provider_reference") is True
    assert is_shareable_edge("parsed_bibliography") is False
    assert is_shareable_edge("manual_override") is False
    assert is_shareable_edge("anything_else") is False


def test_graph_json_export_contains_only_shareable_edges(tmp_path):
    """The exported ``graph.json`` contains only shareable edges — a non-shareable
    authoritative edge (here a ``manual_override``) is filtered out on the export
    path while a ``provider_reference`` edge survives."""
    h = service.create_project("shareexport")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")
    service.add_work(h, ids={"openalex": "W_C"}, title="C")

    conn = _project_conn(h)
    try:
        wid = _wid_by_openalex(conn)
        write_edge(conn, source=wid["W_A"], target=wid["W_B"],
                   provenance="provider_reference", confidence=1.0, run_id="r")
        write_edge(conn, source=wid["W_A"], target=wid["W_C"],
                   provenance="manual_override", confidence=1.0, run_id="r")
        conn.commit()
        g = build_citation_graph(conn, run_id="r", closed_world=True)
    finally:
        conn.close()

    assert g.number_of_edges() == 2  # both authoritative edges are in the view
    out = tmp_path / "out"
    export_graph(g, out, manifest={"citation": {"run_id": "r"}})
    reloaded = load_graph(out / "graph.json")

    assert reloaded.number_of_edges() == 1  # the manual_override edge is filtered out
    assert (wid["W_A"], wid["W_B"]) in reloaded.edges()
    assert (wid["W_A"], wid["W_C"]) not in reloaded.edges()
    for _s, _t, data in reloaded.edges(data=True):
        assert is_shareable_edge(data["provenance"])


# --- closed / open world (decision 13) ------------------------------------

def _world_fixture(slug):
    h = service.create_project(slug)
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")
    service.add_work(h, ids={"openalex": "W_X"}, title="X", inclusion_status="metadata_only")
    conn = _project_conn(h)
    wid = _wid_by_openalex(conn)
    write_edge(conn, source=wid["W_A"], target=wid["W_B"],
               provenance="provider_reference", confidence=1.0, run_id="r")
    write_edge(conn, source=wid["W_A"], target=wid["W_X"],
               provenance="provider_reference", confidence=1.0, run_id="r")
    conn.commit()
    return h, conn, wid


def test_closed_world_drops_out_of_corpus_targets():
    """``build_citation_graph(closed_world=True)`` (default) restricts nodes to the
    included corpus and drops out-of-corpus metadata-only stub targets."""
    h, conn, wid = _world_fixture("closedworld")
    try:
        g = build_citation_graph(conn, run_id="r", closed_world=True)
    finally:
        conn.close()
    assert set(g.nodes()) == {wid["W_A"], wid["W_B"]}
    assert wid["W_X"] not in g.nodes()
    assert (wid["W_A"], wid["W_X"]) not in g.edges()
    assert (wid["W_A"], wid["W_B"]) in g.edges()


def test_open_world_keeps_metadata_only_targets():
    """``build_citation_graph(closed_world=False)`` (``--open-world``) retains the
    metadata-only stub targets as nodes."""
    h, conn, wid = _world_fixture("openworld")
    try:
        g = build_citation_graph(conn, run_id="r", closed_world=False)
    finally:
        conn.close()
    assert wid["W_X"] in g.nodes()
    assert (wid["W_A"], wid["W_X"]) in g.edges()
    assert g.nodes[wid["W_X"]]["inclusion_status"] == "metadata_only"


# --- export round-trip (criterion 4) --------------------------------------

def test_graph_json_export_round_trip_counts(tmp_path):
    """``graph.json`` reloads into NetworkX with node/edge counts matching
    ``authoritative_edges(run_id)`` for that run (criterion 4)."""
    h = service.create_project("roundtrip")
    service.add_work(h, ids={"openalex": "W_A"}, title="A")
    service.add_work(h, ids={"openalex": "W_B"}, title="B")
    service.add_work(h, ids={"openalex": "W_C"}, title="C")
    _seed_refs({"openalex_id": "W_A"}, [{"openalex_id": "W_B"}, {"openalex_id": "W_C"}])
    _seed_refs({"openalex_id": "W_B"}, [{"openalex_id": "W_C"}])

    conn = _project_conn(h)
    cache_conn = open_cache_db(None)
    try:
        project_provider_edges(conn, cache_conn, run_id="r")
        g = build_citation_graph(conn, run_id="r", closed_world=True)
        auth = authoritative_edges(conn, "r")
    finally:
        conn.close()
        cache_conn.close()

    out = tmp_path / "out"
    export_graph(g, out, manifest={"citation": {"run_id": "r"}})
    reloaded = load_graph(out / "graph.json")

    assert reloaded.number_of_nodes() == g.number_of_nodes()
    assert reloaded.number_of_edges() == g.number_of_edges()
    assert g.number_of_edges() == len(auth)  # all in-corpus -> none filtered
    assert reloaded.number_of_edges() == 3


# --- migration == ORM parity (D6) -----------------------------------------

def _affinity(type_str: str) -> str:
    t = type_str.upper()
    if "INT" in t:
        return "INTEGER"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "TEXT"


def _assert_table_parity(insp, table_name, model):
    orm_cols = {c.name: c for c in model.__table__.columns}
    mig_cols = {c["name"]: c for c in insp.get_columns(table_name)}
    assert set(orm_cols) == set(mig_cols), f"{table_name}: column set drift"

    mig_pk = set(insp.get_pk_constraint(table_name)["constrained_columns"])
    orm_pk = {c.name for c in model.__table__.columns if c.primary_key}
    assert mig_pk == orm_pk, f"{table_name}: PK drift"

    orm_dialect = sqlite_dialect.dialect()
    for name, orm_col in orm_cols.items():
        mig_col = mig_cols[name]
        orm_aff = _affinity(str(orm_col.type.compile(dialect=orm_dialect)))
        mig_aff = _affinity(str(mig_col["type"]))
        assert orm_aff == mig_aff, f"{table_name}.{name}: type {orm_aff} != {mig_aff}"
        if name not in mig_pk:
            assert orm_col.nullable == mig_col["nullable"], (
                f"{table_name}.{name}: nullability drift"
            )

    mig_ix = {tuple(ix["column_names"]) for ix in insp.get_indexes(table_name)}
    orm_ix = {tuple(c.name for c in ix.columns) for ix in model.__table__.indexes}
    assert mig_ix == orm_ix, f"{table_name}: index column-sets drift ({mig_ix} != {orm_ix})"


def test_migration_parity_orm_equals_migrated_schema(tmp_path):
    """A fresh ``project.db`` built by applying ``db/schema/project/*.sql`` has
    ``reference_entries`` + ``citation_edges`` whose columns / indexes / CHECKs /
    UNIQUE / FKs equal the ``ReferenceEntry`` / ``CitationEdge`` ORM metadata —
    schema authored by the numbered ``.sql``, not ``create_all`` (decision D6)."""
    db = tmp_path / "parity.db"
    conn = sqlite3.connect(str(db))
    run_migrations(conn, "project")  # the migrate step authors schema (D6)
    conn.close()

    insp = inspect(create_engine(f"sqlite:///{db.as_posix()}"))
    assert "reference_entries" in insp.get_table_names()
    assert "citation_edges" in insp.get_table_names()

    _assert_table_parity(insp, "reference_entries", ReferenceEntry)
    _assert_table_parity(insp, "citation_edges", CitationEdge)

    # citation_edges UNIQUE(source, target, edge_type, provenance, run_id) dedup key.
    uq = [set(u["column_names"]) for u in insp.get_unique_constraints("citation_edges")]
    assert {
        "source_work_id", "target_work_id", "edge_type", "provenance", "run_id",
    } in uq

    # FKs: citation_edges -> works (x2) + reference_entries; reference_entries -> works (x2).
    ce_fks = insp.get_foreign_keys("citation_edges")
    referred = {fk["referred_table"] for fk in ce_fks}
    assert "works" in referred and "reference_entries" in referred
    re_fks = insp.get_foreign_keys("reference_entries")
    assert all(fk["referred_table"] == "works" for fk in re_fks)
    assert len(re_fks) == 2  # citing_work_id + resolved_work_id

    # provenance + edge_type CHECKs present on the migrated table.
    checks = " ".join(ck["sqltext"] for ck in insp.get_check_constraints("citation_edges"))
    assert "provenance" in checks


# --- acceptance milestone (doc 10 §9) -------------------------------------

def test_acceptance_cite_table_matches_known_relationships():
    """After phase_5b has grown the corpus and populated ``provider_cache``,
    ``cite run`` then ``cite table <slug>`` prints the included->included citation
    matrix projected purely from cached provider metadata (no network, no PDFs),
    matching the 5-10 paper corpus's known shared-citation relationships
    (doc 10 §9/§16; criteria 1/2)."""
    h = service.create_project("accept")
    service.add_work(h, ids={"openalex": "W_A", "doi": "10.a"}, title="Paper A")
    service.add_work(h, ids={"openalex": "W_B", "doi": "10.b"}, title="Paper B")
    service.add_work(h, ids={"openalex": "W_C", "doi": "10.c"}, title="Paper C")
    # Known relationships: A cites B and C; B cites C.
    _seed_refs({"openalex_id": "W_A"}, [{"openalex_id": "W_B"}, {"openalex_id": "W_C"}])
    _seed_refs({"openalex_id": "W_B"}, [{"openalex_id": "W_C"}])

    before = _works_count(h)

    run_result = runner.invoke(app, ["cite", "run", "accept"])
    assert run_result.exit_code == 0, run_result.output

    table_result = runner.invoke(app, ["cite", "table", "accept"])
    assert table_result.exit_code == 0, table_result.output
    out = table_result.output

    # included->included matrix matches the known shared-citation relationships.
    assert "Paper A\tPaper B" in out
    assert "Paper A\tPaper C" in out
    assert "Paper B\tPaper C" in out
    # no spurious reverse edges (outbound referenced_works only).
    assert "Paper B\tPaper A" not in out
    assert "Paper C\tPaper A" not in out

    # ZERO works created by the projection (no vivification; D3).
    assert _works_count(h) == before == 3

    # graph.json was written and reloads into NetworkX with matching counts.
    runs_dir = h.db_path.parent / "runs"
    run_id = max(
        (p for p in runs_dir.iterdir() if p.is_dir()),
        key=lambda p: (p.stat().st_mtime, p.name),
    ).name
    graph = load_graph(runs_dir / run_id / "graph.json")
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 3
