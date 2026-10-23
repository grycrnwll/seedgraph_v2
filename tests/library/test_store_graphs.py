"""Context graphs: create/open/list, rename, membership, lenses, in-graph citations (§3)."""

from __future__ import annotations

import sqlite3

import pytest

from seedgraph.errors import ValidationError
from seedgraph.library import graphs, store

from .test_store import NOW, add_paper, add_source_file, attach_file


@pytest.fixture
def lib(tmp_path):
    library = store.open_library(tmp_path)
    with library.write() as conn:
        for pid in ("lp_a", "lp_b", "lp_c", "lp_cited"):
            add_paper(conn, pid)
        for n, pid in enumerate(("lp_a", "lp_b", "lp_c"), start=1):
            attach_file(conn, pid, add_source_file(conn, n))
    # reset the flag so tests can observe graph writes setting it
    with store.transaction(library.conn) as c:
        c.execute("UPDATE library_meta SET value = '0' WHERE key = 'dirty'")
    yield library
    library.close()


def test_create_graph_lays_out_folder_and_meta(lib):
    g = graphs.create_graph(lib, "did", display_name="Invented DiD reading")
    try:
        assert g.path == lib.graphs_root / "did" / "graph.db"
        assert g.path.is_file()
        info = graphs.graph_info(g)
        assert info.slug == "did"
        assert info.display_name == "Invented DiD reading"
        assert info.origin == "library"
        tables = {r[0] for r in g.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"schema_migrations", "graph_meta", "graph_papers", "graph_lenses"} <= tables
        assert not tables & {"literatures", "literature_members", "traversal_records",
                             "traversal_events"}
    finally:
        g.close()
    assert store.is_dirty(lib.conn)


def test_display_name_defaults_to_slug_and_create_refuses_existing(lib):
    graphs.create_graph(lib, "land-value").close()
    with graphs.open_graph(lib, "land-value") as g:
        assert graphs.graph_info(g).display_name == "land-value"
    with pytest.raises(graphs.GraphError):
        graphs.create_graph(lib, "land-value")


def test_bad_slug_and_missing_graph_are_refused(lib):
    with pytest.raises(ValidationError):
        graphs.create_graph(lib, "Bad Slug")
    with pytest.raises(graphs.GraphError):
        graphs.open_graph(lib, "nope")


def test_list_graphs_sorted_by_slug(lib):
    graphs.create_graph(lib, "zeta", display_name="Z").close()
    graphs.create_graph(lib, "alpha", display_name="A").close()
    assert [(i.slug, i.display_name) for i in graphs.list_graphs(lib)] == [
        ("alpha", "A"), ("zeta", "Z")]


def test_list_graphs_on_a_fresh_library_is_empty(lib):
    assert graphs.list_graphs(lib) == []


def test_rename_changes_display_name_not_slug(lib):
    graphs.create_graph(lib, "did", display_name="Old").close()
    graphs.rename_graph(lib, "did", "New invented name")
    with graphs.open_graph(lib, "did") as g:
        info = graphs.graph_info(g)
        assert (info.slug, info.display_name) == ("did", "New invented name")
    assert [i.slug for i in graphs.list_graphs(lib)] == ["did"]


def test_membership_roundtrip(lib):
    with graphs.create_graph(lib, "did") as g:
        graphs.set_membership(g, "lp_a", "member", is_seed=True, reason="seed_document")
        graphs.set_membership(g, "lp_b", "member")
        graphs.set_membership(g, "lp_cited", "frontier")
        graphs.set_membership(g, "lp_c", "excluded", reason="user_excluded")
        rows = {m.paper_id: (m.status, m.is_seed) for m in graphs.membership(g)}
        assert rows == {"lp_a": ("member", True), "lp_b": ("member", False),
                        "lp_cited": ("frontier", False), "lp_c": ("excluded", False)}
        assert [m.paper_id for m in graphs.membership(g, status="member")] == ["lp_a", "lp_b"]
        # changing status updates the row
        graphs.set_membership(g, "lp_c", "member")
        assert {m.paper_id for m in graphs.membership(g, status="member")} == {
            "lp_a", "lp_b", "lp_c"}


def test_cited_only_paper_cannot_be_a_member(lib):
    with graphs.create_graph(lib, "did") as g:
        with pytest.raises(graphs.NotHeldError):
            graphs.set_membership(g, "lp_cited", "member")
        assert graphs.membership(g) == []


def test_unknown_paper_and_bad_status_are_refused(lib):
    with graphs.create_graph(lib, "did") as g:
        with pytest.raises(graphs.GraphError):
            graphs.set_membership(g, "lp_nowhere", "frontier")
        with pytest.raises(graphs.GraphError):
            graphs.set_membership(g, "lp_a", "included")


def test_graph_writes_set_library_dirty_except_in_migration_mode(tmp_path, lib):
    with graphs.create_graph(lib, "did") as g:
        with store.transaction(lib.conn) as c:
            c.execute("UPDATE library_meta SET value = '0' WHERE key = 'dirty'")
        graphs.set_membership(g, "lp_a", "member")
    assert store.is_dirty(lib.conn)

    staging = store.open_library_at(tmp_path / "staging" / "run1" / "library", migration=True)
    try:
        with staging.write() as conn:
            add_paper(conn, "lp_x")
            attach_file(conn, "lp_x", add_source_file(conn, 9))
        with graphs.create_graph(staging, "did", display_name="D", origin="migration") as g:
            assert g.path == tmp_path / "staging" / "run1" / "graphs" / "did" / "graph.db"
            graphs.set_membership(g, "lp_x", "member", origin="v2:proj")
            graphs.choose_lens(g, "widget_lens_v1", origin="migration")
            assert graphs.membership(g)[0].origin == "v2:proj"
        graphs.rename_graph(staging, "did", "Renamed")
        assert not store.is_dirty(staging.conn)
    finally:
        staging.close()


def test_lens_choice(lib):
    with graphs.create_graph(lib, "did") as g:
        graphs.choose_lens(g, "widget_lens_v1")
        graphs.choose_lens(g, "gadget_lens_v2")
        graphs.choose_lens(g, "widget_lens_v1")  # idempotent
        assert graphs.chosen_lenses(g) == ["gadget_lens_v2", "widget_lens_v1"]


def _link(conn, src, dst, prov="parsed_bibliography"):
    conn.execute(
        "INSERT INTO citation_links (source_paper_id, target_paper_id, provenance, confidence, "
        "created_at) VALUES (?, ?, ?, 1.0, ?)", (src, dst, prov, NOW))


def test_in_graph_citations_through_attach(lib):
    with lib.write() as conn:
        _link(conn, "lp_a", "lp_b")
        _link(conn, "lp_a", "lp_b", "provider_reference")
        _link(conn, "lp_b", "lp_cited")
        _link(conn, "lp_a", "lp_c")  # lp_c excluded: never in view
    with graphs.create_graph(lib, "did") as g:
        graphs.set_membership(g, "lp_a", "member")
        graphs.set_membership(g, "lp_b", "member")
        graphs.set_membership(g, "lp_cited", "frontier")
        graphs.set_membership(g, "lp_c", "excluded")
        # library connection still open: the read-only ATTACH must coexist with it
        members = [(c.source_paper_id, c.target_paper_id, c.provenance)
                   for c in graphs.graph_citations(g)]
        assert members == [("lp_a", "lp_b", "parsed_bibliography"),
                           ("lp_a", "lp_b", "provider_reference")]
        with_frontier = [(c.source_paper_id, c.target_paper_id)
                         for c in graphs.graph_citations(g, include_frontier=True)]
        assert with_frontier == [("lp_a", "lp_b"), ("lp_a", "lp_b"), ("lp_b", "lp_cited")]
        # a citation decision made later in the library is seen at once
        with lib.write() as conn:
            conn.execute("DELETE FROM citation_links WHERE provenance = 'provider_reference'")
        assert len(graphs.graph_citations(g)) == 1
        # the attached library is read-only
        with graphs.attached_library(g) as conn:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("DELETE FROM lib.citation_links")
        assert len(graphs.graph_citations(g)) == 1


def test_remove_membership_returns_removed_row_and_sets_dirty(lib):
    with graphs.create_graph(lib, "did") as g:
        graphs.set_membership(g, "lp_a", "member", is_seed=True)
        with store.transaction(lib.conn) as c:
            c.execute("UPDATE library_meta SET value = '0' WHERE key = 'dirty'")
        removed = graphs.remove_membership(g, "lp_a")
        assert removed is not None and removed.is_seed is True
        assert graphs.membership(g) == []
        assert graphs.remove_membership(g, "lp_a") is None
        assert store.is_held(lib.conn, "lp_a")  # library untouched
    assert store.is_dirty(lib.conn)


def test_graph_write_inside_an_open_library_write(lib):
    with graphs.create_graph(lib, "did") as g:
        with store.transaction(lib.conn) as c:
            c.execute("UPDATE library_meta SET value = '0' WHERE key = 'dirty'")
        with lib.write() as conn:
            add_paper(conn, "lp_new")
            attach_file(conn, "lp_new", add_source_file(conn, 7))
            graphs.set_membership(g, "lp_new", "member")
        assert [m.paper_id for m in graphs.membership(g)] == ["lp_new"]
    assert store.is_dirty(lib.conn)
