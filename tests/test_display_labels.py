"""Build B chunk 9 — never-blank display labels (`display.derive_label`).

The full priority chain (title → "Surname et al. (Year)" → visibly-placeholder
id → "untitled <id8>"), plus the adoption points: graph_build stamps `label`,
the 3D payload + nodes.csv show a placeholder for title-less works (never a raw
UUID, never blank), and the stored canonical_title is never touched.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from seedgraph.display import derive_label


# --------------------------------------------------------------------------
# the priority chain
# --------------------------------------------------------------------------

def test_title_wins_verbatim():
    assert derive_label(title="A Real Title", authors=["X"], doi="10.1/x") == "A Real Title"
    # whitespace-only title is NOT a title.
    assert derive_label(title="   ", doi="10.1/x") == "doi:10.1/x"


def test_authors_surname_et_al_year():
    assert derive_label(title=None, authors=["Callaway, Brantly"], year=2021) == "Callaway (2021)"
    assert derive_label(authors=["Brantly Callaway", "P. Sant'Anna"], year=2021) == "Callaway et al. (2021)"
    assert derive_label(authors=["Brantly Callaway"]) == "Callaway"


def test_visibly_placeholder_ids_never_title_shaped():
    assert derive_label(doi="10.1234/abc") == "doi:10.1234/abc"
    assert derive_label(openalex_id="W123456") == "OpenAlex W123456"
    assert derive_label(arxiv_id="2401.01234") == "arXiv:2401.01234"
    # priority within ids: doi > openalex > arxiv.
    assert derive_label(doi="10.1/x", openalex_id="W1", arxiv_id="a") == "doi:10.1/x"


def test_uid_residue_and_final_fallback():
    assert derive_label(work_id="0123456789abcdef") == "untitled 01234567"
    assert derive_label() == "untitled"
    assert derive_label(title=None, authors=[], work_id="") == "untitled"


# --------------------------------------------------------------------------
# adoption: graph_build label attr -> 3D payload + nodes.csv, DB untouched
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed(tmp_path) -> sqlite3.Connection:
    import seedgraph.db.migrations as migrations

    conn = sqlite3.connect(str(tmp_path / "project.db"))
    conn.execute("PRAGMA foreign_keys=ON")
    migrations.run_migrations(conn, "project")
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, year, authors, created_at) "
        "VALUES ('work_titled', 'Titled Paper', 2020, '[\"A. Author\"]', ?)",
        (_now(),),
    )
    # a title-less stub with only a doi (the placeholder case).
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, doi, created_at) "
        "VALUES ('work_stub', NULL, '10.9/stub', ?)",
        (_now(),),
    )
    # a fully-bare work: uid-residue placeholder.
    conn.execute(
        "INSERT INTO works (work_id, created_at) VALUES ('work_bare12345', ?)",
        (_now(),),
    )
    conn.commit()
    return conn


def test_graph_build_stamps_labels(tmp_path):
    from seedgraph.semantic.graph_build import build_graph

    conn = _seed(tmp_path)
    g = build_graph(conn, run_id="r1")
    assert g.nodes["work_titled"]["label"] == "Titled Paper"
    assert g.nodes["work_stub"]["label"] == "doi:10.9/stub"
    assert g.nodes["work_bare12345"]["label"] == "untitled work_bar"
    # authors parsed from the JSON column (list, not raw string).
    assert g.nodes["work_titled"]["authors"] == ["A. Author"]
    # stored canonical_title untouched.
    assert conn.execute(
        "SELECT canonical_title FROM works WHERE work_id='work_stub'"
    ).fetchone()[0] is None
    conn.close()


def test_titleless_work_placeholder_in_3d_payload_and_csv(tmp_path):
    from seedgraph.semantic import export
    from seedgraph.semantic.graph_build import build_graph
    from seedgraph.web.graph3d import build_3d_payload

    conn = _seed(tmp_path)
    import networkx as nx  # noqa: F401 — via export

    g = build_graph(conn, run_id="r1")
    payload = build_3d_payload(nx_node_link(g))
    names = {n["id"]: n["name"] for n in payload["nodes"]}
    assert names["work_titled"] == "Titled Paper"
    assert names["work_stub"] == "doi:10.9/stub"          # placeholder, not a UUID
    assert names["work_bare12345"] == "untitled work_bar"  # never the raw id alone
    assert all(name.strip() for name in names.values())    # never blank

    written = export.export_graph(conn, slug="x", run_id="r1", fmt="csv",
                                  root=tmp_path / "h")
    nodes_csv = [p for p in written if p.name == "nodes.csv"][0].read_text()
    rows = {line.split(",")[0]: line for line in nodes_csv.splitlines()[1:] if line}
    assert "doi:10.9/stub" in rows["work_stub"]
    assert "untitled work_bar" in rows["work_bare12345"]
    conn.close()


def nx_node_link(graph) -> dict:
    import networkx as nx

    try:
        return nx.node_link_data(graph, edges="links")
    except TypeError:  # pragma: no cover - older networkx
        return nx.node_link_data(graph)


def test_list_documents_carries_label(tmp_path, monkeypatch):
    from seedgraph.project import service

    h = service.create_project("labels")
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, doi, created_at) "
        "VALUES ('work_stub', NULL, '10.9/stub', ?)",
        (_now(),),
    )
    conn.execute(
        "INSERT INTO project_documents (work_id, inclusion_status, is_seed, "
        "created_at, updated_at) VALUES ('work_stub', 'included', 0, ?, ?)",
        (_now(), _now()),
    )
    conn.commit()
    conn.close()
    rows = service.list_documents(h)
    assert rows[0]["title"] is None            # verbatim — never fabricated
    assert rows[0]["label"] == "doi:10.9/stub"  # never-blank display label
