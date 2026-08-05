"""Build B chunk 10 — BibTeX/RIS export rider (`graph export --format bibtex|ris`).

One entry per Work including title-less cited-only stubs; `note = {work_id=…}` /
`C1  - work_id=…` round-trip tags; BibTeX brace escaping; RIS framing; and the
allowlist-by-construction guarantee that no gated field can appear.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import seedgraph.db.migrations as migrations
from seedgraph.semantic import export


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


SECRET = "a private full-text-derived definition sentence"


def _seed(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(tmp_path / "project.db"))
    conn.execute("PRAGMA foreign_keys=ON")
    migrations.run_migrations(conn, "project")
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, year, authors, doi, created_at) "
        "VALUES ('work_a', 'Braces {and} More', 2020, "
        "'[\"Callaway, Brantly\", \"Pedro Sant''Anna\"]', '10.1/a', ?)",
        (_now(),),
    )
    # a title-less cited-only stub — must still be emitted.
    conn.execute(
        "INSERT INTO works (work_id, created_at) VALUES ('work_stub', ?)", (_now(),)
    )
    conn.execute(
        "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, "
        "provenance, confidence, run_id, created_at) VALUES "
        "('work_a','work_stub','cites','provider_reference',1.0,'r1',?)",
        (_now(),),
    )
    # a PRIVATE concept with a full-text-derived definition: must never leak.
    conn.execute(
        "INSERT INTO concepts (concept_id, normalized_label, canonical_label, "
        "concept_type, definition, paper_frequency, status, epistemic_type, "
        "access_class, created_at, updated_at) VALUES ('concept::p','p','P','method',"
        "?,1,'auto','deterministic','user_supplied_private',?,?)",
        (SECRET, _now(), _now()),
    )
    conn.commit()
    return conn


def _export(conn, tmp_path, fmt):
    written = export.export_graph(conn, slug="x", run_id="r1", fmt=fmt,
                                  root=tmp_path / f"h_{fmt}")
    return [p for p in written if p.suffix in (".bib", ".ris")][0]


def test_bibtex_one_entry_per_work_with_roundtrip_tag(tmp_path):
    conn = _seed(tmp_path)
    bib = _export(conn, tmp_path, "bibtex").read_text(encoding="utf-8")
    assert bib.count("@article{") == 2          # stub included, concept excluded
    assert "note = {work_id=work_a}" in bib
    assert "note = {work_id=work_stub}" in bib  # title-less stub still round-trips
    # brace escaping: the raw braces are stripped from the emitted title.
    assert "Braces and More" in bib
    assert "{and}" not in bib
    # authors joined with " and "; year + doi emitted.
    assert "author = {Callaway, Brantly and Pedro Sant'Anna}" in bib
    assert "year = {2020}" in bib
    assert "doi = {10.1/a}" in bib
    # cite key: surname + year + uid prefix.
    assert "@article{Callaway2020_work_a" in bib
    conn.close()


def test_ris_framing_and_roundtrip_tag(tmp_path):
    conn = _seed(tmp_path)
    ris = _export(conn, tmp_path, "ris").read_text(encoding="utf-8")
    assert ris.count("TY  - JOUR") == 2
    assert ris.count("ER  - ") == 2
    assert "C1  - work_id=work_a" in ris
    assert "C1  - work_id=work_stub" in ris
    assert "TI  - Braces {and} More" in ris  # RIS needs no brace escaping
    assert "AU  - Callaway, Brantly" in ris
    assert "PY  - 2020" in ris
    assert "DO  - 10.1/a" in ris
    # each record starts TY and ends ER (framing order).
    first = ris.split("ER  - ")[0]
    assert first.startswith("TY  - JOUR")
    conn.close()


def test_no_gated_field_can_appear_by_construction(tmp_path):
    """The writers read only the _BIBLIO_FIELDS allowlist: a graph whose Work
    node is polluted with gated content still exports clean files."""
    import networkx as nx

    g = nx.DiGraph()
    g.add_node(
        "work_x",
        node_type="Work",
        title="Clean Title",
        authors=["A"],
        year=2021,
        doi="10.2/x",
        definition=SECRET,          # gated: full-text-derived
        exact_quote="secret quote",  # gated: span text
        claim_text="secret claim",   # gated: claim text
    )
    g.add_node("concept::c", node_type="Concept", canonical_label="C",
               definition=SECRET)
    records = export._biblio_records(g)
    assert [r["work_id"] for r in records] == ["work_x"]
    assert set(records[0]) == {"work_id", *export._BIBLIO_FIELDS}
    for text in (export._bibtex_text(records), export._ris_text(records)):
        assert SECRET not in text
        assert "secret quote" not in text
        assert "secret claim" not in text
        assert "Clean Title" in text


def test_stub_entry_has_no_fabricated_title_and_key_degrades(tmp_path):
    conn = _seed(tmp_path)
    bib = _export(conn, tmp_path, "bibtex").read_text(encoding="utf-8")
    stub_entry = next(c for c in bib.split("\n\n") if "work_id=work_stub" in c)
    assert "title" not in stub_entry           # never fabricated
    assert stub_entry.startswith("@article{anonnd_work_stub")  # anon + nd key
    conn.close()


def test_deterministic_double_export(tmp_path):
    conn = _seed(tmp_path)
    a = _export(conn, tmp_path / "one", "bibtex").read_bytes()
    b = _export(conn, tmp_path / "two", "bibtex").read_bytes()
    assert a == b
    conn.close()
