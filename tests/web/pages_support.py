"""Fixture library for the U7 page tests (Home, Library, Batches, Review, Context graphs,
Settings). Invented text only; every home is under pytest's tmp.

``build(home)`` writes into ``<home>/library``:

- ``lp_test0001`` held, in graph ``widgets`` (seed), with a current note made through the
  real work routine (batch ``done_batch``, finished);
- ``lp_test0002`` held, in ``widgets``, no note (needs notes), queued in ``open_batch``;
- ``lp_test0003`` held, in ``widgets``, an open ``text_garbled`` item (held back);
- ``lp_cited1`` cited-only with a DOI, on the ``widgets`` frontier (needs a PDF);
- ``lp_cited2`` cited-only, no DOI, in no graph;
- open attention items: ``text_garbled`` (lp_test0003), ``pdf_title_miss`` (lp_test0002),
  ``identity_check`` (lp_cited2 vs lp_cited1), ``renote_offered`` (lp_test0001) and one
  migration-only kind (``converter_markdown_test``);
- a migrated duplicate decision and a migrated citation rejection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from work.routine_support import good_note, make_paper

from seedgraph.library import attention, graphs
from seedgraph.library.store import open_library_at
from seedgraph.work import routine
from seedgraph.work.claims import WorkerRef

NOW = "2026-01-01T00:00:00+00:00"
GRAPH = "widgets"
DOI = "10.9999/invented.2031.7"
T0 = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)


@dataclass
class Fixture:
    home: Path
    noted: str = "lp_test0001"
    no_note: str = "lp_test0002"
    garbled: str = "lp_test0003"
    cited: str = "lp_cited1"
    cited_no_doi: str = "lp_cited2"
    done_batch: str = ""
    open_batch: str = ""
    garbled_item: str = ""
    title_miss_item: str = ""
    identity_item: str = ""
    renote_item: str = ""
    other_item: str = ""


def _cited(lib, paper_id: str, title: str, doi: str | None) -> None:
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, year, authors_json, created_at) VALUES (?, ?, 2031, ?, ?)",
                     (paper_id, title, '["Ada Inventor"]', NOW))
        if doi:
            conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value, source) VALUES (?, 'doi', ?, 'v2')",
                         (paper_id, doi))


def build(home: Path, *, now: datetime = T0) -> Fixture:
    fx = Fixture(home=home)
    lib = open_library_at(home / "library")
    try:
        make_paper(lib, 1, title="Invented widget subsidies")
        make_paper(lib, 2, title="A second invented widget study")
        make_paper(lib, 3, title="A garbled invented scan")
        _cited(lib, fx.cited, "A cited invented treatise", DOI)
        _cited(lib, fx.cited_no_doi, "Another cited invented treatise", None)
        with graphs.create_graph(lib, GRAPH, display_name="Invented widgets") as g:
            graphs.set_membership(g, fx.noted, "member", is_seed=True)
            graphs.set_membership(g, fx.no_note, "member")
            graphs.set_membership(g, fx.garbled, "member")
            graphs.set_membership(g, fx.cited, "frontier")

        done = routine.create_batch(lib, "notes", paper_ids=[fx.noted], now=now)
        reply = routine.claim_next(lib, done.batch_id, WorkerRef("interactive", "w1", client="claude"), now=now)
        pkg = reply.message
        result = routine.submit(lib, pkg["packet_id"], pkg["claim_id"], note=good_note(), now=now)
        assert result.status == "accepted", result
        fx.done_batch = done.batch_id
        fx.open_batch = routine.create_batch(lib, "notes", paper_ids=[fx.no_note], now=now).batch_id

        fx.garbled_item = attention.raise_item(
            lib, "text_garbled", {"paper_id": fx.garbled}, paper_id=fx.garbled,
            detail="the invented text layer looks garbled", hides_notes=False).item_id
        fx.title_miss_item = attention.raise_item(
            lib, "pdf_title_miss", {"paper_id": fx.no_note}, paper_id=fx.no_note,
            detail="the invented title was not found in the PDF").item_id
        fx.identity_item = attention.raise_item(
            lib, "identity_check", {"paper_id": fx.cited_no_doi, "candidates": [fx.cited]},
            paper_id=fx.cited_no_doi, detail="may be the same invented paper").item_id
        fx.renote_item = attention.raise_item(
            lib, "renote_offered", {"paper_id": fx.noted}, paper_id=fx.noted,
            detail="a better invented conversion is available", hides_notes=True).item_id
        fx.other_item = attention.raise_item(
            lib, "converter_markdown_test", {"v2": "invented"}, detail="migrated invented item",
            origin="migration").item_id

        with lib.write() as conn:
            conn.execute("INSERT INTO duplicate_decisions (paper_a, paper_b, verdict, decided_by, reason, decided_at, "
                         "origin) VALUES ('lp_aaa', 'lp_bbb', 'different', 'researcher', 'invented reason', ?, "
                         "'migration')", (NOW,))
            conn.execute("INSERT INTO reference_entries (reference_id, citing_paper_id, raw_reference_text, "
                         "created_at) VALUES ('ref_1', ?, 'Inventor, A. (2001). A made-up treatise.', ?)",
                         (fx.noted, NOW))
            conn.execute("INSERT INTO citation_review_decisions (reference_id, decision, reason, decided_at, origin) "
                         "VALUES ('ref_1', 'reject', 'invented wrong match', ?, 'migration')", (NOW,))
    finally:
        lib.close()
    return fx


def token(client) -> str:
    return re.search(r'name="csrf-token" content="([^"]+)"', client.get("/settings").text).group(1)


def no_inline_script_or_style(html: str) -> bool:
    """No inline script body, style attribute, <style> block or hx-on (the CSP, §8)."""
    for m in re.finditer(r"<script\b([^>]*)>(.*?)</script>", html, re.S | re.I):
        if "src=" not in m.group(1) or m.group(2).strip():
            return False
    return not re.search(r"\sstyle=|<style\b|hx-on|\son[a-z]+=", html, re.I)


def post_forms_lead_with_csrf(html: str) -> bool:
    """Every ``<form method="post">`` has the CSRF field as its first input."""
    for m in re.finditer(r'<form\b[^>]*method="post"[^>]*>(.*?)</form>', html, re.S | re.I):
        first = re.search(r"<(input|select|textarea|button)\b[^>]*>", m.group(1), re.I)
        if first is None or 'name="csrf_token"' not in first.group(0):
            return False
    return True
