"""Every planted item of the synthetic v2 home ends in its expected outcome (design §11 S6).

One check per planted ``kind``; an unknown kind fails. Where the fixture's ``expect``
holds a placeholder label, the check pins it to the concrete outcome the design (and
the S6 rulings) give; those pins are commented. Expectations owned by S7 (parity
classes, the answers export) are checked only as far as S6 provides their inputs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fixtures import build_v2_home
from seedgraph.library.attention import HIDDEN_PAPERS_SQL
from seedgraph.library.identity import paper_id_for_key
from seedgraph.library.titlecheck import check_pdf_text, first_pages_from_markdown, first_pages_from_pdf
from seedgraph.migrate import run_migration
from seedgraph.semantic.merge import normalize as normalize_label

NOW = "2026-10-23T03:00:00+00:00"


def ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


class World:
    def __init__(self, fx, result):
        self.fx, self.r = fx, result
        v3 = result.v3_home
        self.lib = ro(v3 / "library" / "library.db")
        self.graph_conn = {s: ro(v3 / "graphs" / s / "graph.db") for s in ("diff_in_diff", "land_value")}
        self.v3 = v3

    def one(self, sql, *args):
        return self.lib.execute(sql, args).fetchone()

    def all(self, sql, *args):
        return self.lib.execute(sql, args).fetchall()

    def paper(self, key: str) -> str:
        w = self.fx.works[key]
        row = self.one("SELECT v3_id FROM v2_id_map WHERE v2_project = ? AND v2_table = 'works' AND v2_id = ?",
                       w.slug, w.work_id)
        assert row is not None, f"{key} not carried"
        return row[0]

    def carried(self, key: str) -> bool:
        w = self.fx.works[key]
        return self.one("SELECT 1 FROM v2_id_map WHERE v2_project = ? AND v2_table = 'works' AND v2_id = ?",
                        w.slug, w.work_id) is not None

    def status(self, pid: str) -> str:
        return self.one("SELECT status FROM paper_status WHERE paper_id = ?", pid)[0]

    def gstatus(self, slug: str, pid: str):
        row = self.graph_conn[slug].execute("SELECT status, origin FROM graph_papers WHERE paper_id = ?",
                                            (pid,)).fetchone()
        return tuple(row) if row else None

    def attention(self, pid: str | None = None, kind: str | None = None):
        sql, args = "SELECT * FROM attention_items WHERE 1=1", []
        if pid is not None:
            sql += " AND paper_id = ?"
            args.append(pid)
        if kind is not None:
            sql += " AND kind = ?"
            args.append(kind)
        return self.all(sql, *args)

    def hidden(self) -> set[str]:
        return {r[0] for r in self.lib.execute(HIDDEN_PAPERS_SQL)}

    def check(self, pid: str):
        row = self.one("SELECT pdf_check_status, pdf_check_json, markdown_id, source_file_id FROM paper_files "
                       "WHERE paper_id = ?", pid)
        return row

    def note(self, note_id: str):
        return self.one("SELECT * FROM structured_notes WHERE note_id = ?", note_id)

    def ledger(self, slug, table, key):
        return self.r.plan.ledger.get(slug, table, key)

    def merge_keys(self, pid: str) -> set[str]:
        return {r[0] for r in self.all("SELECT joined_by_key_type FROM paper_merge_members WHERE paper_id = ?", pid)}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    fx = build_v2_home(tmp_path_factory.mktemp("v2") / "home")
    r = run_migration(fx.home, tmp_path_factory.mktemp("v3") / "home", run_id="mig-planted", now=NOW,
                      process_lister=lambda: [], disk_free=lambda p: 10 ** 15)
    assert r.errors == [] and r.swapped
    return World(fx, r)


# --- checks per kind ----------------------------------------------------------------------

def c_overlap_diverging_notes(w: World, it):
    a, b = it.ids["work_keys"]
    pid = w.paper(a)
    assert w.paper(b) == pid and it.expect["merged"]
    versions = [r[0] for r in w.all("SELECT note_id FROM structured_notes WHERE paper_id = ? ORDER BY version_no", pid)]
    assert versions == it.expect["version_order"]
    assert w.one("SELECT note_id FROM note_current WHERE paper_id = ?", pid)[0] == it.expect["current_note_id"]
    assert w.gstatus(it.expect["graph"], pid) == ("member", "v2:second_test")
    canon = it.expect["manual_spans_dedupe_to"]
    for sid in canon:
        assert w.one("SELECT 1 FROM evidence_spans WHERE span_id = ?", sid)
    for slug, sids in it.ids["span_ids"].items():
        for sid, c in zip(sids, canon):
            if sid != c:
                assert w.one("SELECT v3_id FROM v2_id_map WHERE v2_project = ? AND v2_table = 'evidence_spans' "
                             "AND v2_id = ?", slug, sid)[0] == c
                assert w.one("SELECT 1 FROM evidence_spans WHERE span_id = ?", sid) is None
    # claim anchors of the test_proj note were re-pointed at the canonical spans
    anchored = {r[0] for r in w.all(
        "SELECT cs.span_id FROM claim_spans cs JOIN extracted_claims c USING (claim_id) "
        "WHERE c.structured_note_id = ?", it.expect["current_note_id"])}
    assert anchored == set(canon)
    # S7 parity class 'current_elsewhere': S6 lists the paper
    assert any(x["paper_id"] == pid and x["from_project"] == "test_proj" and x["graph"] == "diff_in_diff"
               for x in w.r.report["current_elsewhere"])


def c_cited_only(w: World, it):
    a, b = it.ids["work_keys"]
    pid = w.paper(a)
    assert w.paper(b) == pid and w.status(pid) == "cited_only"
    assert w.gstatus("diff_in_diff", pid)[0] == it.expect["graph_status"]["diff_in_diff"]
    links = {r["provenance"]: r["confidence"] for r in w.all(
        "SELECT * FROM citation_links WHERE target_paper_id = ?", pid)}
    assert links == it.expect["citation_links_dedupe"]
    # the winning parsed_bibliography link (test_proj's) carries the chosen list's reference id
    ref = w.one("SELECT reference_id FROM citation_links WHERE target_paper_id = ? AND "
                "provenance = 'parsed_bibliography'", pid)[0]
    assert w.one("SELECT origin FROM reference_entries WHERE reference_id = ?", ref)[0] == "v2:second_test"


def c_double_import(w: World, it):
    a, b = it.ids["work_keys"]
    pid = w.paper(a)
    assert w.paper(b) == pid
    assert it.expect["merge_key"] in w.merge_keys(pid)
    assert w.one("SELECT COUNT(*) FROM paper_merge_members WHERE paper_id = ?", pid)[0] == 2
    assert w.one("SELECT COUNT(*) FROM duplicate_decisions WHERE ? IN (paper_a, paper_b)", pid)[0] == 0
    versions = [r[0] for r in w.all("SELECT note_id FROM structured_notes WHERE paper_id = ? ORDER BY version_no", pid)]
    assert versions == it.expect["version_order"]
    assert w.one("SELECT source_file_id FROM paper_files WHERE paper_id = ?", pid)[0] == it.ids["source_file_id"]


def c_shared_pdf_conflicting_dois(w: World, it):
    a, b = it.ids["work_keys"]
    pa, pb = w.paper(a), w.paper(b)
    assert pa != pb and not it.expect["merged"]
    for key, pid in ((a, pa), (b, pb)):
        assert w.check(pid)["source_file_id"] == it.ids["source_file_id"]
        assert w.check(pid)["pdf_check_status"] == it.expect["pdf_check"][key]
        assert not w.attention(pid, "shared_pdf")  # the item is about the PDF, not one side
        # a side whose title misses is hidden through its own per-paper title-miss item
        misses = w.attention(pid, "pdf_title_miss")
        assert len(misses) == int(bool(it.expect["hides_notes"][key]))
        assert all(m["hides_notes"] == 1 for m in misses)
        assert (pid in w.hidden()) == bool(it.expect["hides_notes"][key])
    items = [r for r in w.attention(kind="shared_pdf")
             if json.loads(r["subject_json"]) == {"source_file_id": it.ids["source_file_id"]}]
    assert len(items) == 1
    item = items[0]
    assert item["paper_id"] is None and item["hides_notes"] == 0
    sides = {s["paper_id"]: s["title_check"] for s in json.loads(item["detail"])["papers"]}
    assert sides == {pa: it.expect["pdf_check"][a], pb: it.expect["pdf_check"][b]}


def c_shared_pdf_one_side_no_id(w: World, it):
    a, b = it.ids["work_keys"]
    pid = w.paper(a)
    assert w.paper(b) == pid and it.expect["merge_key"] in w.merge_keys(pid)
    # Deviation from the fixture's "graph": None: the merged paper holds test_proj's work, and
    # design §6.2 folds every held test_proj paper into the DiD graph.
    assert w.gstatus("diff_in_diff", pid) == ("member", "v2:test_proj")
    assert w.gstatus("land_value", pid) is None


def c_foreign_markdown(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert w.check(pid)["markdown_id"] == it.expect["relink_to"]
    assert w.note(it.ids["note_id"])["version_label"] == it.expect["version_label"]
    assert any(x["paper_id"] == pid and x["repair"] == "relinked" for x in w.r.report["relinks"])


def c_foreign_markdown_unconverted(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert w.check(pid)["markdown_id"] is None
    assert w.one("SELECT reason FROM conversion_queue WHERE source_file_id = ?",
                 it.ids["source_file_id"])[0] == it.expect["queue_reason"]
    assert w.note(it.ids["note_id"])["version_label"] == it.expect["version_label"]
    assert w.one("SELECT COUNT(*) FROM note_current WHERE paper_id = ?", pid)[0] == 0


def c_conversion_failed(w: World, it):
    for run_id in it.ids["conversion_run_ids"]:
        assert w.one("SELECT run_status FROM conversion_runs WHERE conversion_run_id = ?", run_id)[0] == "failed"
    assert w.one("SELECT reason FROM conversion_queue WHERE source_file_id = ?",
                 it.ids["source_file_id"])[0] == it.expect["queue_reason"]


def c_noid_title_year_match(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert w.paper(it.ids["matches"]) == pid
    assert it.expect["merge_key"] in w.merge_keys(pid)
    assert w.gstatus("land_value", pid)[0] == it.expect["graph_status"]["land_value"]


def c_noid_ambiguous(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert all(w.paper(c) != pid for c in it.ids["candidates"])
    items = [json.loads(r["subject_json"]) for r in w.attention(pid, "identity_check")]
    # placeholder 'ambiguous_title_year' pinned to identity's reason 'ambiguous_title'
    assert [i["reason"] for i in items] == ["ambiguous_title"]
    cands = {c["paper_id"] for c in json.loads(w.attention(pid, "identity_check")[0]["detail"])["candidates"]}
    assert cands == {w.paper(c) for c in it.ids["candidates"]}


def c_noid_unmatched(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert pid == paper_id_for_key(it.expect["paper_id_key"])
    items = w.attention(pid, "identity_check")
    assert [json.loads(r["subject_json"])["reason"] for r in items] == ["no_match"]
    cands = [c["paper_id"] for c in json.loads(items[0]["detail"])["candidates"]]
    assert w.paper(it.ids["near_miss"]) in cands


def c_mojibake_markdown(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert w.check(pid)["pdf_check_status"] == it.expect["pdf_check"]
    # 'matched_on: folded_variant' pinned: the stored check passes on the printed DOI first; the
    # title alone still matches the markdown through the folded (mojibake) page variant.
    md = (w.v3 / "library" / "markdown" / f"{it.ids['markdown_id'][3:]}.md").read_text(encoding="utf-8")
    title = w.one("SELECT title FROM papers WHERE paper_id = ?", pid)[0]
    res = check_pdf_text(title, None, first_pages_from_markdown(md))
    assert res.status == "pass" and "page folded" in res.evidence


def c_mojibake_title(w: World, it):
    pid = w.paper(it.ids["work_key"])
    row = w.check(pid)
    assert row["pdf_check_status"] == it.expect["pdf_check"]
    assert "title folded" in json.loads(row["pdf_check_json"])["evidence"]  # 'folded_variant'


def c_wrong_pdf(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert w.check(pid)["pdf_check_status"] == it.expect["pdf_check"]
    items = w.attention(pid, it.expect["attention"])
    assert len(items) == 1 and items[0]["hides_notes"] == it.expect["hides_notes"]
    assert pid in w.hidden()
    assert w.note(it.ids["note_id"])["version_label"] is None  # not confirmed: hidden, not labelled


def c_malformed_title_cached(w: World, it):
    pid = w.paper(it.ids["work_key"])
    row = w.one("SELECT title, title_source FROM papers WHERE paper_id = ?", pid)
    assert (row[0], row[1]) == (it.expect["title"], it.expect["title_source"])
    assert not w.attention(pid, "malformed_title")
    assert w.check(pid)["pdf_check_status"] == "pass"


def c_malformed_title_uncached(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert len(w.attention(pid, it.expect["attention"])) == 1
    # 'title_source: None' pinned: not repaired, so the v2 title stays (title_source 'v2')
    assert w.one("SELECT title_source FROM papers WHERE paper_id = ?", pid)[0] == "v2"
    # a malformed title says nothing about the PDF: no title-miss item, notes not hidden
    assert not w.attention(pid, "pdf_title_miss") and pid not in w.hidden()


def c_malformed_title_unheld(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert w.status(pid) != "held"
    assert not w.attention(pid, "malformed_title")  # only held papers get the attention item


def c_duplicate_verdict_different(w: World, it):
    a, b = (w.paper(k) for k in it.ids["work_keys"])
    assert a != b
    row = w.one("SELECT * FROM duplicate_decisions WHERE paper_a = ? AND paper_b = ?", *sorted((a, b)))
    assert (row["verdict"], row["decided_by"]) == (it.expect["duplicate_decision"], it.expect["decided_by"])
    subj = [json.loads(r["subject_json"]) for r in w.attention(b, "identity_check")]
    assert [(s["reason"], s["key_type"]) for s in subj] == [("never_merge_skipped", it.expect["rule_skipped"])]


def c_duplicate_verdict_same(w: World, it):
    a, b = (w.paper(k) for k in it.ids["work_keys"])
    assert a != b  # two DOIs: precedence never resolves a conflict
    row = w.one("SELECT * FROM duplicate_decisions WHERE paper_a = ? AND paper_b = ?", *sorted((a, b)))
    assert (row["verdict"], row["decided_by"]) == ("same", "researcher")
    assert w.status(b) == "cited_only"  # carried as the subject of a decision


def c_citation_review_decided(w: World, it):
    refs_st = it.ids["reference_ids"]["second_test"]
    overlap = w.paper("overlap@second_test")
    lst = w.all("SELECT reference_id, ordinal, origin FROM reference_entries WHERE citing_paper_id = ? "
                "ORDER BY ordinal", overlap)
    assert [r[0] for r in lst] == refs_st and {r[2] for r in lst} == {"v2:second_test"}
    for item_id, ref in it.expect["mapped"].items():
        assert w.one("SELECT v3_id FROM v2_id_map WHERE v2_table = 'review_queue' AND v2_id = ?", item_id)[0] == ref
        assert w.one("SELECT decision FROM citation_review_decisions WHERE reference_id = ?", ref)[0] == "reject"
    unmapped = {u["item_id"] for u in w.r.report["unmapped_decisions"]}
    assert set(it.expect["unmapped"]) <= unmapped
    for slug, item_id, _ in it.ids["items"]:
        d = w.ledger(slug, "review_queue", item_id)
        assert d is not None and d.outcome in ("carried", "merged", "not_carried")
        if item_id in it.expect["unmapped"]:
            assert d.reason == "unmapped_citation_decision"
        else:
            assert d.v3_table == "citation_review_decisions"
    assert {r[0] for r in w.all("SELECT decision FROM citation_review_decisions")} == {it.expect["decision"]}
    rec = w.one("SELECT rejected_record_json, reason FROM citation_review_decisions WHERE reference_id = ?", refs_st[2])
    assert json.loads(rec[0])["doi"] == "10.5555/sg.unrelated" and rec[1] == "low_title_overlap"


def c_citation_review_open(w: World, it):
    slug, item_id = it.ids["item"]
    assert w.ledger(slug, "review_queue", item_id).reason == "open_review_item"


def c_fake_converter_markdown(w: World, it):
    pid = w.paper(it.ids["work_key"])
    # carried (design §6.5) ...
    assert w.one("SELECT 1 FROM markdown_documents WHERE markdown_id = ?", it.ids["markdown_id"])
    assert (w.v3 / "library" / "markdown" / f"{it.ids['markdown_id'][3:]}.md").is_file()
    # ... but not as the paper's conversion (ruling 4): listed not converted, queued, flagged
    assert w.check(pid)["markdown_id"] is None
    sf = w.check(pid)["source_file_id"]
    assert w.one("SELECT reason FROM conversion_queue WHERE source_file_id = ?", sf)[0] == "not_converted"
    assert any(x["paper_id"] == pid for x in w.r.report["not_converted"])
    assert len(w.attention(pid, "test_converter_markdown")) == 1  # placeholder pinned to this kind


def c_external_markdown_no_markers(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert w.check(pid)["pdf_check_status"] == it.expect["pdf_check"]
    md = (w.v3 / "library" / "markdown" / f"{it.ids['markdown_id'][3:]}.md").read_text(encoding="utf-8")
    pages = first_pages_from_markdown(md)  # 'first_8000_chars'
    assert len(pages) == 1 and pages[0] == md[:8000]


def c_pdf_no_text_layer(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert w.check(pid)["pdf_check_status"] == it.expect["pdf_check"]
    sha = it.ids["source_file_id"][3:]
    pages = first_pages_from_pdf(w.v3 / "library" / "pdfs" / f"{sha}.pdf")
    title = w.one("SELECT title FROM papers WHERE paper_id = ?", pid)[0]
    assert check_pdf_text(title, None, pages or []).status == it.expect["pdf_check_pdf_only"]


def _unheld(w: World, it):
    row = w.one("SELECT paper_id FROM paper_files WHERE source_file_id = ?", it.ids["source_file_id"])
    pid = row[0]
    assert w.status(pid) == "held"
    assert w.one("SELECT title FROM papers WHERE paper_id = ?", pid)[0] is None
    assert len(w.attention(pid, "metadata_needed")) == 1
    assert w.check(pid)["pdf_check_status"] == "unknown"
    assert w.check(pid)["markdown_id"] == it.ids["markdown_id"]
    assert pid == paper_id_for_key("pdf:" + it.ids["source_file_id"][3:])


def c_unheld_pdf(w, it):
    _unheld(w, it)


def c_unheld_pdf_second_copy(w, it):
    _unheld(w, it)


def c_conversion_pending(w: World, it):
    assert w.one("SELECT 1 FROM conversion_runs WHERE conversion_run_id = ?", it.ids["conversion_run_id"]) is None
    assert w.ledger("cache", "conversion_runs", it.ids["conversion_run_id"]).reason == "pending_run"


def c_excluded_in_graph(w: World, it):
    pid = w.paper(it.ids["work_key"])
    assert w.gstatus("land_value", pid)[0] == "excluded" and w.status(pid) == "cited_only"


def c_metadata_only_unreferenced(w: World, it):
    for key in [it.ids["work_key"], *it.ids["also"]]:
        assert not w.carried(key)
        ref = w.fx.works[key]
        assert w.ledger(ref.slug, "works", ref.work_id).reason == it.expect["reason"]


def c_held_unconverted(w: World, it):
    pid = w.paper(it.ids["work_key"])
    sf = w.check(pid)["source_file_id"]
    assert w.one("SELECT reason FROM conversion_queue WHERE source_file_id = ?", sf)[0] == it.expect["queue_reason"]
    assert w.check(pid)["markdown_id"] is None


def _concept(w: World, a: str, b: str):
    a, b = sorted((normalize_label(a), normalize_label(b)))  # labels are stored normalized
    return w.all("SELECT * FROM concept_decisions WHERE label_a = ? AND label_b = ? ORDER BY kind", a, b)


def _concept_one(w: World, it):
    rows = _concept(w, *it.ids["pair"])
    assert len(rows) == 1
    assert (rows[0]["kind"], rows[0]["decided_by"], rows[0]["state"]) == \
        (it.expect["kind"], it.expect["decided_by"], it.expect["state"])


def c_concept_llm_merge(w, it):
    _concept_one(w, it)


def c_concept_user_must_link(w, it):
    _concept_one(w, it)


def c_concept_user_cannot_link(w, it):
    _concept_one(w, it)


def c_concept_contradiction(w: World, it):
    rows = _concept(w, *it.ids["pair"])
    assert len(rows) == it.expect["rows_kept"] == 2  # design §6.10: both rows kept
    assert {r["kind"] for r in rows} == {"merge", "never_merge"}
    assert len(w.attention(None, it.expect["attention"])) == 1
    (c,) = w.r.report["concept_contradictions"]  # ruling 4: the newest decision is named
    assert (c["newest_kind"], c["newest_project"]) == ("merge", "second_test")


def c_concept_agent_and_researcher_same_pair(w: World, it):
    rows = _concept(w, *it.ids["pair"])
    assert len(rows) == it.expect["rows"]
    assert (rows[0]["kind"], rows[0]["decided_by"]) == (it.expect["kind"], it.expect["decided_by"])


def c_lens_with_outputs(w: World, it):
    rows = w.all("SELECT * FROM lens_definitions")
    assert len(rows) == it.expect["lens_definitions"]
    assert rows[0]["definition_hash"] == it.ids["definition_hash"]
    assert rows[0]["definition_yaml"] == (w.fx.home / it.ids["yaml"]).read_text(encoding="utf-8")
    lenses = [r[0] for r in w.graph_conn["diff_in_diff"].execute("SELECT lens_id FROM graph_lenses")]
    assert lenses == it.expect["graph_lenses"]["diff_in_diff"]
    n = w.one("SELECT COUNT(*) FROM lens_outputs WHERE lens_output_id IN (?, ?)", *it.ids["lens_output_ids"])[0]
    assert n == it.expect["lens_outputs_carried"]


def c_open_review_items(w: World, it):
    for slug, item_id in it.ids["items"]:
        assert w.ledger(slug, "review_queue", item_id).reason == "open_review_item"


def c_dropped_tables(w: World, it):
    for table in it.ids["tables"]:
        entries = [d for (s, t, _), d in w.r.plan.ledger.entries.items() if t == table and s == "second_test"]
        assert entries and all(d.reason == "dropped_table" for d in entries), table


def c_saved_answer(w: World, it):
    # S7 exports answers; S6 captures the file list in the snapshot.
    assert it.ids["path"] in w.r.report["answers"]["saved"][it.ids["slug"]]


def c_run_scoped_answer(w: World, it):
    assert it.ids["path"] not in w.r.report["answers"]["saved"][it.ids["slug"]]
    assert it.ids["path"] in w.r.report["answers"]["run_scoped_not_carried"][it.ids["slug"]]


def c_dbg_project(w: World, it):
    for slug in it.ids["slugs"]:
        mine = [d for (s, _, _), d in w.r.plan.ledger.entries.items() if s == slug]
        assert mine and all(d.outcome == "not_carried" for d in mine)
        assert w.one("SELECT COUNT(*) FROM v2_id_map WHERE v2_project = ?", slug)[0] == 0


CHECKS = {name[2:]: fn for name, fn in list(globals().items()) if name.startswith("c_") and callable(fn)}


def test_every_planted_kind_has_a_check(world):
    assert sorted(CHECKS) == world.fx.kinds


@pytest.mark.parametrize("kind", sorted(CHECKS))
def test_planted_item_outcome(world, kind):
    items = world.fx.by_kind(kind)
    assert items, kind
    for it in items:
        CHECKS[kind](world, it)
