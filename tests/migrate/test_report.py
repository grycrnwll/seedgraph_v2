"""The migration report (design §6.12): report.md, report.json and attention.md in
``migration/runs/<run_id>/`` account for every v2 row and every planted case."""

from __future__ import annotations

import copy
import json

import pytest

from fixtures import build_v2_home
from seedgraph.migrate import report as rep
from seedgraph.migrate.plan import Disposition

NO_PROCS = lambda: []  # noqa: E731
BIG_DISK = lambda p: 10 ** 15  # noqa: E731
NOW = "2026-10-23T04:00:00+00:00"


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_v2_home(tmp_path_factory.mktemp("v2") / "home")


@pytest.fixture(scope="module")
def outcome(fx, tmp_path_factory):
    v3 = tmp_path_factory.mktemp("v3") / "home"
    return rep.run_and_report(fx.home, v3, run_id="mig-rep-1", now=NOW, process_lister=NO_PROCS,
                              disk_free=BIG_DISK, graph_names={"did": "Difference-in-Differences"})


@pytest.fixture(scope="module")
def data(outcome):
    return json.loads(outcome.files.report_json.read_text(encoding="utf-8"))


def pid(outcome, fx, key):
    w = fx.works[key]
    return outcome.result.plan.ledger.get(w.slug, "works", w.work_id).v3_id


def test_files_are_written_into_the_run_folder(outcome):
    run_dir = outcome.result.run_dir
    assert outcome.files.report_md == run_dir / "report.md"
    assert outcome.files.report_json == run_dir / "report.json"
    assert outcome.files.attention_md == run_dir / "attention.md"
    for p in (outcome.files.report_md, outcome.files.report_json, outcome.files.attention_md):
        assert p.is_file() and p.stat().st_size > 0
    assert outcome.ok and outcome.result.ok


def test_every_v2_row_is_reconciled(data):
    assert data["problems"] == []
    rec = data["reconciliation"]
    assert set(rec) >= {"cache", "second_test", "land_value", "test_proj", "oct2026test", "dbg_probe"}
    for source, tables in rec.items():
        for table, r in tables.items():
            total = r["carried"] + r["merged"] + r["rebuilt"] + sum(r["not_carried"].values()) + r["unaccounted"]
            assert total == r["v2_rows"], (source, table, r)
            assert r["unaccounted"] == 0
    assert data["totals"]["unaccounted"] == 0


def test_an_unaccounted_row_fails_loudly_after_writing_the_report(outcome, tmp_path):
    result = copy.deepcopy(outcome.result)
    key = next(k for k, d in result.plan.ledger.entries.items() if k[1] == "works" and d.outcome == "carried")
    result.plan.ledger.entries[key] = Disposition("unaccounted")
    result.run_dir = tmp_path / "run"
    result.run_dir.mkdir()
    (result.run_dir / rep.V2_ROWS_FILE).write_text(
        (outcome.result.run_dir / rep.V2_ROWS_FILE).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(rep.ReportError) as err:
        rep.write_report(result)
    assert key[2] in str(err.value)
    written = json.loads((result.run_dir / "report.json").read_text(encoding="utf-8"))
    assert written["problems"] and written["totals"]["unaccounted"] == 1
    assert "unaccounted" in (result.run_dir / "report.md").read_text(encoding="utf-8").lower()


def test_a_row_missing_from_the_ledger_fails(outcome, tmp_path):
    result = copy.deepcopy(outcome.result)
    key = next(k for k in result.plan.ledger.entries if k[0] == "second_test" and k[1] == "structured_notes")
    del result.plan.ledger.entries[key]
    result.run_dir = tmp_path / "run"
    result.run_dir.mkdir()
    (result.run_dir / rep.V2_ROWS_FILE).write_text(
        (outcome.result.run_dir / rep.V2_ROWS_FILE).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(rep.ReportError) as err:
        rep.write_report(result)
    assert "structured_notes" in str(err.value)


def test_missing_v2_row_counts_fail(outcome, tmp_path):
    result = copy.deepcopy(outcome.result)
    result.run_dir = tmp_path / "run"
    result.run_dir.mkdir()
    with pytest.raises(rep.ReportError):
        rep.write_report(result)


def test_report_md_is_plain_english_and_names_the_sections(outcome):
    text = outcome.files.report_md.read_text(encoding="utf-8")
    for heading in ("Counts", "Papers merged", "test_proj", "current note", "Repairs", "Not converted",
                    "Needs your attention", "Concept decisions", "Citation decisions", "v2 answers",
                    "v2 home unchanged", "Dump"):
        assert heading.lower() in text.lower(), heading
    assert outcome.result.dump_sha256 in text
    assert "\r\n" not in outcome.files.report_md.read_bytes().decode("utf-8")


def test_attention_md_lists_items_with_their_ids(outcome):
    text = outcome.files.attention_md.read_text(encoding="utf-8")
    from seedgraph.library.attention import item_id_for
    for a in outcome.result.plan.attention:
        assert item_id_for(a.kind, a.subject) in text
    assert "notes hidden" in text.lower()


def test_totals_are_flat_numbers(data, outcome):
    t = data["totals"]
    assert all(isinstance(v, int) for v in t.values())
    assert t["library.papers"] == outcome.result.counts["papers"]
    assert t["graph.did.member"] == outcome.result.graph_counts["did"]["member"]
    assert t["attention.total"] == len(outcome.result.plan.attention)
    assert t["answers.exported"] == 3


def test_report_is_deterministic_across_runs(fx, data, tmp_path):
    other = rep.run_and_report(fx.home, tmp_path / "v3b", run_id="mig-rep-2", now="2026-10-25T00:00:00+00:00",
                               process_lister=NO_PROCS, disk_free=BIG_DISK,
                               graph_names={"did": "Difference-in-Differences"})
    d2 = json.loads(other.files.report_json.read_text(encoding="utf-8"))
    assert d2["totals"] == data["totals"]
    assert d2["dump_sha256"] == data["dump_sha256"]


# --- every planted case is accounted for in the report -------------------------------------

def _merge_of(data, paper_id):
    return next((m for m in data["merges"]["papers"] if m["paper_id"] == paper_id), None)


def _attn(data, kind, paper_id=None):
    return [a for a in data["attention"]["items"] if a["kind"] == kind and (paper_id is None or a["paper_id"] == paper_id)]


def _rec(data, source, table, label):
    r = data["reconciliation"][source][table]
    if label.startswith("not_carried:"):
        return r["not_carried"].get(label.split(":", 1)[1], 0)
    return r[label]


def check_merged(item, data, o, fx):
    keys = item.ids.get("work_keys")
    p = pid(o, fx, keys[0])
    m = _merge_of(data, p)
    assert m is not None and m["splittable"] is True
    assert {(fx.works[k].slug, fx.works[k].work_id) for k in keys} <= {tuple(x) for x in m["members"]}


def check_overlap(item, data, o, fx):
    check_merged(item, data, o, fx)
    assert item.expect["current_note_id"] in {c["note_id"] for c in data["current_elsewhere"]}


def check_shared_conflict(item, data, o, fx):
    miss = pid(o, fx, item.expect["miss_side"])
    a = _attn(data, "shared_pdf", miss)
    assert a and a[0]["hides_notes"] is True
    assert _merge_of(data, miss) is None


def check_relinked(item, data, o, fx):
    p = pid(o, fx, item.ids["work_key"])
    assert any(r["paper_id"] == p for r in data["repairs"]["relinks"])


def check_relink_failed(item, data, o, fx):
    p = pid(o, fx, item.ids["work_key"])
    assert any(r["paper_id"] == p and r["repair"] == "relink_failed" for r in data["repairs"]["relinks"])
    assert p in {r["paper_id"] for r in data["not_converted"]}


def check_conversion_failed(item, data, o, fx):
    assert _rec(data, "cache", "conversion_runs", "carried") >= len(item.ids["conversion_run_ids"])
    assert item.ids["source_file_id"] in {r["source_file_id"] for r in data["not_converted"]}


def check_noid_match(item, data, o, fx):
    p = pid(o, fx, item.ids["work_key"])
    assert p == pid(o, fx, item.ids["matches"]) and _merge_of(data, p) is not None


def check_identity(item, data, o, fx):
    assert _attn(data, "identity_check", pid(o, fx, item.ids["work_key"]))


def check_no_attention(item, data, o, fx):
    p = pid(o, fx, item.ids["work_key"])
    assert not [a for a in data["attention"]["items"] if a["paper_id"] == p and a["kind"] == "pdf_title_miss"]


def check_wrong_pdf(item, data, o, fx):
    a = _attn(data, "pdf_title_miss", pid(o, fx, item.ids["work_key"]))
    assert a and a[0]["hides_notes"] is True


def check_title_repaired(item, data, o, fx):
    p = pid(o, fx, item.ids["work_key"])
    assert any(t["paper_id"] == p and t["new"] == item.expect["title"] for t in data["repairs"]["titles"])


def check_malformed(item, data, o, fx):
    assert _attn(data, "malformed_title", pid(o, fx, item.ids["work_key"]))


def check_dup_decision(item, data, o, fx):
    assert item.ids["item_id"] in {d["item_id"] for d in data["citations"]["duplicate_decisions"]}


def check_citation_decided(item, data, o, fx):
    assert data["citations"]["review_decisions"] >= 1
    for unmapped in item.expect["unmapped"]:
        assert unmapped in {u["item_id"] for u in data["citations"]["unmapped_decisions"]}
    for collapsed in item.expect["mapped"]:
        assert collapsed in {c["item_id"] for c in data["citations"]["collapsed_onto_existing"]}


def check_open_reviews(item, data, o, fx):
    pairs = item.ids.get("items") or [item.ids["item"]]
    for slug, _ in pairs:
        assert _rec(data, slug, "review_queue", "not_carried:open_review_item") >= 1


def check_fake(item, data, o, fx):
    assert _attn(data, "test_converter_markdown", pid(o, fx, item.ids["work_key"]))


def check_unheld(item, data, o, fx):
    assert any(a["kind"] == "metadata_needed" and item.ids["source_file_id"][3:15] in (a["detail"] or "")
               for a in data["attention"]["items"])


def check_pending(item, data, o, fx):
    assert _rec(data, "cache", "conversion_runs", "not_carried:pending_run") == 1


def check_excluded(item, data, o, fx):
    assert data["graphs"]["land-value"]["excluded"] >= 1


def check_unreferenced(item, data, o, fx):
    w = fx.works[item.ids["work_key"]]
    assert _rec(data, w.slug, "works", "not_carried:" + item.expect["reason"]) >= 1


def check_held_unconverted(item, data, o, fx):
    assert pid(o, fx, item.ids["work_key"]) in {r["paper_id"] for r in data["not_converted"]}


def check_concept(item, data, o, fx):
    c = data["concepts"]
    want = f"{item.expect['kind']}:{item.expect['decided_by']}"
    assert c["counts"].get(want, 0) >= 1


def check_contradiction(item, data, o, fx):
    assert any(sorted([x["label_a"], x["label_b"]]) == sorted(item.ids["pair"])
               for x in data["concepts"]["contradictions"])
    assert _attn(data, "concept_contradiction")


def check_lens(item, data, o, fx):
    assert _rec(data, "second_test", "lens_outputs", "carried") == item.expect["lens_outputs_carried"]
    assert data["totals"]["library.lens_definitions"] == item.expect["lens_definitions"]


def check_dropped(item, data, o, fx):
    for t in item.ids["tables"]:
        assert _rec(data, "second_test", t, "not_carried:dropped_table") >= 1


def check_saved_answer(item, data, o, fx):
    assert item.expect["export"] in {e["path"] for e in data["answers"]["exported"]}


def check_run_scoped(item, data, o, fx):
    assert item.ids["path"] in data["answers"]["run_scoped_not_carried"]
    assert item.ids["answer_id"] not in {e["answer_id"] for e in data["answers"]["exported"]}


def check_dbg(item, data, o, fx):
    for slug in item.ids["slugs"]:
        r = data["reconciliation"][slug]
        assert all(v["carried"] == 0 and v["merged"] == 0 for v in r.values())


def check_cited_only(item, data, o, fx):
    check_merged(item, data, o, fx)
    assert data["graphs"]["did"]["frontier"] >= 1


CHECKS = {
    "overlap_diverging_notes": check_overlap,
    "cited_only": check_cited_only,
    "double_import": check_merged,
    "shared_pdf_conflicting_dois": check_shared_conflict,
    "shared_pdf_one_side_no_id": check_merged,
    "foreign_markdown": check_relinked,
    "foreign_markdown_unconverted": check_relink_failed,
    "conversion_failed": check_conversion_failed,
    "noid_title_year_match": check_noid_match,
    "noid_ambiguous": check_identity,
    "noid_unmatched": check_identity,
    "mojibake_markdown": check_no_attention,
    "mojibake_title": check_no_attention,
    "wrong_pdf": check_wrong_pdf,
    "malformed_title_cached": check_title_repaired,
    "malformed_title_uncached": check_malformed,
    "duplicate_verdict_different": check_dup_decision,
    "duplicate_verdict_same": check_dup_decision,
    "citation_review_decided": check_citation_decided,
    "citation_review_open": check_open_reviews,
    "fake_converter_markdown": check_fake,
    "external_markdown_no_markers": check_no_attention,
    "pdf_no_text_layer": check_no_attention,
    "unheld_pdf": check_unheld,
    "unheld_pdf_second_copy": check_unheld,
    "conversion_pending": check_pending,
    "excluded_in_graph": check_excluded,
    "metadata_only_unreferenced": check_unreferenced,
    "held_unconverted": check_held_unconverted,
    "concept_llm_merge": check_concept,
    "concept_user_must_link": check_concept,
    "concept_user_cannot_link": check_concept,
    "concept_contradiction": check_contradiction,
    "concept_agent_and_researcher_same_pair": check_concept,
    "lens_with_outputs": check_lens,
    "open_review_items": check_open_reviews,
    "dropped_tables": check_dropped,
    "saved_answer": check_saved_answer,
    "run_scoped_answer": check_run_scoped,
    "dbg_project": check_dbg,
}


def test_every_planted_kind_has_a_report_check(fx):
    assert sorted(CHECKS) == fx.kinds


def test_every_planted_item_is_in_the_report(fx, data, outcome):
    for item in fx.items:
        CHECKS[item.kind](item, data, outcome, fx)


def test_folded_in_papers_are_listed_by_title(data):
    folded = data["graphs"]["did"]["folded_in"]
    assert folded and all(f["title"] is not None or f["paper_id"] for f in folded)
    assert data["totals"]["graph.did.folded_in"] == len(folded)
