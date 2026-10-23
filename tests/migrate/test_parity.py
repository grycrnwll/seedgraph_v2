"""Search parity (design §8.2-§8.4): v2 match sets captured from the read-only v2 home,
compared with v3 graph-scoped match sets; every difference is classified into a
named known-difference class, and anything unclassified fails."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from fixtures import build_v2_home
from seedgraph.library.store import is_dirty, open_library_at
from seedgraph.migrate import parity
from seedgraph.migrate import report as rep

NO_PROCS = lambda: []  # noqa: E731
BIG_DISK = lambda p: 10 ** 15  # noqa: E731
QUERIES = {"did": ["eleven percent", "responds to", "made-up sample"],
           "land-value": ["eleven percent", "responds to"]}


def tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    """The synthetic v2 home, with one fixture gap closed in a test-local copy: the
    fixture's lens claim has no ``claim_fts`` row, which v2's lens runner always
    writes (``lenses/runner.py``). Without it v3's rebuilt index finds a claim v2's
    index cannot, an FTS drift no named class covers (and correctly unexplained)."""
    built = build_v2_home(tmp_path_factory.mktemp("v2") / "built")
    home = tmp_path_factory.mktemp("v2copy") / "home"
    shutil.copytree(built.home, home)
    conn = sqlite3.connect(home / "projects" / "second_test" / "project.db")
    with conn:
        conn.execute("INSERT INTO claim_fts (claim_id, normalized_label, claim_text) "
                     "SELECT c.claim_id, c.normalized_label, c.claim_text FROM extracted_claims c "
                     "WHERE c.claim_id NOT IN (SELECT claim_id FROM claim_fts)")
    conn.close()
    built.home = home
    return built


@pytest.fixture(scope="module")
def home(fx, tmp_path_factory):
    v3 = tmp_path_factory.mktemp("v3") / "home"
    rep.run_and_report(fx.home, v3, run_id="mig-par-1", now="2026-10-23T05:00:00+00:00",
                       process_lister=NO_PROCS, disk_free=BIG_DISK)
    parity.write_queries(v3, QUERIES)
    return v3


@pytest.fixture(scope="module")
def checked(fx, home):
    before = tree(fx.home)
    parity.capture(fx.home, home)
    assert tree(fx.home) == before
    return parity.check(home)


def classes(result, graph=None):
    out = {}
    for row in result.differences:
        if graph is None or row.graph == graph:
            out.setdefault(row.cls, []).append(row)
    return out


def test_capture_writes_v2_expected_with_attribution(checked, home):
    data = json.loads((home / parity.EXPECTED_FILE).read_text(encoding="utf-8"))
    did = data["graphs"]["did"]
    assert did["project"] == "second_test"
    q = did["queries"]["eleven percent"]
    assert q["spans"] and q["claims"] and q["notes"]
    span = q["spans"][0]
    assert len(span["key"]) == 3 and span["work_ids"] and span["markdown_id"]
    assert q["claims"][0]["work_id"] and q["notes"][0]["work_id"]
    assert data["graphs"]["land-value"]["project"] == "land_value"


def test_baseline_parity_holds_with_every_difference_explained(checked):
    assert checked.ok, [(d.graph, d.query, d.kind, d.key, d.detail) for d in checked.unexplained]
    assert checked.unexplained == []
    assert checked.differences


def test_each_named_class_is_produced(checked):
    assert set(classes(checked)) == set(parity.CLASSES)


def _note_class(result, note_id):
    return {d.cls for d in result.differences if d.kind == "notes" and d.key == note_id}


def test_planted_items_land_in_their_class(fx, checked):
    for item in fx.items:
        cls = item.expect.get("parity_class")
        if cls is None:
            continue
        if item.kind == "overlap_diverging_notes":
            assert _note_class(checked, item.ids["note_ids"][0]) == {cls}
        elif item.kind == "double_import":
            assert _note_class(checked, item.ids["note_ids"][0]) == {cls}
        elif item.kind == "shared_pdf_conflicting_dois":
            assert _note_class(checked, item.ids["note_ids"][1]) == {cls}
        elif item.kind in ("wrong_pdf", "foreign_markdown", "foreign_markdown_unconverted"):
            assert _note_class(checked, item.ids["note_id"]) == {cls}
        else:
            raise AssertionError(f"no parity expectation for {item.kind}")


def test_folded_in_and_other_project_spans_are_extra_hits(checked):
    by = classes(checked, "did")
    assert all(d.direction == "extra" for d in by["folded_in"])
    assert all(d.direction == "extra" and d.kind == "spans" for d in by["spans_from_other_projects"])
    assert any(d.direction == "missing" and d.kind == "spans" for d in classes(checked, "land-value")["relinked"])


def test_parity_md_and_json_are_written_into_the_run(checked, home):
    run_dir = home / "migration" / "runs" / "mig-par-1"
    assert checked.path == run_dir / "parity.md"
    text = checked.path.read_text(encoding="utf-8")
    assert "eleven percent" in text and "unexplained" in text.lower()
    data = json.loads((run_dir / "parity.json").read_text(encoding="utf-8"))
    assert data["ok"] is True
    row = next(r for r in data["queries"] if r["graph"] == "did" and r["query"] == "eleven percent")
    assert {"matched", "allowed", "unexplained", "top20_overlap"} <= set(row["spans"])


def test_reading_the_live_library_leaves_it_clean(checked, home):
    lib = open_library_at(home / "library")
    try:
        assert not is_dirty(lib.conn)
    finally:
        lib.close()


def test_an_injected_unexplained_difference_fails(checked, home, tmp_path):
    data = json.loads((home / parity.EXPECTED_FILE).read_text(encoding="utf-8"))
    q = data["graphs"]["did"]["queries"]["eleven percent"]
    dropped = q["spans"].pop(0)                       # v3 now has a span hit v2 "did not"
    q["claims"].append({"id": "claim_invented_missing", "work_id": "work_nowhere"})
    res = parity.check(home, expected=data, write=False)
    assert not res.ok
    keys = {(d.kind, d.key if isinstance(d.key, str) else tuple(d.key)) for d in res.unexplained}
    assert ("claims", "claim_invented_missing") in keys
    assert ("spans", tuple(dropped["key"])) in keys


def test_check_without_capture_or_queries_says_what_to_do(tmp_path):
    with pytest.raises(parity.ParityError, match="queries"):
        parity.load_queries(tmp_path)
    with pytest.raises(parity.ParityError, match="capture"):
        parity.load_expected(tmp_path)


def test_default_queries_are_deterministic_frequent_terms(fx, tmp_path):
    q1 = parity.default_queries(fx.home, per_graph=12)
    q2 = parity.default_queries(fx.home, per_graph=12)
    assert q1 == q2
    assert set(q1) == {"did", "land-value"}
    assert 0 < len(q1["did"]) <= 12
    assert "eleven" in q1["did"] or "eleven percent" in q1["did"]
    assert all(q == q.strip() and q for qs in q1.values() for q in qs)


def test_seeding_queries_never_overwrites_the_researchers_file(fx, tmp_path):
    v3 = tmp_path / "v3"
    path = parity.seed_queries(fx.home, v3, per_graph=5)
    assert path == v3 / parity.QUERIES_FILE
    path.write_text(json.dumps({"did": ["my own query"]}), encoding="utf-8")
    with pytest.raises(parity.ParityError, match="exists"):
        parity.seed_queries(fx.home, v3, per_graph=5)
    assert parity.load_queries(v3) == {"did": ["my own query"]}
    parity.seed_queries(fx.home, v3, per_graph=5, force=True)
    assert parity.load_queries(v3) != {"did": ["my own query"]}


def test_frozen_v2_retrieval_sql_matches_v2_code():
    from seedgraph.semantic.current import CURRENT_CLAIMS_CTE
    from seedgraph.answer import retrieve
    assert parity.V2_CURRENT_CLAIMS_CTE == CURRENT_CLAIMS_CTE
    assert parity.V2_NOT_EXCLUDED == retrieve._NOT_EXCLUDED


def test_a_superseded_lens_run_on_a_merged_paper_is_classified_merged(fx, tmp_path):
    """Two lens runs on the double import's two works (one paper in v3): v2 counts both
    lens claims current, v3 only the newer run's; the older one is a 'merged' miss."""
    home = tmp_path / "v2"
    shutil.copytree(fx.home, home)
    dbl = [fx.works[k] for k in fx.one("double_import").ids["work_keys"]]
    conn = sqlite3.connect(home / "projects" / "second_test" / "project.db")
    with conn:
        for n, (w, at) in enumerate(zip(dbl, ("2026-07-13T01:00:00+00:00", "2026-07-14T01:00:00+00:00"))):
            md_id, md_hash = conn.execute("SELECT markdown_id, markdown_hash FROM extraction_runs WHERE work_id = ? "
                                          "AND lens_id IS NULL", (w.work_id,)).fetchone()
            conn.execute(
                "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, markdown_hash, schema_id, "
                "lens_id, schema_version, prompt_version, model_name, provider, access_mode, run_status, created_at, "
                "lens_definition_hash) SELECT ?, ?, ?, ?, schema_id, lens_id, schema_version, prompt_version, "
                "model_name, provider, access_mode, run_status, ?, lens_definition_hash FROM extraction_runs "
                "WHERE lens_id IS NOT NULL ORDER BY extraction_run_id LIMIT 1",
                (f"extr_testlens_{n}", w.work_id, md_id, md_hash, at))
            conn.execute(
                "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, work_id, claim_type, "
                "field_key, normalized_label, claim_text, status, epistemic_type, assertion_status, confidence, "
                "created_at) VALUES (?,NULL,?,?,'data','plot_layout_v1[0]','plot layout',?,'found','llm_extracted',"
                "'stated',0.8,?)",
                (f"claim_testlens_{n}", f"extr_testlens_{n}", w.work_id,
                 "The made-up sample covers forty sites observed over nine seasons.", at))
            conn.execute("INSERT INTO claim_fts (claim_id, normalized_label, claim_text) SELECT claim_id, "
                         "normalized_label, claim_text FROM extracted_claims WHERE claim_id = ?",
                         (f"claim_testlens_{n}",))
    conn.close()
    v3 = tmp_path / "v3"
    rep.run_and_report(home, v3, run_id="mig-par-lens", now="2026-10-23T05:30:00+00:00",
                       process_lister=NO_PROCS, disk_free=BIG_DISK)
    parity.write_queries(v3, {"did": ["made-up sample"]})
    parity.capture(home, v3)
    res = parity.check(v3)
    assert res.ok, [(d.kind, d.key, d.detail) for d in res.unexplained]
    got = {d.key: d.cls for d in res.differences if d.kind == "claims"}
    assert got.get("claim_testlens_0") == "merged"
    assert "claim_testlens_1" not in got
