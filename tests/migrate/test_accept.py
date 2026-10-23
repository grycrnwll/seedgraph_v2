"""The final acceptance run (design §11 "Final real-data acceptance run"), on the
synthetic v2 home: preflight, migrate, identical re-run, v2 guard, parity, and the
report compared with the researcher's ``migration/acceptance/expected.json``."""

from __future__ import annotations

import copy
import json

import pytest

from fixtures import build_v2_home
from seedgraph.migrate import accept as acc
from seedgraph.migrate import parity
from seedgraph.migrate.v2guard import ProcessInfo

NO_PROCS = lambda: []  # noqa: E731
BIG_DISK = lambda p: 10 ** 15  # noqa: E731
QUERIES = {"did": ["eleven percent", "responds to"], "land-value": ["eleven percent"]}
# Expected values for the synthetic fixture, read off its planted manifest (three saved
# answers, one run-scoped answer, one researcher contradiction) and the attention list
# the design asks for (two sides of one shared PDF, one wrong PDF, one malformed title,
# two cached PDFs held by no project, one test-converter text, one contradiction).
EXPECTED_TOTALS = {"answers.exported": 3, "answers.run_scoped_not_carried": 1, "concepts.contradictions": 1,
                   "unaccounted": 0, "errors": 0, "graph.did.member": {"expected": 18, "tolerance": 6}}
EXPECTED_ATTENTION = {"shared_pdf": 2, "pdf_title_miss": 1, "malformed_title": 1, "metadata_needed": 2,
                      "test_converter_markdown": 1, "concept_contradiction": 1,
                      "identity_check": {"expected": 4, "tolerance": 2}}


@pytest.fixture(scope="module")
def v2(tmp_path_factory):
    return build_v2_home(tmp_path_factory.mktemp("v2") / "built").home


def prepared_home(root, *, totals=EXPECTED_TOTALS, attention=EXPECTED_ATTENTION):
    v3 = root / "v3"
    parity.write_queries(v3, QUERIES)
    acc.write_expected(v3, totals=totals, attention=attention)
    return v3


@pytest.fixture(scope="module")
def accepted(v2, tmp_path_factory):
    v3 = prepared_home(tmp_path_factory.mktemp("acc"))
    printed: list[str] = []
    res = acc.accept(v2, v3, run_id="mig-acc-1", now="2026-10-23T06:00:00+00:00", process_lister=NO_PROCS,
                     disk_free=BIG_DISK, echo=printed.append)
    return v3, res, printed


def test_acceptance_passes_on_the_fixture(accepted):
    v3, res, _ = accepted
    assert res.ok, acc.render_table(res.rows)
    assert [r.step for r in res.rows] == list(acc.STEPS)
    assert all(r.status == "pass" for r in res.rows)


def test_second_run_gives_the_same_dump_sha(accepted):
    _, res, _ = accepted
    assert res.outcome is not None and res.rerun is not None
    assert res.rerun.dump_sha256 == res.outcome.result.dump_sha256
    assert res.rerun.swapped is False


def test_one_table_is_printed_and_acceptance_md_written(accepted):
    v3, res, printed = accepted
    assert len(printed) == 1
    table = printed[0]
    for step in acc.STEPS:
        assert step in table
    assert res.path == v3 / "migration" / "runs" / "mig-acc-1" / "acceptance.md"
    text = res.path.read_text(encoding="utf-8")
    assert "PASS" in text and "expected values" in text.lower()


def test_parity_was_captured_and_checked(accepted):
    v3, res, _ = accepted
    assert (v3 / parity.EXPECTED_FILE).is_file()
    assert (v3 / "migration" / "runs" / "mig-acc-1" / "parity.md").is_file()


def test_compare_expected_flags_a_mismatch(accepted):
    v3, res, _ = accepted
    report = json.loads((v3 / "migration" / "runs" / "mig-acc-1" / "report.json").read_text(encoding="utf-8"))
    expected = json.loads((v3 / acc.EXPECTED_FILE).read_text(encoding="utf-8"))
    assert all(c.ok for c in acc.compare_expected(report, expected))
    bad = copy.deepcopy(expected)
    bad["totals"]["answers.exported"] = {"expected": 4, "tolerance": 0}
    bad["totals"]["no.such.key"] = {"expected": 1, "tolerance": 0}
    del bad["attention"]["shared_pdf"]
    failed = {c.name for c in acc.compare_expected(report, bad) if not c.ok}
    assert failed == {"answers.exported", "no.such.key", "attention.shared_pdf"}


def test_tolerance_is_respected():
    report = {"totals": {"x": 10}, "attention": {"by_kind": {}}}
    ok = acc.compare_expected(report, {"totals": {"x": {"expected": 12, "tolerance": 2}}, "attention": {}})
    bad = acc.compare_expected(report, {"totals": {"x": {"expected": 13, "tolerance": 2}}, "attention": {}})
    pct = acc.compare_expected(report, {"totals": {"x": {"expected": 11, "tolerance_pct": 10}}, "attention": {}})
    assert [c.ok for c in ok] == [True] and [c.ok for c in bad] == [False] and [c.ok for c in pct] == [True]


def test_acceptance_fails_on_a_mismatch_in_the_home_file(v2, tmp_path):
    totals = dict(EXPECTED_TOTALS, **{"answers.exported": 5})
    v3 = prepared_home(tmp_path, totals=totals)
    res = acc.accept(v2, v3, run_id="mig-acc-2", now="2026-10-23T07:00:00+00:00", process_lister=NO_PROCS,
                     disk_free=BIG_DISK, echo=None)
    assert not res.ok
    status = {r.step: r.status for r in res.rows}
    assert status["expected values"] == "fail"
    assert status["migrate run"] == "pass" and status["parity"] == "pass"
    assert "answers.exported" in next(r.detail for r in res.rows if r.step == "expected values")


def test_missing_researcher_files_fail_their_steps_with_a_hint(v2, tmp_path):
    v3 = tmp_path / "v3"
    res = acc.accept(v2, v3, run_id="mig-acc-3", now="2026-10-23T08:00:00+00:00", process_lister=NO_PROCS,
                     disk_free=BIG_DISK, echo=None)
    rows = {r.step: r for r in res.rows}
    assert rows["migrate run"].status == "pass" and rows["re-run identical"].status == "pass"
    assert rows["parity"].status == "fail" and "queries" in rows["parity"].detail
    assert rows["expected values"].status == "fail" and "expected.json" in rows["expected values"].detail
    assert not res.ok


def test_a_v2_writer_fails_preflight_and_skips_the_rest(v2, tmp_path):
    v3 = prepared_home(tmp_path)
    writer = lambda: [ProcessInfo(4242, "python -m seedgraph serve --project second_test")]  # noqa: E731
    res = acc.accept(v2, v3, run_id="mig-acc-4", process_lister=writer, disk_free=BIG_DISK, echo=None)
    rows = {r.step: r for r in res.rows}
    assert rows["preflight"].status == "fail" and "v2 writer" in rows["preflight"].detail
    assert all(rows[s].status == "skipped" for s in acc.STEPS[1:])
    assert not (v3 / "library").exists()
    assert res.path.is_file()


def test_starter_expected_file_is_written_from_given_values_and_never_overwritten(tmp_path):
    path = acc.write_expected(tmp_path, totals={"graph.did.member": 7}, attention={"shared_pdf": 1}, tolerance=2)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["totals"]["graph.did.member"] == {"expected": 7, "tolerance": 2}
    assert data["attention"]["shared_pdf"] == {"expected": 1, "tolerance": 2}
    with pytest.raises(acc.AcceptanceError, match="exists"):
        acc.write_expected(tmp_path, totals={}, attention={})
    acc.write_expected(tmp_path, totals={}, attention={}, force=True)


def test_starter_keys_exist_in_a_real_report(accepted):
    v3, _, _ = accepted
    report = json.loads((v3 / "migration" / "runs" / "mig-acc-1" / "report.json").read_text(encoding="utf-8"))
    missing = [k for k in acc.STARTER_TOTALS if k not in report["totals"]]
    assert missing == []
    template = acc.expected_template()
    assert set(template["totals"]) == set(acc.STARTER_TOTALS)
    assert all(v["expected"] is None for v in template["totals"].values())
