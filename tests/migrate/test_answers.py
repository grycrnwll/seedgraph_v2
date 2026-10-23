"""v2 answers export (design §6.11): every saved answer of a carried project becomes
markdown under ``v2_answers/<slug>/``, dumped and swapped with the library."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fixtures import build_v2_home
from seedgraph.migrate import answers
from seedgraph.migrate import run_migration

NO_PROCS = lambda: []  # noqa: E731
BIG_DISK = lambda p: 10 ** 15  # noqa: E731


def run(v2, v3, **kw):
    kw.setdefault("process_lister", NO_PROCS)
    kw.setdefault("disk_free", BIG_DISK)
    return run_migration(v2, v3, **kw)


def tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_v2_home(tmp_path_factory.mktemp("v2") / "home")


@pytest.fixture(scope="module")
def migrated(fx, tmp_path_factory):
    v3 = tmp_path_factory.mktemp("v3") / "home"
    r = run(fx.home, v3, run_id="mig-ans-1", now="2026-10-23T01:00:00+00:00", stage_hook=answers.stage_hook)
    return v3, r


def test_every_saved_answer_is_exported_and_run_scoped_ones_are_not(fx, migrated):
    v3, r = migrated
    assert r.ok and r.swapped
    out = v3 / "v2_answers"
    for item in fx.by_kind("saved_answer"):
        md = v3 / item.expect["export"]
        assert md.is_file(), item.expect["export"]
        text = md.read_text(encoding="utf-8")
        assert item.ids["answer_id"] in text
    run_scoped = fx.one("run_scoped_answer").ids["answer_id"]
    assert not any(run_scoped in p.name for p in out.rglob("*"))
    assert run_scoped not in (out / "index.md").read_text(encoding="utf-8")


def test_answer_markdown_holds_question_answer_trace_and_original_json(fx, migrated):
    v3, _ = migrated
    item = next(i for i in fx.by_kind("saved_answer") if i.ids["slug"] == "second_test")
    text = (v3 / item.expect["export"]).read_text(encoding="utf-8")
    assert "How does pruning frequency relate to orchard yield?" in text
    assert "orchard yield moves with pruning frequency" in text
    assert "2026-07-15T10:00:00+00:00" in text          # the trace's created_at
    assert "```json" in text and '"answer_category": "source_grounded"' in text
    assert "\r\n" not in (v3 / item.expect["export"]).read_bytes().decode("utf-8")


def test_cited_works_are_named_by_their_library_paper(fx, migrated):
    v3, r = migrated
    item = next(i for i in fx.by_kind("saved_answer") if i.ids["slug"] == "second_test"
                and "orchard" in (v3 / i.expect["export"]).read_text(encoding="utf-8"))
    overlap = fx.works["overlap@second_test"]
    paper_id = r.plan.ledger.get("second_test", "works", overlap.work_id).v3_id
    assert paper_id in (v3 / item.expect["export"]).read_text(encoding="utf-8")


def test_index_is_sorted_by_answer_time(fx, migrated):
    v3, _ = migrated
    index = (v3 / "v2_answers" / "index.md").read_text(encoding="utf-8")
    ids = {i.ids["answer_id"]: i.ids["slug"] for i in fx.by_kind("saved_answer")}
    positions = sorted((index.index(a), a) for a in ids)
    # test_proj's answer (07-04) comes before second_test's (07-15, 07-16)
    assert [ids[a] for _, a in positions] == ["test_proj", "second_test", "second_test"]


def test_export_is_deterministic_and_in_the_dump(fx, migrated, tmp_path):
    v3, r1 = migrated
    other = tmp_path / "v3b"
    r2 = run(fx.home, other, run_id="mig-ans-2", now="2026-10-24T05:00:00+00:00", stage_hook=answers.stage_hook)
    assert tree(v3 / "v2_answers") == tree(other / "v2_answers")
    assert r1.dump_sha256 == r2.dump_sha256
    # the answers are part of the dump: a run without the hook differs
    r3 = run(fx.home, tmp_path / "v3c", run_id="mig-ans-3", now="2026-10-24T05:00:00+00:00")
    assert r3.dump_sha256 != r1.dump_sha256


def test_export_answers_reports_what_it_wrote(fx, tmp_path):
    from seedgraph.migrate.snapshot import take_snapshot
    snap = take_snapshot(fx.home, now="t")
    projects = {s: ("not_carried" if s.startswith("dbg_") else "library_only") for s in snap.projects}
    res = answers.export_answers(fx.home, snap.answer_files, tmp_path / "v2_answers", projects=projects)
    assert sorted((e.slug, e.answer_id) for e in res.exported) == sorted(
        (i.ids["slug"], i.ids["answer_id"]) for i in fx.by_kind("saved_answer"))
    assert all((tmp_path / e.path).is_file() for e in res.exported)
    assert (tmp_path / "v2_answers" / "index.md").is_file()


def test_empty_export_still_writes_an_index(tmp_path):
    res = answers.export_answers(tmp_path, {}, tmp_path / "out" / "v2_answers", projects={})
    assert res.exported == []
    assert (tmp_path / "out" / "v2_answers" / "index.md").is_file()
