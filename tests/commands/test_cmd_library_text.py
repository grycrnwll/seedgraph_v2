"""`seedgraph library textcheck` and `library reconvert --ocr` smoke tests. Invented text."""

from __future__ import annotations

import json

import pytest
from work.routine_support import make_paper

from seedgraph.cli import app
from seedgraph.library import store as libstore

GARBLED = "# Invented\n\n" + " ".join(
    f"Sentence {k} notes a speciÖc invented detail about di§erent loaves." for k in range(60)) + "\n"


@pytest.fixture
def home(isolated_home):
    lib = libstore.open_library_at(isolated_home / "library")
    try:
        clean = make_paper(lib, 1)
        bad = make_paper(lib, 2, text=GARBLED, sections=[])
    finally:
        lib.close()
    (isolated_home / "golden_set.json").write_text(json.dumps([
        {"id": "GS01", "category": "math", "v2_markdown_id": clean.markdown_id},
        {"id": "GS06", "category": "garbled", "v2_markdown_id": bad.markdown_id},
    ]), encoding="utf-8")
    return isolated_home


def test_textcheck_reports_each_papers_verdict(runner, home):
    result = runner.invoke(app, ["library", "textcheck", "--root", str(home), "--json"])
    assert result.exit_code == 0, result.output
    rows = {r["paper_id"]: r["status"] for r in json.loads(result.stdout)}
    assert rows == {"lp_test0001": "pass", "lp_test0002": "garbled"}
    text = runner.invoke(app, ["library", "textcheck", "lp_test0002", "--root", str(home)])
    assert text.exit_code == 0 and "garbled" in text.output and "lp_test0001" not in text.output


def test_textcheck_golden_compares_verdict_and_category(runner, home):
    result = runner.invoke(app, ["library", "textcheck", "--golden", "--root", str(home), "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert [(r["golden_id"], r["status"], r["agrees"]) for r in rows] == [
        ("GS01", "pass", True), ("GS06", "garbled", True)]


def test_textcheck_golden_disagreement_exits_1(runner, home):
    data = json.loads((home / "golden_set.json").read_text(encoding="utf-8"))
    data[0]["category"] = "garbled"
    (home / "golden_set.json").write_text(json.dumps(data), encoding="utf-8")
    result = runner.invoke(app, ["library", "textcheck", "--golden", "--root", str(home)])
    assert result.exit_code == 1
    assert "GS01" in result.output and "disagree" in result.output


def test_textcheck_golden_without_a_manifest_is_refused(runner, home):
    (home / "golden_set.json").unlink()
    result = runner.invoke(app, ["library", "textcheck", "--golden", "--root", str(home)])
    assert result.exit_code == 3


def test_reconvert_ocr_shares_one_backend_across_papers(runner, home, monkeypatch):
    from seedgraph.commands import library as library_cmd
    from seedgraph.library import reconvert

    made, seen = [], []

    def factory():
        made.append(object())
        return made[-1]

    def fake(lib, paper_id, *, backend=None):
        seen.append((paper_id, backend))
        return reconvert.ReconvertResult(paper_id=paper_id, source_file_id="sf_x", status="ocr_did_not_help",
                                         old_markdown_id="md_a", new_markdown_id="md_b", text_quality="garbled",
                                         detail="forced OCR did not help", attention_item_id=None)

    monkeypatch.setattr(library_cmd, "reconvert_backend", factory)
    monkeypatch.setattr(reconvert, "reconvert_ocr", fake)
    result = runner.invoke(app, ["library", "reconvert", "--ocr", "lp_test0001", "lp_test0002",
                                 "--root", str(home), "--json"])
    assert result.exit_code == 0, result.output
    assert [p for p, _ in seen] == ["lp_test0001", "lp_test0002"]
    assert len(made) == 1 and all(b is made[0] for _, b in seen)
    assert [r["status"] for r in json.loads(result.stdout)] == ["ocr_did_not_help"] * 2


def test_reconvert_needs_the_ocr_flag(runner, home):
    result = runner.invoke(app, ["library", "reconvert", "lp_test0001", "--root", str(home)])
    assert result.exit_code == 2
