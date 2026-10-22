"""`seedgraph-v3 golden init|check` end to end on a synthetic v2 home."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from _golden_fixtures import FakeV2, synthetic_markdown
from seedgraph.cli import app
from seedgraph.conversion import set_pool
from support.converter import fake_pool, synthetic_pdf

runner = CliRunner()
QUOTES = ["alpha beta", "gamma delta"]


def _init(tmp_path):
    fake = FakeV2(tmp_path / "v2home")
    md = synthetic_markdown(3, QUOTES)
    p = fake.add_paper(markdown=md, quotes=QUOTES)
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps([{"id": "GS01", "category": "math", "md": p["md_prefix"],
                                 "pdf": p["pdf_prefix"], "project": p["slug"]}]))
    home = tmp_path / "v3home"
    res = runner.invoke(app, ["golden", "init", str(seed), "--root", str(home), "--v2-home", str(fake.home)])
    return fake, md, home, res


def test_init_writes_manifest(tmp_path):
    _, _, home, res = _init(tmp_path)
    assert res.exit_code == 0, res.output
    entries = json.loads((home / "golden_set.json").read_text(encoding="utf-8"))
    assert [e["id"] for e in entries] == ["GS01"]


def test_init_refuses_overwrite_without_force(tmp_path):
    fake, _, home, _ = _init(tmp_path)
    res = runner.invoke(app, ["golden", "init", str(tmp_path / "seed.json"), "--root", str(home),
                              "--v2-home", str(fake.home)])
    assert res.exit_code == 2 and "exists" in res.output


def test_init_bad_prefix_exits_nonzero(tmp_path):
    fake = FakeV2(tmp_path / "v2home")
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps([{"id": "GS01", "category": "x", "md": "abcdef012345", "project": "proj"}]))
    res = runner.invoke(app, ["golden", "init", str(seed), "--root", str(tmp_path / "h"), "--v2-home", str(fake.home)])
    assert res.exit_code == 2 and "GS01" in res.output


def test_check_identity_passes_and_writes_report(tmp_path):
    _, _, home, _ = _init(tmp_path)
    res = runner.invoke(app, ["golden", "check", "--old", "v2-stored", "--new", "v2-stored", "--root", str(home)])
    assert res.exit_code == 0, res.output
    assert "GS01" in res.output
    [rdir] = list((home / "golden_reports").iterdir())
    assert (rdir / "report.json").exists() and (rdir / "report.md").exists()


def test_check_failure_gives_nonzero_exit(tmp_path):
    _, md, home, _ = _init(tmp_path)
    d = tmp_path / "newmd"
    d.mkdir()
    (d / "GS01.md").write_text(md.replace("## References", "## Nothing"), encoding="utf-8")
    res = runner.invoke(app, ["golden", "check", "--old", "v2-stored", "--new", str(d), "--root", str(home)])
    assert res.exit_code == 1, res.output
    assert "FAIL" in res.output


def test_check_unknown_id_exits_2(tmp_path):
    _, _, home, _ = _init(tmp_path)
    res = runner.invoke(app, ["golden", "check", "--new", "v2-stored", "--ids", "GS77", "--root", str(home)])
    assert res.exit_code == 2 and "GS77" in res.output


def test_check_new_convert_uses_shared_pool_and_shuts_it_down(tmp_path):
    fake, _, home, _ = _init(tmp_path)
    [entry] = json.loads((home / "golden_set.json").read_text(encoding="utf-8"))
    pdf_dir = fake.home / "cache" / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    synthetic_pdf(pdf_dir, f"{entry['pdf_sha256']}.pdf", pages=3)
    pool = fake_pool()
    set_pool(pool)  # conftest restores (and shuts down) the previous pool afterwards

    res = runner.invoke(app, ["golden", "check", "--old", "convert", "--new", "convert", "--root", str(home)])

    assert res.exit_code == 0, res.output
    [rdir] = list((home / "golden_reports").iterdir())
    report = json.loads((rdir / "report.json").read_text(encoding="utf-8"))
    assert report["meta"]["new"] == "convert"
    assert [c["name"] for c in report["results"][0]["checks"]] != ["source"]
    assert report["meta"]["converter_load_seconds"]  # one entry per converter that started
    assert report["meta"]["wall_seconds"] >= 0
    assert pool.running_count() == 0  # the check shut the pool down when it finished
