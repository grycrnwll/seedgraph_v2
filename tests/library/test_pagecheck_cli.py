"""``seedgraph library pagecheck`` (design §3.4): a thin, read-only adapter over
``library.pagecheck``. Synthetic PDFs and invented text only."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.library.store import open_library_at

from .test_pages import PAGES, make_paper, paged_md, save_quotes


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    lib = open_library_at(h / "library")
    try:
        pid, _, md = make_paper(lib, tmp_path, paged_md(PAGES))
        save_quotes(lib, pid, md, ["Its ovens warm slowly.", "The index falls in spring."])
    finally:
        lib.close()
    return h


def test_pagecheck_prints_the_tiers_and_writes_the_report(home):
    result = CliRunner().invoke(app, ["library", "pagecheck", "--set", "current", "--root", str(home),
                                      "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["totals"]["page_assigned"] == 2 and data["totals"]["verified"] == 2
    assert data["verdict"] == "pass"
    [run_dir] = list((home / "ui" / "pagecheck").iterdir())
    assert data["report_dir"] == str(run_dir) and (run_dir / "report.md").is_file()


def test_pagecheck_text_output_without_a_report(home):
    result = CliRunner().invoke(app, ["library", "pagecheck", "--set", "current", "--no-report",
                                      "--no-verify", "--root", str(home)])

    assert result.exit_code == 0, result.output
    assert "page assigned" in result.output and "verdict: pass" in result.output
    assert not (home / "ui" / "pagecheck").exists()


def test_pagecheck_refuses_prototype_mode_on_the_current_set(home):
    result = CliRunner().invoke(app, ["library", "pagecheck", "--set", "current", "--mode", "prototype",
                                      "--root", str(home)])

    assert result.exit_code != 0
    assert "prototype set" in result.output


def test_pagecheck_refuses_a_home_without_a_library(tmp_path):
    result = CliRunner().invoke(app, ["library", "pagecheck", "--set", "current",
                                      "--root", str(tmp_path / "nohome")])

    assert result.exit_code != 0
    assert not (tmp_path / "nohome" / "library" / "library.db").exists()
