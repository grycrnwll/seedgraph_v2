"""`seedgraph library reassign-pdf PAPER --to TARGET | --new-paper [--yes]`: shows the plan;
--yes moves the PDF. Invented text only."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner
from web.paper_support import text_pdf

from seedgraph.cli import app
from seedgraph.library import files
from seedgraph.library.store import is_held, open_library

from .metadata_support import insert_paper

TGT = "Harbour Crane Productivity Under Invented Tides"


@pytest.fixture
def home(isolated_home, tmp_path):
    with open_library(isolated_home) as lib:
        insert_paper(lib, "lp_src", title="Seasonal Patterns in Invented Lighthouse Maintenance")
        insert_paper(lib, "lp_tgt", title=TGT)
        files.attach_pdf(lib, "lp_src", text_pdf(tmp_path / "w.pdf", [f"{TGT}. Ada Inventor. 2029.", "x"]))
    return isolated_home


def run(home, *args):
    return CliRunner().invoke(app, ["library", "reassign-pdf", *args, "--root", str(home)])


def held(home, pid):
    with open_library(home) as lib:
        return is_held(lib.conn, pid)


def test_without_yes_the_plan_and_suggestions_are_shown(home):
    r = run(home, "lp_src", "--to", "lp_tgt")
    assert r.exit_code == 0, r.output
    assert "move the PDF of lp_src to lp_tgt" in r.output and "re-run with --yes" in r.output
    assert held(home, "lp_src") and not held(home, "lp_tgt")


def test_yes_moves_the_pdf(home):
    r = run(home, "lp_src", "--to", "lp_tgt", "--yes", "--json")
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    assert out["plan"]["receiving_paper_id"] == "lp_tgt" and out["moved"]["receiving_paper_id"] == "lp_tgt"
    assert held(home, "lp_tgt") and not held(home, "lp_src")


def test_new_paper(home):
    r = run(home, "lp_src", "--new-paper", "--yes")
    assert r.exit_code == 0, r.output
    assert "moved the PDF of lp_src to a new paper" in r.output
    assert not held(home, "lp_src")


def test_naming_both_or_neither_is_refused(home):
    assert run(home, "lp_src").exit_code != 0
    assert run(home, "lp_src", "--to", "lp_tgt", "--new-paper").exit_code != 0


def test_suggestions_are_listed_when_no_target_is_named(home):
    r = run(home, "lp_src", "--suggest")
    assert r.exit_code == 0, r.output
    assert "lp_tgt" in r.output and "1.00" in r.output
