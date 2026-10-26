"""`seedgraph library merge KEEP OTHER [--primary SF] [--yes]`: shows the plan; --yes merges."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.library.store import open_library, paper_exists

from .metadata_support import insert_paper


@pytest.fixture
def home(isolated_home):
    with open_library(isolated_home) as lib:
        insert_paper(lib, "lp_keep", title="An Invented Treatise", dois=(("10.9999/keep.1", "v2"),))
        insert_paper(lib, "lp_dup", title="An invented treatise")
    return isolated_home


def run(home, *args):
    return CliRunner().invoke(app, ["library", "merge", *args, "--root", str(home)])


def exists(home, pid):
    with open_library(home) as lib:
        return paper_exists(lib.conn, pid)


def test_without_yes_the_plan_is_shown_and_nothing_changes(home):
    r = run(home, "lp_keep", "lp_dup")
    assert r.exit_code == 0, r.output
    assert "merge lp_dup into lp_keep" in r.output and "re-run with --yes" in r.output
    assert exists(home, "lp_dup")


def test_yes_merges(home):
    r = run(home, "lp_keep", "lp_dup", "--yes", "--json")
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    assert out["plan"]["keep_id"] == "lp_keep" and out["merged"]["other_id"] == "lp_dup"
    assert not exists(home, "lp_dup") and exists(home, "lp_keep")


def test_a_blocked_merge_fails_and_names_the_blocker(home):
    with open_library(home) as lib, lib.write() as conn:
        conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value) VALUES ('lp_dup', 'doi', '10.9999/dup.1')")
    r = run(home, "lp_keep", "lp_dup", "--yes")
    assert r.exit_code != 0
    assert "two ids of one type cannot be merged" in r.output
    assert exists(home, "lp_dup")
