"""`seedgraph library refetch-metadata [--malformed] [PAPER…]` (docs/V3_UI_DESIGN.md §7): re-runs
the provider lookups through ``metadata_verify``. The chain is a fake; invented text."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.commands import library as library_cmd
from seedgraph.library.store import open_library

from .metadata_support import FakeChain, insert_paper, openalex_record, paper_row

TITLE = "Invented Ovens and the Morning Loaf"
DOI = "10.9999/oven.2031.1"


@pytest.fixture
def home(isolated_home):
    with open_library(isolated_home) as lib:
        insert_paper(lib, "lp_ok", title="A Fine Invented Title", title_source="v2")
        insert_paper(lib, "lp_marked", title=TITLE + "†", title_source="ollama", dois=((DOI, "ollama"),))
        insert_paper(lib, "lp_user", title="Preface", title_source="user")
    return isolated_home


@pytest.fixture
def chain(monkeypatch):
    fake = FakeChain({DOI: openalex_record(DOI, TITLE, 2031)})
    monkeypatch.setattr(library_cmd, "refetch_chain", lambda lib: fake)
    return fake


def run(home, *args):
    return CliRunner().invoke(app, ["library", "refetch-metadata", *args, "--root", str(home)])


def test_malformed_papers_are_refetched(home, chain):
    result = run(home, "--malformed", "--json")
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert [r["paper_id"] for r in rows] == ["lp_marked"]
    assert rows[0]["status"] == "verified"
    with open_library(home) as lib:
        assert tuple(paper_row(lib, "lp_marked"))[:3] == (TITLE, "openalex", 2031)
        assert paper_row(lib, "lp_user")["title"] == "Preface"


def test_named_papers_are_refetched_with_a_table(home, chain):
    result = run(home, "lp_ok", "lp_marked")
    assert result.exit_code == 0, result.output
    assert "lp_ok" in result.output and "no_doi" in result.output and "verified" in result.output


def test_nothing_named_is_refused(home, chain):
    result = run(home)
    assert result.exit_code != 0 and "--malformed" in result.output
