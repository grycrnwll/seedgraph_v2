"""CLI adapters for the concept tasks: ``work create ... --option KEY=VALUE`` and
``concepts estimate | check`` (read-only). Invented text; every home is under pytest's tmp."""

from __future__ import annotations

import json

import pytest
import typer
from concepts._world import World, a  # tests/ is on sys.path (rootdir conftest)
from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.commands import concepts_work
from seedgraph.commands.work import parse_options
from seedgraph.library import store as libstore

runner = CliRunner()


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    lib = libstore.open_library_at(h / "library")
    w = World(lib)
    w.note("lp_p1", [a("sprocket independence", "Invented: cog lever flange pivots freely."),
                     a("gadget overlap", "Invented: every gadget may get the widget.")])
    w.note("lp_p2", [a("sprocket exclusion", "Invented: cog lever flange stays put."),
                     a("gadget overlap rule", "Invented: each gadget could get the widget.")])
    lib.close()
    return h


@pytest.fixture
def concepts_cli():
    local = typer.Typer()
    concepts_work.register(local)

    @local.callback()
    def _root() -> None:  # keep the two commands as subcommands
        pass

    return local


def test_parse_options_reads_json_values_else_text():
    assert parse_options(["k=3", "families=assumption,method", "budget_tokens=200000", "flag=true"]) == {
        "k": 3, "families": "assumption,method", "budget_tokens": 200000, "flag": True}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_options(["families"])


def test_work_create_passes_task_options(home):
    r = runner.invoke(app, ["work", "create", "concepts:bridge", "--option", "families=assumption",
                            "--option", "k=2", "--root", str(home)])
    assert r.exit_code == 0, r.output
    body = json.loads(r.stdout)
    assert body["options"]["families"] == "assumption" and body["options"]["k"] == 2
    assert body["items"][0]["subject_kind"] == "concept_package"
    r = runner.invoke(app, ["work", "create", "concepts:bridge", "--option", "colour=red", "--root", str(home)])
    assert r.exit_code == 3 and "unknown batch option" in r.stdout


def test_concepts_estimate_is_read_only_and_matches_the_library(home, concepts_cli):
    before = (home / "library" / "library.db").read_bytes()
    r = runner.invoke(concepts_cli, ["estimate", "bridge", "--family", "assumption", "--root", str(home), "--json"])
    assert r.exit_code == 0, r.output
    body = json.loads(r.stdout)
    assert body["task"] == "concepts:bridge" and body["pairs"] >= 1 and body["packages"] >= 1
    assert body["agent_tokens_low"] < body["agent_tokens_high"]
    r = runner.invoke(concepts_cli, ["estimate", "merge", "--budget-tokens", "1", "--root", str(home)])
    assert r.exit_code == 0 and "budget: the first 0 of" in r.stdout
    assert (home / "library" / "library.db").read_bytes() == before  # nothing written
    r = runner.invoke(concepts_cli, ["estimate", "bridge", "--family", "widgets", "--root", str(home)])
    assert r.exit_code == 3


def test_concepts_check_reports_counts(home, concepts_cli):
    r = runner.invoke(concepts_cli, ["check", "--root", str(home), "--json"])
    assert r.exit_code == 0, r.output
    body = json.loads(r.stdout)
    assert body["ok"] is True and body["violations"]["spans_families"] == 0
