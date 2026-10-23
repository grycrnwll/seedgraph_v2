"""`seedgraph library ...` smoke tests on a home migrated from the synthetic v2 fixture."""

from __future__ import annotations

import json
import re

import pytest

from seedgraph.cli import app
from seedgraph.library.graphs import graph_info, membership, open_graph
from seedgraph.library.store import is_dirty, open_library_at, paper_exists


def _lib(home):
    return open_library_at(home / "library")


def _member_of_did(home) -> str:
    with _lib(home) as lib, open_graph(lib, "did") as g:
        held = {r[0] for r in lib.conn.execute("SELECT paper_id FROM paper_status WHERE status = 'held'")}
        return sorted(p.paper_id for p in membership(g, status="member") if p.paper_id in held)[0]


def _invoke(runner, home, *args):
    return runner.invoke(app, ["library", *args, "--root", str(home)])


# --- --root and the missing library -------------------------------------------------------------

def test_library_commands_refuse_when_there_is_no_library(runner, isolated_home):
    result = runner.invoke(app, ["library", "status"])
    assert result.exit_code == 3
    assert "no library" in result.output
    assert not (isolated_home / "library" / "library.db").exists()


def test_status_reads_the_root_home(runner, migrated_home):
    result = _invoke(runner, migrated_home, "status")
    assert result.exit_code == 0, result.output
    assert "papers:" in result.output
    data = json.loads(_invoke(runner, migrated_home, "status", "--json").output)
    with _lib(migrated_home) as lib:
        total = lib.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    assert data["papers"] == total and data["held"] + data["cited_only"] == total
    assert data["graphs"] >= 1 and data["dirty"] is False


def test_read_only_commands_leave_the_library_clean(runner, migrated_home):
    pid = _member_of_did(migrated_home)
    for args in (["status"], ["show", pid], ["search", "the"], ["attention"], ["graphs", "list"],
                 ["graphs", "show", "did"], ["delete", pid], ["prune-replaced"]):
        assert _invoke(runner, migrated_home, *args).exit_code == 0, args
    with _lib(migrated_home) as lib:
        assert not is_dirty(lib.conn)


# --- search, show, check ------------------------------------------------------------------------

def test_search_spans_finds_a_word_of_a_span(runner, migrated_home):
    with _lib(migrated_home) as lib:
        quote = lib.conn.execute("SELECT exact_quote FROM evidence_spans ORDER BY span_id").fetchone()[0]
    word = max(re.findall(r"[A-Za-z]{5,}", quote), key=len)
    result = _invoke(runner, migrated_home, "search", word, "--json")
    assert result.exit_code == 0, result.output
    hits = json.loads(result.output)
    assert hits and all(h["kind"] == "spans" for h in hits)
    assert any(word.lower() in h["text"].lower() for h in hits)


def test_search_scoped_to_an_unknown_graph_is_refused(runner, migrated_home):
    result = _invoke(runner, migrated_home, "search", "anything", "--graph", "no-such-graph")
    assert result.exit_code == 3


def test_search_bad_kind_is_a_usage_error(runner, migrated_home):
    assert _invoke(runner, migrated_home, "search", "x", "--kind", "papers").exit_code == 2


def test_show_a_paper_and_an_unknown_one(runner, migrated_home):
    pid = _member_of_did(migrated_home)
    result = _invoke(runner, migrated_home, "show", pid, "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["paper_id"] == pid and data["status"] == "held"
    assert any(g["slug"] == "did" for g in data["graphs"])
    assert _invoke(runner, migrated_home, "show", "lp_nope").exit_code == 3


def test_check_one_paper_and_all_held(runner, migrated_home):
    pid = _member_of_did(migrated_home)
    one = _invoke(runner, migrated_home, "check", pid)
    assert one.exit_code == 0, one.output
    assert pid in one.output
    every = _invoke(runner, migrated_home, "check", "--json")
    assert every.exit_code == 0, every.output
    rows = json.loads(every.output)
    with _lib(migrated_home) as lib:
        held = lib.conn.execute("SELECT COUNT(*) FROM paper_status WHERE status = 'held'").fetchone()[0]
    assert len({r["paper_id"] for r in rows}) == held
    assert {r["status"] for r in rows} <= {"pass", "weak", "miss", "unknown"}


# --- attention ----------------------------------------------------------------------------------

def test_attention_lists_and_resolves(runner, migrated_home):
    listed = _invoke(runner, migrated_home, "attention", "--json")
    assert listed.exit_code == 0, listed.output
    items = json.loads(listed.output)
    assert items and all(i["status"] == "open" for i in items)
    item_id = items[0]["item_id"]
    resolved = _invoke(runner, migrated_home, "attention", "--resolve", item_id, "--resolution", "checked by hand")
    assert resolved.exit_code == 0, resolved.output
    after = json.loads(_invoke(runner, migrated_home, "attention", "--status", "resolved", "--json").output)
    assert [i["item_id"] for i in after] == [item_id]
    assert _invoke(runner, migrated_home, "attention", "--resolve", "att_nope", "--resolution", "x").exit_code == 3
    assert _invoke(runner, migrated_home, "attention", "--resolve", item_id).exit_code == 2


# --- remove, delete -----------------------------------------------------------------------------

def test_remove_takes_the_paper_out_of_one_graph_only(runner, migrated_home):
    pid = _member_of_did(migrated_home)
    result = _invoke(runner, migrated_home, "remove", pid, "--graph", "did")
    assert result.exit_code == 0, result.output
    with _lib(migrated_home) as lib:
        assert paper_exists(lib.conn, pid)
        with open_graph(lib, "did") as g:
            assert pid not in {p.paper_id for p in membership(g)}


def test_delete_without_yes_shows_the_plan_and_deletes_nothing(runner, migrated_home):
    pid = _member_of_did(migrated_home)
    result = _invoke(runner, migrated_home, "delete", pid)
    assert result.exit_code == 0, result.output
    assert "did" in result.output
    assert "nothing deleted" in result.output
    with _lib(migrated_home) as lib:
        assert paper_exists(lib.conn, pid)
        with open_graph(lib, "did") as g:
            assert pid in {p.paper_id for p in membership(g)}


def test_delete_with_yes_deletes(runner, migrated_home):
    pid = _member_of_did(migrated_home)
    result = _invoke(runner, migrated_home, "delete", pid, "--yes")
    assert result.exit_code == 0, result.output
    with _lib(migrated_home) as lib:
        assert not paper_exists(lib.conn, pid)
    assert _invoke(runner, migrated_home, "delete", pid, "--yes").exit_code == 3


# --- refetch-metadata, prune-replaced ------------------------------------------------------------

def test_refetch_metadata_is_not_available_yet(runner, migrated_home):
    result = _invoke(runner, migrated_home, "refetch-metadata", "--malformed")
    assert result.exit_code == 4
    assert "stage 2" in result.output


def test_prune_replaced_needs_yes(runner, migrated_home):
    old = migrated_home / "migration" / "replaced" / "20260101T000000Z-mig-old"
    (old / "library").mkdir(parents=True)
    (old / "library" / "library.db").write_bytes(b"x" * 10)
    listed = _invoke(runner, migrated_home, "prune-replaced")
    assert listed.exit_code == 0, listed.output
    assert old.name in listed.output and old.is_dir()
    pruned = _invoke(runner, migrated_home, "prune-replaced", "--yes")
    assert pruned.exit_code == 0, pruned.output
    assert not old.exists()
    assert (migrated_home / "library" / "library.db").is_file()


# --- graphs -------------------------------------------------------------------------------------

def test_graphs_list_show_rename(runner, migrated_home):
    listed = json.loads(_invoke(runner, migrated_home, "graphs", "list", "--json").output)
    assert "did" in {g["slug"] for g in listed}
    shown = _invoke(runner, migrated_home, "graphs", "show", "did", "--json")
    assert shown.exit_code == 0, shown.output
    data = json.loads(shown.output)
    assert data["slug"] == "did" and data["counts"]["member"] >= 1
    assert _invoke(runner, migrated_home, "graphs", "show", "nope").exit_code == 3
    renamed = _invoke(runner, migrated_home, "graphs", "rename", "did", "Event studies")
    assert renamed.exit_code == 0, renamed.output
    with _lib(migrated_home) as lib, open_graph(lib, "did") as g:
        assert graph_info(g).display_name == "Event studies"


@pytest.mark.parametrize("args", [["--help"], ["graphs", "--help"]])
def test_help_says_context_graph_not_project(runner, args):
    result = runner.invoke(app, ["library", *args])
    assert result.exit_code == 0
    assert "project" not in result.output.lower()
