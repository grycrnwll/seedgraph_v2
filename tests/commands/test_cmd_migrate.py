"""`seedgraph migrate run|verify|parity|accept` smoke tests (synthetic v2 home only)."""

from __future__ import annotations

import json
import sys

import pytest

from seedgraph.cli import app
from seedgraph.library.graphs import graph_info, open_graph
from seedgraph.library.store import is_dirty, open_library_at

pytestmark = pytest.mark.usefixtures("no_preflight_probes")


def test_run_builds_the_library_in_root(runner, v2_home, tmp_path):
    v3 = tmp_path / "v3"
    result = runner.invoke(app, ["migrate", "run", "--v2-home", str(v2_home), "--root", str(v3)])
    assert result.exit_code == 0, result.output
    assert (v3 / "library" / "library.db").is_file()
    assert "swapped" in result.output
    with open_library_at(v3 / "library") as lib:
        assert not is_dirty(lib.conn)


def test_run_json_and_graph_names(runner, v2_home, tmp_path):
    v3 = tmp_path / "v3"
    result = runner.invoke(app, ["migrate", "run", "--v2-home", str(v2_home), "--root", str(v3), "--json",
                                 "--graph-name", "did=Diff, in Diff", "--graph-name", "land-value=Land Value"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["ok"] is True and data["swapped"] is True
    assert data["run_id"].startswith("mig-")
    assert "did" in data["graph_counts"]
    with open_library_at(v3 / "library") as lib, open_graph(lib, "did") as g:
        assert graph_info(g).display_name == "Diff, in Diff"


def test_bad_graph_name_is_a_usage_error(runner, v2_home, tmp_path):
    result = runner.invoke(app, ["migrate", "run", "--v2-home", str(v2_home), "--root", str(tmp_path / "v3"),
                                 "--graph-name", "no-equals-sign"])
    assert result.exit_code == 2


def test_missing_v2_home_is_refused(runner, tmp_path):
    result = runner.invoke(app, ["migrate", "run", "--v2-home", str(tmp_path / "nope"),
                                 "--root", str(tmp_path / "v3")])
    assert result.exit_code == 3
    assert "no v2 home" in result.output


def test_a_dirty_library_is_refused_until_replace_dirty(runner, v2_home, migrated_home):
    # `library check` writes (it stores the recheck), which marks the library dirty.
    check = runner.invoke(app, ["library", "check", "--root", str(migrated_home)])
    assert check.exit_code == 0, check.output
    with open_library_at(migrated_home / "library") as lib:
        assert is_dirty(lib.conn)

    refused = runner.invoke(app, ["migrate", "run", "--v2-home", str(v2_home), "--root", str(migrated_home)])
    assert refused.exit_code == 3, refused.output
    replaced = runner.invoke(app, ["migrate", "run", "--v2-home", str(v2_home), "--root", str(migrated_home),
                                   "--replace-dirty"])
    assert replaced.exit_code == 0, replaced.output


@pytest.mark.parametrize("verb, module", [("verify", "seedgraph.migrate.report"),
                                          ("parity", "seedgraph.migrate.parity"),
                                          ("accept", "seedgraph.migrate.accept")])
def test_s7_verbs_say_not_available_when_their_module_is_missing(runner, monkeypatch, tmp_path, v2_home,
                                                                 verb, module):
    monkeypatch.setitem(sys.modules, module, None)  # makes `import` raise ImportError
    result = runner.invoke(app, ["migrate", verb, "--v2-home", str(v2_home), "--root", str(tmp_path / "v3")])
    assert result.exit_code == 4, result.output
    assert "not available yet" in result.output
