"""`seedgraph migrate run|verify|parity|accept` smoke tests (synthetic v2 home only)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

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


# --- run writes the report -------------------------------------------------------------------

def _run(runner, v2_home, v3, *extra):
    result = runner.invoke(app, ["migrate", "run", "--v2-home", str(v2_home), "--root", str(v3), *extra])
    assert result.exit_code == 0, result.output
    return result


def test_run_writes_the_report_and_prints_its_path(runner, v2_home, tmp_path):
    v3 = tmp_path / "v3"
    result = _run(runner, v2_home, v3, "--json")
    data = json.loads(result.output)
    report = Path(data["report"])
    assert report.name == "report.md" and report.is_file()
    assert report.parent == Path(data["run_dir"])
    assert (report.parent / "attention.md").is_file()
    assert (report.parent / "report.json").is_file()
    text = runner.invoke(app, ["migrate", "run", "--v2-home", str(v2_home), "--root", str(v3)])
    assert text.exit_code == 0, text.output
    assert "report.md" in text.output


def test_run_whose_report_cannot_account_for_everything_exits_1(runner, v2_home, tmp_path, monkeypatch):
    from seedgraph.migrate import report as rep

    real = rep.write_report

    def short(result, **kw):
        files = real(result, **kw)
        raise rep.ReportError("1 v2 row unaccounted", files=files)

    monkeypatch.setattr(rep, "write_report", short)
    result = runner.invoke(app, ["migrate", "run", "--v2-home", str(v2_home), "--root", str(tmp_path / "v3")])
    assert result.exit_code == 1, result.output
    assert "unaccounted" in result.output
    assert "report.md" in result.output


# --- verify ----------------------------------------------------------------------------------

def _rewrite_first_line(dump: Path) -> None:
    with gzip.open(dump, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    lines[0] = "tampered " + lines[0]  # at the start: the printed difference is truncated
    with gzip.open(dump, "wt", encoding="utf-8") as f:
        f.writelines(lines)


def test_verify_compares_the_latest_run_with_the_previous_one(runner, v2_home, tmp_path):
    v3 = tmp_path / "v3"
    first = json.loads(_run(runner, v2_home, v3, "--json").output)["run_id"]
    second = json.loads(_run(runner, v2_home, v3, "--json").output)["run_id"]
    result = runner.invoke(app, ["migrate", "verify", "--root", str(v3)])
    assert result.exit_code == 0, result.output
    assert "identical" in result.output
    assert first in result.output and second in result.output

    _rewrite_first_line(v3 / "migration" / "runs" / second / "dump.jsonl.gz")
    differs = runner.invoke(app, ["migrate", "verify", "--root", str(v3), "--run", second, "--against", first])
    assert differs.exit_code == 1, differs.output
    assert "line 1" in differs.output and "tampered" in differs.output


def test_verify_with_one_run_is_refused(runner, migrated_home):
    result = runner.invoke(app, ["migrate", "verify", "--root", str(migrated_home)])
    assert result.exit_code == 3, result.output
    assert "two" in result.output


# --- parity ----------------------------------------------------------------------------------

@pytest.fixture
def reported_home(runner, v2_home, tmp_path):
    """A home made by `migrate run`, so its run has the report parity reads re-links from."""
    v3 = tmp_path / "reported"
    _run(runner, v2_home, v3)
    return v3


def test_parity_seeds_captures_and_checks(runner, v2_home, reported_home):
    from seedgraph.migrate import parity as par

    result = runner.invoke(app, ["migrate", "parity", "--seed-queries", "--capture", "--check",
                                 "--v2-home", str(v2_home), "--root", str(reported_home)])
    assert result.exit_code == 0, result.output
    assert (reported_home / par.QUERIES_FILE).is_file()
    assert (reported_home / par.EXPECTED_FILE).is_file()
    assert "0 unexplained" in result.output
    assert "parity.md" in result.output


def test_parity_check_exits_1_on_an_unexplained_difference(runner, v2_home, reported_home):
    from seedgraph.migrate import parity as par

    seeded = runner.invoke(app, ["migrate", "parity", "--seed-queries", "--capture",
                                 "--v2-home", str(v2_home), "--root", str(reported_home)])
    assert seeded.exit_code == 0, seeded.output
    path = reported_home / par.EXPECTED_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = next(iter(data["graphs"].values()))
    query = next(iter(graph["queries"].values()))
    query["claims"].append({"id": "claim_not_in_any_library", "work_id": "work_not_in_any_library"})
    path.write_text(json.dumps(data), encoding="utf-8")

    result = runner.invoke(app, ["migrate", "parity", "--check", "--root", str(reported_home)])
    assert result.exit_code == 1, result.output
    assert "1 unexplained" in result.output


def test_parity_without_a_capture_exits_1(runner, reported_home):
    result = runner.invoke(app, ["migrate", "parity", "--check", "--root", str(reported_home)])
    assert result.exit_code == 1, result.output
    assert "error" in result.output


# --- accept ----------------------------------------------------------------------------------

ACCEPT_QUERIES = {"did": ["eleven percent", "responds to"], "land-value": ["eleven percent"]}


def test_accept_passes_and_prints_the_table_once(runner, v2_home, tmp_path):
    from seedgraph.migrate import accept as acc
    from seedgraph.migrate import parity as par

    v3 = tmp_path / "v3"
    par.write_queries(v3, ACCEPT_QUERIES)
    acc.write_expected(v3, totals={"unaccounted": 0, "errors": 0}, attention={},
                       allow_other_attention_kinds=True)
    result = runner.invoke(app, ["migrate", "accept", "--v2-home", str(v2_home), "--root", str(v3)])
    assert result.exit_code == 0, result.output
    assert result.output.count("expected values") == 1
    assert "acceptance.md" in result.output


def test_accept_without_expected_json_exits_1(runner, v2_home, tmp_path):
    from seedgraph.migrate import parity as par

    v3 = tmp_path / "v3"
    par.write_queries(v3, ACCEPT_QUERIES)
    result = runner.invoke(app, ["migrate", "accept", "--v2-home", str(v2_home), "--root", str(v3)])
    assert result.exit_code == 1, result.output
    assert "expected values" in result.output and "FAIL" in result.output
