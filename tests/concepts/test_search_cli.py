"""`seedgraph concepts refresh | search | eval-search` (thin adapters). Invented text; no
Ollama: keyword mode and the keyword-only adoption run."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.library import store

from ._embed import index_fts
from ._world import World, a

runner = CliRunner()


def test_refresh_search_and_a_keyword_only_eval(tmp_path):
    home = tmp_path / "home"
    lib = store.open_library_at(home / "library")
    w = World(lib)
    w.note("cl_1", [a("gadget alignment", "Untreated outcome paths move in parallel.")])
    w.note("cl_2", [a("gadget concord", "Outcome paths share a common trend.")])
    index_fts(lib)
    lib.close()

    r = runner.invoke(app, ["concepts", "refresh", "--root", str(home), "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["concepts"] == 2

    r = runner.invoke(app, ["concepts", "search", "outcome paths", "--root", str(home), "--json"])
    assert r.exit_code == 0, r.output
    body = json.loads(r.output)
    assert body["search_mode"] == "keyword" and len(body["hits"]) == 2

    r = runner.invoke(app, ["concepts", "search", "assumption/gadget alignment", "--sense", "--root", str(home)])
    assert r.exit_code == 0 and "assumption/gadget concord" in r.output

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"partners": [{"id": "P1", "a": "gadget alignment", "b": "gadget concord",
                                               "kind": "synonym"}], "sense_probes": []}), encoding="utf-8")
    out = tmp_path / "eval"
    r = runner.invoke(app, ["concepts", "eval-search", "--cases", str(cases), "--out", str(out), "--keyword-only",
                            "--root", str(home)])
    assert r.exit_code == 0, r.output
    assert (out / "protocol.json").is_file() and "| P1 |" in (out / "report.md").read_text(encoding="utf-8")
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["modes"]["keyword"]["partner_ranks"] == \
        {"P1": 1}
