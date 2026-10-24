"""`seedgraph work ...` smoke tests: the CLI mirror of the protocol (design §10.3).

One command per call, JSON on stdout, errors as a JSON body with a non-zero exit.
Invented text only; every home is under pytest's tmp.
"""

from __future__ import annotations

import json

import pytest
from work.routine_support import good_note, make_paper

from seedgraph.cli import app
from seedgraph.library import store as libstore


@pytest.fixture
def home(isolated_home):
    lib = libstore.open_library_at(isolated_home / "library")
    try:
        make_paper(lib, 1, title="Invented gadget subsidies and μ")
        make_paper(lib, 2)
    finally:
        lib.close()
    return isolated_home


def work(runner, home, *args, code: int = 0) -> dict:
    result = runner.invoke(app, ["work", *args, "--root", str(home)])
    assert result.exit_code == code, result.output
    return json.loads(result.stdout)


def create(runner, home, *extra) -> str:
    data = work(runner, home, "create", "notes", "--paper", "lp_test0001", "--paper", "lp_test0002", *extra)
    return data["batch_id"]


def test_create_claim_submit_round_trip(runner, home, tmp_path):
    batch = create(runner, home)
    claimed = work(runner, home, "claim", batch, "--worker", "w1", "--client", "claude")
    assert claimed["status"] == "claimed"
    pkg = claimed["package"]
    assert pkg["paper_id"] == "lp_test0001" and "μ" in pkg["title"]  # UTF-8, not \u escapes

    note = tmp_path / "note.json"
    note.write_text(json.dumps(good_note()), encoding="utf-8")
    usage = tmp_path / "usage.json"
    usage.write_text(json.dumps({"input_tokens": 10, "output_tokens": 5}), encoding="utf-8")
    submitted = work(runner, home, "submit", pkg["packet_id"], "--claim", pkg["claim_id"],
                     "--note-file", str(note), "--model", "invented-model", "--usage-file", str(usage))
    assert submitted["status"] == "accepted" and submitted["paper_status"] == "done"

    status = work(runner, home, "status", batch)
    assert status["batch_id"] == batch and status["item_counts"]["done"] == 1


def test_create_with_options_and_a_query(runner, home):
    data = work(runner, home, "create", "notes", "--paper", "lp_test0001", "--ttl", "60",
                "--split-threshold", "50000", "--replace")
    assert data["options"]["claim_ttl_s"] == 60 and data["options"]["replace"] is True
    assert data["options"]["split_threshold_tokens"] == 50000


def test_a_wrong_claim_is_a_json_refusal_with_its_own_exit_code(runner, home, tmp_path):
    batch = create(runner, home)
    pkg = work(runner, home, "claim", batch, "--worker", "w1")["package"]
    note = tmp_path / "note.json"
    note.write_text(json.dumps(good_note()), encoding="utf-8")
    err = work(runner, home, "submit", pkg["packet_id"], "--claim", "pc_not_mine", "--note-file", str(note),
               code=5)
    assert err["error"]["code"] == "claim_unknown" and err["error"]["next_action"]


def test_an_unknown_batch_is_refused(runner, home):
    err = work(runner, home, "status", "wb_nope", code=3)
    assert err["error"]["code"] == "work_refused"


def test_no_library_is_refused_as_json(runner, tmp_path):
    err = work(runner, tmp_path / "empty", "list", code=3)
    assert err["error"]["code"] == "library_not_found"


def test_release_stop_continue_limit_list_and_command(runner, home):
    batch = create(runner, home)
    pkg = work(runner, home, "claim", batch, "--worker", "w1")["package"]
    released = work(runner, home, "release", pkg["packet_id"], "--claim", pkg["claim_id"],
                    "--reason", "paper_problem", "--message", "the invented scan is blank")
    assert released["item_counts"]["failed"] == 1

    assert work(runner, home, "stop", batch, "--mode", "now")["state"] == "stopped"
    assert work(runner, home, "continue", batch)["state"] != "stopped"
    limited = work(runner, home, "limit", batch, "--resets-at", "2031-01-01T00:00:00+00:00",
                   "--message", "invented limit", "--worker", "w1")
    assert limited["state"] == "usage_limit" and limited["resets_at"].startswith("2031")
    listed = work(runner, home, "list", "--active")
    assert [b["batch_id"] for b in listed["batches"]] == [batch]
    cmds = work(runner, home, "command", batch)
    assert batch in cmds["claude_code"] and batch in cmds["codex"]


def test_bad_release_reason_is_a_usage_error(runner, home):
    result = runner.invoke(app, ["work", "release", "pk_x", "--claim", "c", "--reason", "bored",
                                 "--root", str(home)])
    assert result.exit_code == 2


def test_ollama_runs_the_worker_loop(runner, home, monkeypatch):
    from seedgraph.work import ollama_worker

    calls = {}

    def fake_run(lib, batch_id, *, model, max_packages=None, **kw):
        calls.update(batch_id=batch_id, model=model, max_packages=max_packages)
        return ollama_worker.OllamaRunReport(batch_id=batch_id, worker_id="ollama-test", model=model)

    monkeypatch.setattr(ollama_worker, "run_ollama_worker", fake_run)
    batch = work(runner, home, "create", "metadata", "--paper", "lp_test0001")["batch_id"]
    report = work(runner, home, "ollama", batch, "--model", "invented:1b", "--max", "3")
    assert calls == {"batch_id": batch, "model": "invented:1b", "max_packages": 3}
    assert report["batch_id"] == batch
