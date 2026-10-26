"""`seedgraph setup` smoke tests."""

from __future__ import annotations

import json

import pytest

from seedgraph.cli import app
from seedgraph.library.store import is_dirty, open_library


@pytest.fixture(autouse=True)
def _no_gpu_probe(monkeypatch):
    from seedgraph.conversion import pool

    monkeypatch.setattr(pool, "_nvidia_smi_gpu_count", lambda: 2)


def test_setup_creates_the_home_and_library_under_root(runner, tmp_path):
    home = tmp_path / "fresh"
    result = runner.invoke(app, ["setup", "--root", str(home)])
    assert result.exit_code == 0, result.output
    assert (home / "library" / "library.db").is_file()
    assert (home / "graphs").is_dir()
    assert "created" in result.output
    assert "gpus: 2" in result.output


def test_setup_is_idempotent_and_leaves_the_library_clean(runner, tmp_path):
    home = tmp_path / "fresh"
    assert runner.invoke(app, ["setup", "--root", str(home)]).exit_code == 0
    again = runner.invoke(app, ["setup", "--root", str(home)])
    assert again.exit_code == 0, again.output
    assert "already exists" in again.output
    with open_library(home) as lib:
        assert not is_dirty(lib.conn)


def test_setup_json(runner, tmp_path):
    home = tmp_path / "fresh"
    result = runner.invoke(app, ["setup", "--root", str(home), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["library_created"] is True
    assert data["gpus"] == 2
    assert data["home"] == str(home)


def test_setup_without_root_uses_seedgraph_home(runner, isolated_home):
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0, result.output
    assert (isolated_home / "library" / "library.db").is_file()


def test_setup_stamps_a_new_library_origin_setup_and_keeps_it(runner, tmp_path):
    from seedgraph.library.store import read_meta

    home = tmp_path / "fresh"
    assert runner.invoke(app, ["setup", "--root", str(home)]).exit_code == 0
    assert runner.invoke(app, ["setup", "--root", str(home)]).exit_code == 0
    with open_library(home) as lib:
        meta = read_meta(lib.conn)
    assert meta["origin"] == "setup" and meta["dirty"] == "0"


# The `doctor` command is gone (design §1.3); its two checks that still matter in v3
# moved here: which scholarly-API variables are set, and an opt-in Ollama probe.

def test_setup_lists_scholarly_variables_by_name_never_by_value(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "invented-secret-value-123")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    home = tmp_path / "fresh"
    text = runner.invoke(app, ["setup", "--root", str(home)])
    assert text.exit_code == 0, text.output
    assert "S2_API_KEY=set" in text.output and "OPENALEX_API_KEY=unset" in text.output
    data = json.loads(runner.invoke(app, ["setup", "--root", str(home), "--json"]).output)
    assert data["scholarly_env"]["S2_API_KEY"] is True
    assert data["scholarly_env"]["OPENALEX_API_KEY"] is False
    assert "invented-secret-value-123" not in text.output + json.dumps(data)


def test_setup_probes_ollama_only_when_asked(runner, tmp_path, monkeypatch):
    import seedgraph.llm.ollama as ollama
    from seedgraph.llm.ollama import Reachability

    calls = []

    def fake(*a, **k):
        calls.append(1)
        return Reachability(ok=True, base_url="http://invented:1", models=["invented:1b"])

    monkeypatch.setattr(ollama, "check_reachable", fake)
    home = tmp_path / "fresh"
    plain = json.loads(runner.invoke(app, ["setup", "--root", str(home), "--json"]).output)
    assert calls == [] and "ollama" not in plain
    probed = runner.invoke(app, ["setup", "--root", str(home), "--probe-ollama", "--json"])
    assert probed.exit_code == 0, probed.output
    assert json.loads(probed.output)["ollama"] == {
        "reachable": True, "base_url": "http://invented:1", "models": ["invented:1b"], "error": None}
    text = runner.invoke(app, ["setup", "--root", str(home), "--probe-ollama"])
    assert "ollama: reachable at http://invented:1" in text.output
