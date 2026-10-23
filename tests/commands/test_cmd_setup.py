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
