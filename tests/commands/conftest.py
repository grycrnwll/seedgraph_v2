"""Shared fixtures for the CLI smoke tests (v3 design §10: CLI tests are smoke tests).

``migrated_home`` is a v3 home built once per session by migrating the synthetic v2
fixture (tests/migrate/fixtures, invented text only), then copied per test so a test
may mutate it. Nothing here touches a real home: every path is under pytest's tmp.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

_MIGRATE_TESTS = Path(__file__).resolve().parents[1] / "migrate"
if str(_MIGRATE_TESTS) not in sys.path:
    sys.path.insert(0, str(_MIGRATE_TESTS))
_loaded = sys.modules.get("fixtures")
if _loaded is not None and not str(getattr(_loaded, "__file__", "") or "").startswith(str(_MIGRATE_TESTS)):
    del sys.modules["fixtures"]  # a different 'fixtures' (e.g. a namespace package) got there first

import fixtures  # noqa: E402

assert str(Path(fixtures.__file__).resolve()).startswith(str(_MIGRATE_TESTS)), fixtures.__file__

from seedgraph.migrate import run_migration  # noqa: E402

NOW = "2026-08-01T12:00:00+00:00"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(scope="session")
def v2_template(tmp_path_factory) -> Path:
    return fixtures.build_v2_home(tmp_path_factory.mktemp("v2tpl") / "v2home").home


@pytest.fixture(scope="session")
def migrated_template(tmp_path_factory, v2_template) -> Path:
    v3 = tmp_path_factory.mktemp("v3tpl") / "home"
    result = run_migration(v2_template, v3, run_id="mig-cli-template", now=NOW,
                           graph_names={"diff_in_diff": "Difference-in-Differences"},
                           process_lister=lambda: [], disk_free=lambda p: 10 ** 15)
    assert result.ok, result.errors
    return v3


@pytest.fixture
def v2_home(tmp_path, v2_template) -> Path:
    dst = tmp_path / "v2home"
    shutil.copytree(v2_template, dst)
    return dst


@pytest.fixture
def migrated_home(tmp_path, migrated_template) -> Path:
    dst = tmp_path / "v3home"
    shutil.copytree(migrated_template, dst)
    return dst


@pytest.fixture
def no_preflight_probes(monkeypatch):
    """The migration preflight lists real processes and real disk space; tests fake both."""
    from seedgraph.migrate import files as mfiles
    from seedgraph.migrate import v2guard

    monkeypatch.setattr(v2guard, "default_process_lister", lambda: [])
    monkeypatch.setattr(mfiles, "default_disk_free", lambda p: 10 ** 15)
