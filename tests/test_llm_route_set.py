"""`seedgraph llm route set` — re-route a task_type to a different profile.

Offline / keyless throughout. Every invocation passes an explicit ``--root`` tmp
home (belt-and-suspenders on top of the autouse ``isolated_home`` fixture) so the
real ``~/.seedgraph`` is never read or written. A read-back is via the loader
functions (no `route list` command — `llm profiles list` already prints the table).
"""

from __future__ import annotations

from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.config.loader import load_global_config, load_project_config
from seedgraph.paths import project_dir

runner = CliRunner()


def _home(tmp_path):
    return tmp_path / "home"


def _set(root, *args):
    return runner.invoke(app, ["llm", "route", "set", *args, "--root", str(root)])


def test_global_set_writes_overlay_and_reroutes(tmp_path):
    root = _home(tmp_path)
    # default note_extraction preferred is local_ollama_default → change is visible.
    res = _set(root, "--task", "note_extraction", "--preferred", "anthropic_api_default")
    assert res.exit_code == 0, res.output
    assert (root / "config.yaml").exists()  # a thin overlay was written
    cfg = load_global_config(root)
    assert cfg.llm.routes["note_extraction"].preferred_profile == "anthropic_api_default"


def test_fallback_preserved_then_cleared(tmp_path):
    root = _home(tmp_path)
    # Establish an explicit fallback.
    res = _set(
        root, "--task", "note_extraction",
        "--preferred", "anthropic_api_default", "--fallback", "local_ollama_default",
    )
    assert res.exit_code == 0, res.output
    assert load_global_config(root).llm.routes["note_extraction"].fallback_profile == "local_ollama_default"

    # Omitting --fallback preserves the current fallback (does not clobber).
    res = _set(root, "--task", "note_extraction", "--preferred", "anthropic_api_default")
    assert res.exit_code == 0, res.output
    assert load_global_config(root).llm.routes["note_extraction"].fallback_profile == "local_ollama_default"

    # --fallback "" clears it to None.
    res = _set(root, "--task", "note_extraction", "--preferred", "anthropic_api_default", "--fallback", "")
    assert res.exit_code == 0, res.output
    assert load_global_config(root).llm.routes["note_extraction"].fallback_profile is None


def test_unknown_task_exits_2_and_writes_nothing(tmp_path):
    root = _home(tmp_path)
    res = _set(root, "--task", "not_a_real_task", "--preferred", "anthropic_api_default")
    assert res.exit_code == 2, res.output
    assert "unknown task" in res.output
    assert not (root / "config.yaml").exists()  # nothing written


def test_unknown_preferred_profile_exits_2_and_writes_nothing(tmp_path):
    root = _home(tmp_path)
    res = _set(root, "--task", "note_extraction", "--preferred", "ghost_profile")
    assert res.exit_code == 2, res.output
    assert "config error" in res.output
    assert not (root / "config.yaml").exists()  # validation raised before any write


def test_project_scope_writes_override_leaving_global_untouched(tmp_path):
    root = _home(tmp_path)
    res = _set(
        root, "--task", "note_extraction",
        "--preferred", "anthropic_api_default", "--project", "proj",
    )
    assert res.exit_code == 0, res.output
    assert (project_dir("proj", root) / "project.yaml").exists()
    assert load_project_config("proj", root).llm.routes["note_extraction"].preferred_profile == "anthropic_api_default"
    assert not (root / "config.yaml").exists()  # global file untouched
