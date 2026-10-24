"""`seedgraph mcp serve` and `seedgraph setup agents` smoke tests (design §7.1, §7.4).

`setup agents` only ever sees a temp user home here (--user-home); nothing touches the
real Claude Code or Codex configuration. Secret-looking values are invented.
"""

from __future__ import annotations

import json
from pathlib import Path

from seedgraph.cli import app

OTHER_SECRET = "sk-invented-OTHER-0123456789"


def test_mcp_serve_runs_the_v3_server_on_the_root_home(runner, tmp_path, monkeypatch):
    from seedgraph.mcp import server

    seen = {}
    monkeypatch.setattr(server, "main", lambda home=None: seen.setdefault("home", home))
    result = runner.invoke(app, ["mcp", "serve", "--root", str(tmp_path / "h")])
    assert result.exit_code == 0, result.output
    assert Path(seen["home"]) == tmp_path / "h"


def test_mcp_serve_has_no_redact_private_flag(runner):
    result = runner.invoke(app, ["mcp", "serve", "--redact-private"])
    assert result.exit_code == 2


def _user_home(tmp_path) -> Path:
    uh = tmp_path / "user"
    uh.mkdir()
    (uh / ".claude.json").write_text(json.dumps(
        {"mcpServers": {"other": {"command": "npx", "env": {"API_KEY": OTHER_SECRET}}}}, indent=2),
        encoding="utf-8")
    return uh


def _setup(runner, tmp_path, uh, *extra):
    return runner.invoke(app, ["setup", "agents", "--user-home", str(uh), "--worktree", str(tmp_path / "wt"),
                               "--root", str(tmp_path / "v3home"), *extra])


def test_setup_agents_shows_the_change_and_writes_nothing(runner, tmp_path):
    uh = _user_home(tmp_path)
    before = (uh / ".claude.json").read_bytes()
    result = _setup(runner, tmp_path, uh, "--client", "all")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "seedgraph_v3" in out and "Nothing has been written" in out
    assert str(tmp_path / "wt" / ".venv" / "Scripts" / "seedgraph-v3.exe") in out
    assert str(tmp_path / "v3home") in out
    assert OTHER_SECRET not in out
    assert (uh / ".claude.json").read_bytes() == before
    assert not (uh / ".claude" / "skills").exists() and not (uh / ".codex").exists()


def test_setup_agents_write_registers_claude_and_installs_skills(runner, tmp_path):
    uh = _user_home(tmp_path)
    result = _setup(runner, tmp_path, uh, "--client", "claude", "--write")
    assert result.exit_code == 0, result.output
    data = json.loads((uh / ".claude.json").read_text(encoding="utf-8"))
    assert data["mcpServers"]["other"]["env"]["API_KEY"] == OTHER_SECRET
    assert data["mcpServers"]["seedgraph_v3"]["args"] == ["mcp", "serve"]
    assert (uh / ".claude" / "skills" / "seedgraph-v3-work" / "SKILL.md").is_file()
    assert not (uh / ".codex").exists()


def test_setup_agents_codex_uses_the_skills_layout(runner, tmp_path):
    uh = _user_home(tmp_path)
    result = _setup(runner, tmp_path, uh, "--client", "codex", "--write")
    assert result.exit_code == 0, result.output
    assert "[mcp_servers.seedgraph_v3]" in (uh / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert (uh / ".codex" / "skills" / "seedgraph-v3" / "SKILL.md").is_file()
    assert (uh / ".codex" / "skills" / "seedgraph-v3-work" / "SKILL.md").is_file()


def test_bare_setup_still_creates_the_home(runner, tmp_path, monkeypatch):
    from seedgraph.conversion import pool

    monkeypatch.setattr(pool, "_nvidia_smi_gpu_count", lambda: 0)
    result = runner.invoke(app, ["setup", "--root", str(tmp_path / "fresh")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "fresh" / "library" / "library.db").is_file()
