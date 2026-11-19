"""§13 Q12: `setup agents` tells the researcher to approve `record_answer` once with "always
allow" (Seedgraph never edits Claude Code's permission settings). Shown for Claude Code only,
with and without --write."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from seedgraph.cli import app


def setup(tmp_path, *extra):
    uh = tmp_path / "user"
    uh.mkdir(exist_ok=True)
    return CliRunner().invoke(app, ["setup", "agents", "--user-home", str(uh), "--worktree", str(tmp_path / "wt"),
                                    "--root", str(tmp_path / "v3home"), *extra]), uh


def test_the_hint_is_shown_for_claude_with_and_without_write(tmp_path):
    for extra in (("--client", "claude"), ("--client", "all"), ("--client", "claude", "--write")):
        res, _ = setup(tmp_path, *extra)
        assert res.exit_code == 0, res.output
        assert "record_answer" in res.output and "always allow" in res.output, extra
        assert "does not edit Claude Code's permission settings" in res.output


def test_the_hint_is_not_shown_for_codex_and_no_permission_is_written(tmp_path):
    res, _ = setup(tmp_path, "--client", "codex")
    assert "always allow" not in res.output
    res, uh = setup(tmp_path, "--client", "claude", "--write")
    assert "permissions" not in json.loads((uh / ".claude.json").read_text(encoding="utf-8"))
    assert not (uh / ".claude" / "settings.json").exists()
